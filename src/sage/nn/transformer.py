
# =============================================================================
# SAT-style transformer building blocks: TransformerBlock (pre-norm, RoPE,
# differential attention, DYT/RMS norm, conformer, GLU-FFN, LayerScale, optional
# AdaLN global-cond), ContinuousTransformer (minimal DiT-capable backbone) and
# TransformerResamplingBlock (strided encoder for discriminators).
# Stripped from stable-audio-tools: no local cond, no flash-attn, no flex_attention,
# no varlen — pure SDPA with optional sliding-window mask. AdaLN global-cond kept
# (DiT timestep conditioning, LATENT_ALIGNMENT_PLAN Fase 6).
# =============================================================================

import math
from functools import reduce
from typing import Callable, Literal, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
from torch.nn.utils.parametrizations import weight_norm

try:
    from dac.model.discriminator import WNConv1d
except ImportError:
    WNConv1d = None  # falls back to plain weight_norm(Conv1d)


# ---------------------------------------------------------------------------
# Channel-folding helpers (also used by pretransforms.py)
# ---------------------------------------------------------------------------

def fold_channels_into_batch(x: torch.Tensor) -> torch.Tensor:
    """[B, C, ...] → [B*C, ...]."""
    return rearrange(x, 'b c ... -> (b c) ...')


def unfold_channels_from_batch(x: torch.Tensor, channels: int) -> torch.Tensor:
    """[B*C, ...] → [B, C, ...]."""
    if channels == 1:
        return x.unsqueeze(1)
    return rearrange(x, '(b c) ... -> b c ...', c=channels)


# ---------------------------------------------------------------------------
# Sequence padding helper
# ---------------------------------------------------------------------------

def _zero_pad_modulo_sequence(x: torch.Tensor, modulo: int, dim: int = -2) -> torch.Tensor:
    """Zero-pad tensor along `dim` so its length is a multiple of `modulo`."""
    length = x.shape[dim]
    pad_len = (modulo - length % modulo) % modulo
    if pad_len == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE — learnable via inv_freq buffer, supports linear interpolation."""

    def __init__(self, dim: int, base: int = 10000, interpolation_factor: float = 1.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.interpolation_factor = interpolation_factor

    def forward_from_seq_len(self, seq_len: int) -> Tuple[torch.Tensor, float]:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        t = t.to(torch.float32) / self.interpolation_factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def _apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    out_dtype = t.dtype
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim = freqs.shape[-1]
    t, freqs = t.to(dtype), freqs.to(dtype)
    freqs = freqs[-t.shape[-2]:, :]
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * freqs.cos() + _rotate_half(t_rot) * freqs.sin()
    return torch.cat((t_rot.to(out_dtype), t_unrot.to(out_dtype)), dim=-1)


# ---------------------------------------------------------------------------
# Normalisation variants
# ---------------------------------------------------------------------------

class DynamicTanh(nn.Module):
    """DynamicTanh (DYT) — learnable tanh norm, stable alternative to LayerNorm."""

    def __init__(self, dim: int, init_alpha: float = 4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * F.tanh(self.alpha * x) + self.beta


class RMSNorm(nn.Module):
    """RMSNorm — bias-free, optional fp32 upcast for stability."""

    def __init__(self, dim: int, eps: float = 1e-5, force_fp32: bool = False, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm requires PyTorch >= 2.4; use manual impl for compat with cineca-ai 2.2.0a0
        if self.force_fp32:
            x_f = x.float()
            out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma.float()
            return out.to(x.dtype)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma


class SATLayerNorm(nn.Module):
    """Bias-less LayerNorm (SAT variant), stable with optional fp32 upcast."""

    def __init__(self, dim: int, eps: float = 1e-5, force_fp32: bool = False, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            return F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps).to(x.dtype)
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)


class LayerScale(nn.Module):
    """LayerScale — per-channel learnable scalar for branch outputs."""

    def __init__(self, dim: int, init_val: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ---------------------------------------------------------------------------
# Feed-Forward (SwiGLU)
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit with configurable activation."""

    def __init__(self, dim_in: int, dim_out: int, activation: Callable):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    """SwiGLU feedforward with zero-init output projection."""

    def __init__(self, dim: int, mult: float = 4.0, no_bias: bool = False, zero_init_output: bool = True, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        self.glu = GLU(dim, inner_dim, nn.SiLU())
        self.out = nn.Linear(inner_dim, dim, bias=not no_bias)
        if zero_init_output:
            nn.init.zeros_(self.out.weight)
            if not no_bias:
                nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.glu(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def _sliding_window_additive_mask(seq_q: int, seq_k: int, w_left: int, w_right: int,
                                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build additive sliding-window mask: 0 in band, -inf outside."""
    ii = torch.arange(seq_q, device=device)
    jj = torch.arange(seq_k, device=device)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -w_left) & (delta <= w_right)
    mask = torch.zeros(seq_q, seq_k, dtype=dtype, device=device)
    return mask.masked_fill(~in_band, float('-inf'))


class Attention(nn.Module):
    """
    Multi-head self-attention with optional differential attention, RoPE,
    qk_norm (dyt / rms / l2 / none) and sliding-window SDPA mask.
    """

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm: Literal['l2', 'rms', 'dyt', 'none'] = 'none',
        qk_norm_eps: float = 1e-6,
        differential: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.num_heads = dim // dim_heads
        self.causal = causal
        self.differential = differential

        mult = 5 if differential else 3
        self.to_qkv = nn.Linear(dim, dim * mult, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm
        if qk_norm == 'rms':
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)
        else:
            self.q_norm = self.k_norm = None

    def _qk_normalize(self, q: torch.Tensor, k: torch.Tensor):
        if self.qk_norm == 'l2':
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm in ('rms', 'dyt'):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: Optional[Tuple] = None,
        flash_attn_sliding_window: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.num_heads

        if self.differential:
            q, k, v, q2, k2 = self.to_qkv(x).chunk(5, dim=-1)
            q, k, v, q2, k2 = [rearrange(t, 'b n (h d) -> b h n d', h=h)
                                for t in (q, k, v, q2, k2)]
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=h) for t in (q, k, v)]
            q2 = k2 = None

        q, k = self._qk_normalize(q, k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q = _apply_rotary_pos_emb(q.float(), freqs.float()).to(v.dtype)
            k = _apply_rotary_pos_emb(k.float(), freqs.float()).to(v.dtype)
            if q2 is not None:
                q2 = _apply_rotary_pos_emb(q2.float(), freqs.float()).to(v.dtype)
                k2 = _apply_rotary_pos_emb(k2.float(), freqs.float()).to(v.dtype)

        attn_mask = None
        if flash_attn_sliding_window is not None:
            wl, wr = flash_attn_sliding_window
            attn_mask = _sliding_window_additive_mask(
                q.shape[2], k.shape[2], wl, wr, q.device, q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              is_causal=(self.causal and attn_mask is None))
        if self.differential:
            out2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=attn_mask,
                                                   is_causal=(self.causal and attn_mask is None))
            out = out - out2

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Conformer module
# ---------------------------------------------------------------------------

class ConformerModule(nn.Module):
    """Lite conformer: pointwise → GLU → depthwise → pointwise."""

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.in_norm = SATLayerNorm(dim)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = SATLayerNorm(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.out_norm = SATLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)                                                  # (B, T, C)
        x = self.glu(x)                                                      # (B, T, C)
        x = rearrange(x, 'b t c -> b c t')
        x = self.pointwise_conv(x)                                           # (B, C, T)
        x = self.depthwise_conv(x)                                           # (B, C, T)
        x = rearrange(x, 'b c t -> b t c')
        x = self.mid_norm(x)                                                 # (B, T, C)
        x = rearrange(x, 'b t c -> b c t')
        x = self.pointwise_conv2(x)                                          # (B, C, T)
        return rearrange(x, 'b c t -> b t c')                               # (B, T, C)


# ---------------------------------------------------------------------------
# TransformerBlock (stripped of global/local conditioning)
# ---------------------------------------------------------------------------

_NORM_MAP = {
    'layer_norm': SATLayerNorm,
    'rms_norm': RMSNorm,
    'dyt': DynamicTanh,
}


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with optional conformer, RoPE, LayerScale,
    and differential attention. Global/local conditioning stripped.
    """

    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        layer_scale: bool = False,
        add_rope: bool = False,
        global_cond_dim: Optional[int] = None,
        norm_type: Literal['layer_norm', 'rms_norm', 'dyt'] = 'dyt',
        attn_kwargs: dict = {},
        ff_kwargs: dict = {},
        norm_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.add_rope = add_rope

        if layer_scale and zero_init_branch_outputs:
            zero_init_branch_outputs = False  # redundant with LayerScale

        norm_cls = _NORM_MAP.get(norm_type, SATLayerNorm)

        self.pre_norm = norm_cls(dim, **norm_kwargs)
        self.self_attn = Attention(
            dim, dim_heads=min(dim_heads, dim), causal=causal,
            zero_init_output=zero_init_branch_outputs, **attn_kwargs
        )
        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.ff_norm = norm_cls(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.conformer = ConformerModule(dim, **norm_kwargs) if conformer else None
        self.conformer_scale = LayerScale(dim) if (layer_scale and conformer) else nn.Identity()

        self.rope = RotaryEmbedding(min(dim_heads, dim) // 2) if add_rope else None

        # AdaLN timestep/global conditioning (DiT-style). Inert unless global_cond_dim
        # is set: a per-block 6-way scale/shift/gate, faithful to stable-audio-tools'
        # adaLN. Modulation sits on top of DyT (SAME §2.2) — norm-agnostic.
        self.global_cond_dim = global_cond_dim
        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6 * dim) / dim ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: Optional[Tuple] = None,
        global_cond: Optional[torch.Tensor] = None,
        self_attention_flash_sliding_window: Optional[List[int]] = None,
        **kwargs,  # absorb unused SAT kwargs (context, etc.)
    ) -> torch.Tensor:
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])  # (T, D)

        if self.global_cond_dim is not None and global_cond is not None:
            # AdaLN-zero path: cond (B, 6*dim) → per-block scale/shift/gate ×2.
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
                self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)  # 6×(B,1,C)

            residual = x                                                      # self-attn w/ adaLN
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb=rotary_pos_emb,
                               flash_attn_sliding_window=self_attention_flash_sliding_window)
            x = x * torch.sigmoid(1 - gate_self)
            x = residual + self.self_attn_scale(x)                           # (B, T, C)

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))             # (B, T, C)

            residual = x                                                     # feed-forward w/ adaLN
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = residual + self.ff_scale(x)                                  # (B, T, C)
            return x

        x = x + self.self_attn_scale(self.self_attn(
            self.pre_norm(x),
            rotary_pos_emb=rotary_pos_emb,
            flash_attn_sliding_window=self_attention_flash_sliding_window,
        ))                                                                   # (B, T, C)

        if self.conformer is not None:
            x = x + self.conformer_scale(self.conformer(x))                 # (B, T, C)

        x = x + self.ff_scale(self.ff(self.ff_norm(x)))                    # (B, T, C)
        return x


class ContinuousTransformer(nn.Module):
    """Minimal continuous transformer with optional AdaLN timestep/global conditioning.

    Reproduces only the subset of stable-audio-tools' ContinuousTransformer needed by
    the flow-matching LatentDiT (LATENT_ALIGNMENT_PLAN Fase 6): in/out projections, a
    global-cond AdaLN path, and a stack of TransformerBlocks. Uses SAME's transformer
    primitives (differential attention + QK-norm + RoPE + DyT + GLU-FFN, zero-init
    branches — the TransformerBlock defaults, §2.2). The heavyweight features of the
    monolith (cross-attend, memory tokens, sliding-window/varlen, abs/sinusoidal
    pos-emb) are intentionally omitted.

    Args:
        dim: transformer width.
        depth: number of TransformerBlocks.
        dim_in/dim_out: optional in/out projection dims (else Identity).
        global_cond_dim: if set, enables AdaLN conditioning on a (B, global_cond_dim)
            vector (e.g. a Fourier timestep embedding) via a Linear→SiLU→Linear(6*dim).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        dim_heads: int = 64,
        global_cond_dim: Optional[int] = None,
        rotary_pos_emb: bool = True,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        norm_type: Literal['layer_norm', 'rms_norm', 'dyt'] = 'dyt',
        **block_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6),
            )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, dim_heads=dim_heads, global_cond_dim=global_cond_dim,
                zero_init_branch_outputs=zero_init_branch_outputs, conformer=conformer,
                norm_type=norm_type, add_rope=rotary_pos_emb, **block_kwargs,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, global_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.project_in(x)                                             # (B, T, dim)
        g = None
        if self.global_cond_embedder is not None and global_cond is not None:
            g = self.global_cond_embedder(global_cond)                    # (B, 6*dim)
        for layer in self.layers:                                          # each block builds/applies its own RoPE
            x = layer(x, global_cond=g)                                   # (B, T, dim)
        return self.project_out(x)                                        # (B, T, dim_out)


# ---------------------------------------------------------------------------
# TransformerResamplingBlock (encoder mode only)
# ---------------------------------------------------------------------------

def _make_wn_conv1d(in_ch: int, out_ch: int, kernel_size: int = 1, **kwargs) -> nn.Module:
    """Weight-normalised Conv1d — uses DAC's WNConv1d when available."""
    if WNConv1d is not None and kernel_size == 1:
        return WNConv1d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)
    return weight_norm(conv)


class TransformerResamplingBlock(nn.Module):
    """
    Strided transformer encoder: maps (B, C_in, T) → (B, C_out, T//stride).

    Each group of `stride` input tokens is concatenated with one learned
    `new_token`, processed by a TransformerBlock stack, and the new_token
    output is used as the single output token for that group.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sliding_window: Optional[List[int]] = None,
        transformer_depth: int = 3,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        dim_heads: int = 64,
        differential: bool = True,
        ff_mult: float = 3.0,
        dyt: bool = True,
        mask_noise: float = 0.0,
        mapping_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.checkpointing = checkpointing
        self.mask_noise = mask_noise

        # Stride + 1 tokens per segment (stride input tokens + 1 new token)
        self.sub_chunk_size = stride + 1

        # Sliding window in sequence space: scale by (stride+1) since each
        # input token maps to (stride+1) sequence tokens after appending new_token
        self.sliding_window_seq: Optional[List[int]] = None
        if sliding_window is not None:
            self.sliding_window_seq = [w * (stride + 1) for w in sliding_window]

        self.mapping = (_make_wn_conv1d(in_channels, out_channels, 1, bias=mapping_bias)
                        if in_channels != out_channels else nn.Identity())

        norm_type = 'dyt' if dyt else 'rms_norm'
        blocks = []
        for _ in range(transformer_depth):
            blocks.append(TransformerBlock(
                out_channels,
                dim_heads=dim_heads,
                causal=False,
                zero_init_branch_outputs=True if not layer_scale else False,
                norm_type=norm_type,
                conformer=conformer,
                layer_scale=layer_scale,
                add_rope=True,
                attn_kwargs={
                    'qk_norm': 'dyt' if dyt else 'rms',
                    'qk_norm_eps': 1e-3,
                    'differential': differential,
                },
                ff_kwargs={'mult': ff_mult, 'no_bias': False},
                norm_kwargs={'eps': 1e-3},
            ))
        self.transformers = nn.ModuleList(blocks)

        # One learned token per stride group (appended to each group)
        self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, 1, out_channels))

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
        Returns:
            out: (B, C_out, T//stride)
            features (optional): list of intermediate tensors
        """
        B = x.shape[0]
        features: List[torch.Tensor] = []

        # Pad time to multiple of stride and apply channel mapping
        x = _zero_pad_modulo_sequence(x, self.stride, dim=-1)               # (B, C_in, T')
        x = self.mapping(x)                                                  # (B, C_out, T')

        x = rearrange(x, 'b d t -> b t d')                                  # (B, T', C_out)
        if return_features:
            features.append(x)

        # Split into non-overlapping segments of length `stride`
        x = rearrange(x, 'b (n s) d -> (b n) s d', s=self.stride)          # (B*N, stride, C_out)

        # Append one new_token per segment
        new_toks = self.new_tokens.expand(x.shape[0], 1, -1)
        if self.mask_noise > 0 and self.training:
            new_toks = new_toks + torch.randn_like(new_toks) * self.mask_noise
        x = torch.cat([x, new_toks], dim=1)                                 # (B*N, stride+1, C_out)

        # Fold segments back into batch sequence for transformer
        x = rearrange(x, '(b n) c d -> b (n c) d', b=B)                    # (B, N*(stride+1), C_out)

        # Run transformer blocks with sliding-window attention
        for layer in self.transformers:
            x = layer(x, self_attention_flash_sliding_window=self.sliding_window_seq)
            if return_features:
                features.append(x)                                           # (B, N*(stride+1), C_out)

        # Extract the last token from each segment (the new_token position)
        x = rearrange(x, 'b (n c) d -> (b n) c d', c=self.sub_chunk_size)  # (B*N, stride+1, C_out)
        x = x[:, -1:, :]                                                    # (B*N, 1, C_out) — new_token output
        x = rearrange(x, '(b n) c d -> b d (n c)', b=B)                    # (B, C_out, N)

        if return_features:
            return x, features
        return x
