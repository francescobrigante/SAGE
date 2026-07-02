# ===============================================================
# Swin Transformer V2: window-based multi-head self-attention.
# Implements cosine attention with learnable per-head temperature
# (logit_scale) and continuous relative position bias via a small
# Log-CPB MLP on log-transformed coordinates.
# ===============================================================

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ar_spectra.blocks.conv.normed import NormLinear


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA / SW-MSA) with Swin V2 cosine bias.

    Real path  (is_complex=False): standard cosine attention — F.normalize(q) @ F.normalize(k)ᵀ.
    Complex path (is_complex=True): hermitian cosine — Re(q̂ · k̂*) where q̂ = q/|q|.
        Attention scores are always real → softmax unchanged.
        attn @ V is split into re/im to avoid mixed-dtype matmul.
        cpb_mlp, logit_scale, and relative_position_bias always remain real.

    Args:
        dim: Token channel dimension.
        window_size: (Wh, Ww) height and width of the attention window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add learnable bias to Q and V projections (K bias omitted, V2 convention).
        attn_drop: Dropout rate on attention weights.
        proj_drop: Dropout rate on the output projection.
        pretrained_window_size: Window size used during pre-training for Log-CPB coordinate
            normalisation. None or [0, 0] falls back to window_size.
        is_complex: If True, uses NormLinear with complex64 dtype and hermitian cosine attention.
    """

    def __init__(
        self,
        dim: int,
        window_size,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size=None,
        is_complex: bool = False,
        attention_variant: str = "baseline",
        xsa_eps: float = 1.0e-6,
        xsa_strength: float = 1.0,
    ) -> None:

        super().__init__()
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]          # no pre-training transfer by default
        if attention_variant not in {"baseline", "xsa"}:
            raise ValueError(
                f"Unsupported attention_variant={attention_variant!r}; expected 'baseline' or 'xsa'"
            )

        self.dim = dim
        self.window_size = window_size                                  # (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.is_complex = is_complex
        self.attention_variant = attention_variant
        self.xsa_eps = float(xsa_eps)
        self.xsa_strength = float(xsa_strength)

        # Per-head learnable temperature — always real (scalar, device-agnostic)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # Log-CPB MLP: 2-D coordinate → 512 hidden → num_heads bias values
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # -----------------------------------------------------------------
        # Relative coordinate table (1, 2Wh-1, 2Ww-1, 2) — normalised + log-scaled
        # -----------------------------------------------------------------
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0).contiguous().unsqueeze(0)                # (1, 2Wh-1, 2Ww-1, 2)
        )
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            
        relative_coords_table *= 8                                     # normalise to [-8, 8]
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # -----------------------------------------------------------------
        # Pair-wise relative position index: (Wh*Ww, Wh*Ww)
        # Used to index into relative_coords_table via cpb_mlp.
        # -----------------------------------------------------------------
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))     # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                      # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1           # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)              # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection — no bias here; Q and V biases are added separately (Swin V2 convention)
        # NormLinear with norm='none' is mathematically identical to nn.Linear;
        # is_complex=True sets dtype=complex64 on the weight matrix.
        self.qkv = NormLinear(dim, dim * 3, bias=False, is_complex=is_complex)

        # Q and V learnable biases (K bias = 0 per Swin V2).
        # dtype matches token dtype so torch.cat() in forward is consistent.
        _bias_dtype = torch.complex64 if is_complex else torch.float32
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
            self.v_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
        # Output projection — bias=True (same as original nn.Linear default)
        self.proj = NormLinear(dim, dim, bias=True, is_complex=is_complex)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def _apply_xsa(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Remove each token output's projection onto its own value vector."""
        if self.xsa_strength == 0.0:
            return y
        if y.is_complex():
            den = v.abs().square().sum(dim=-1, keepdim=True).clamp_min(self.xsa_eps)
            coeff = (y * v.conj()).sum(dim=-1, keepdim=True) / den
            return y - self.xsa_strength * coeff * v

        v_unit = F.normalize(v, dim=-1, eps=self.xsa_eps)
        coeff = (y * v_unit).sum(dim=-1, keepdim=True)
        return y - self.xsa_strength * coeff * v_unit

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Compute windowed cosine self-attention with Log-CPB relative position bias.

        Args:
            x: (num_windows * B, N, C) flattened window tokens.
            mask: (num_windows, N, N) cyclic-shift mask for SW-MSA, or None for W-MSA.

        Returns:
            (num_windows * B, N, C)
        """
        B_, N, C = x.shape                                             # (num_windows * B, N,  C)

        # Build combined Q/V bias (K bias = 0 per Swin V2 convention).
        # torch.zeros_like preserves dtype → works for both float32 and complex64.
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),   # K bias = 0
                self.v_bias,
            ))

        # Apply QKV projection then add bias separately (avoids accessing internal weight)
        qkv = self.qkv(x)                                             # (num_windows * B, N,  3C)
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                             # each (num_windows * B, num_heads, N, C/num_heads)

        # --- Attention scores (always real → softmax unchanged) ---
        if self.is_complex:
            # Hermitian cosine: score(i,j) = Re(q̂ᵢ · k̂ⱼ*) ∈ [−1, 1]
            # Normalise by magnitude only — the phase pattern drives attention,
            # not the energy. Phase-coherent token pairs (harmonics, transients)
            # score high regardless of loudness.
            q_norm = q / q.abs().clamp(min=1e-6)                      # (nW*B, h, N, D) complex
            k_norm = k / k.abs().clamp(min=1e-6)                      # (nW*B, h, N, D) complex
            # Re(A @ conj(B)ᵀ) = Re(A)@Re(B)ᵀ + Im(A)@Im(B)ᵀ
            # 2 real SGEMM instead of 1 complex ZGEMM (≡ 4 SGEMM) → 2× faster here.
            attn = q_norm.real @ k_norm.real.mT + q_norm.imag @ k_norm.imag.mT  # (nW*B, h, N, N) float
        else:
            # Standard real cosine attention (Swin V2)
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)  # (nW*B, h, N, N)

        # logit_scale is always real (per-head temperature scalar)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale                                      # (num_windows * B, num_heads, N, N)

        # Log-CPB relative position bias -> (num_heads, N, N) -> broadcast over batch
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)    # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)             # (num_windows * B, num_heads, N, N)

        # SW-MSA: add cyclic-shift mask to prevent cross-region attention
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)                                      # (num_windows * B, num_heads, N, N)
        attn = self.attn_drop(attn)

        # attn is always real; V may be complex.
        # PyTorch does not broadcast real @ complex automatically, so split re/im.
        if self.is_complex:
            # `torch.complex` does not accept bf16/half. Under autocast (bf16-mixed)
            # the matmul output is bf16 → we upcast both parts to fp32 before
            # rebuilding the complex tensor. With fp32 precision this is a no-op.
            out_re = (attn @ v.real).to(torch.float32)
            out_im = (attn @ v.imag).to(torch.float32)
            x = torch.complex(out_re, out_im)                         # (nW*B, h, N, D) complex
        else:
            x = attn @ v                                               # (nW*B, h, N, D) float
        if self.attention_variant == "xsa":
            x = self._apply_xsa(x, v)

        x = x.transpose(1, 2).reshape(B_, N, C)                       # (num_windows * B, N,  C)
        x = self.proj(x)                                               # (num_windows * B, N,  C)
        x = self.proj_drop(x)                                          # (num_windows * B, N,  C)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )
