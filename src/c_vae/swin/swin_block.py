# ===============================================================
# Swin Transformer V2: atomic computation block.
# Contains Mlp (2-layer FFN) and SwinTransformerBlock
# (W-MSA / SW-MSA + FFN with configurable residual norm placement).
# Optional fused CUDA window kernels are loaded at import time.
# ===============================================================

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple

from ar_spectra.utils.console import ok, warn
from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.blocks.activations import get_activation
from .windowing import window_partition, window_reverse
from .attention import WindowAttention
from .utils import make_norm, make_drop_path

if TYPE_CHECKING:                       # avoids a circular import at runtime
    from .varlen import VarlenConfig

# -----------------------------------------------------------------
# Optional fused CUDA window kernels
# -----------------------------------------------------------------
try:
    from .cuda_kernels import WindowProcess, WindowProcessReverse, FUSED_WINDOW_AVAILABLE
    ok("Fused CUDA window-process kernels loaded.", prefix="CUDA")
except (ImportError, ModuleNotFoundError):
    WindowProcess = None
    WindowProcessReverse = None
    FUSED_WINDOW_AVAILABLE = False
    warn("Fused CUDA window kernels unavailable: using torch.roll + window_partition.", prefix="CUDA")


# --------------------------------------------------------------------------
# Mlp
# --------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP with GELU activation (FFN inside SwinTransformerBlock).

    With is_complex=False uses nn.Linear + nn.GELU (identical to original).
    With is_complex=True uses NormLinear(is_complex=True) + a phase-equivariant
    complex activation (default: ComplexGELU1d — gate on |x|, phase preserved).

    Args:
        in_features: Input channel dimension.
        hidden_features: Hidden layer width (defaults to in_features).
        out_features: Output width (defaults to in_features).
        act_layer: Activation class used when is_complex=False. Default: nn.GELU.
        drop: Dropout rate applied after each linear layer.
        is_complex: If True, switches to NormLinear + complex activation.
        complex_activation: Name of the complex activation (eulero.nn registry).
            Default ``"ComplexGELU1d"`` (phase-equivariant). Use ``"CGeLU"`` to
            reproduce the legacy split-GELU behaviour (ablation only).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
    ) -> None:

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.is_complex = is_complex

        if is_complex:
            self.fc1 = NormLinear(in_features, hidden_features, is_complex=True)
            self.act = get_activation(complex_activation, is_complex=True, channels=hidden_features)
            self.fc2 = NormLinear(hidden_features, out_features, is_complex=True)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)

        # Skip the Dropout module entirely when drop=0 — even if F.dropout has a
        # fast-path for p=0, this removes the module from the autograd graph too.
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)    # (*, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)    # (*, out_features)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU FFN for Swin blocks.

    The real path follows the standard SwiGLU form:
    Linear(in -> 2 * hidden), split into value/gate, SiLU gate, then
    Linear(hidden -> out). The complex path uses a real magnitude gate and is
    kept intentionally small for real-Swin-first experiments.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        is_complex: bool = False,
        complex_swiglu_gate: str = "magnitude_silu",
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.is_complex = is_complex
        self.complex_swiglu_gate = complex_swiglu_gate

        if hidden_features <= 0:
            raise ValueError(f"hidden_features must be positive, got {hidden_features}")
        if is_complex and complex_swiglu_gate != "magnitude_silu":
            raise ValueError(
                "complex SwiGLU currently supports only complex_swiglu_gate='magnitude_silu'"
            )

        linear_cls = NormLinear if is_complex else nn.Linear
        linear_kwargs = {"is_complex": True} if is_complex else {}
        self.fc1 = linear_cls(in_features, 2 * hidden_features, **linear_kwargs)
        self.fc2 = linear_cls(hidden_features, out_features, **linear_kwargs)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.fc1(x).chunk(2, dim=-1)
        if self.is_complex:
            gate = F.silu(gate.abs())
        else:
            gate = F.silu(gate)
        x = u * gate
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _round_up_to_multiple(value: int, multiple_of: Optional[int]) -> int:
    if multiple_of is None:
        return value
    if multiple_of <= 0:
        raise ValueError(f"swiglu_multiple_of must be positive or None, got {multiple_of}")
    return ((value + multiple_of - 1) // multiple_of) * multiple_of


def make_mlp(
    *,
    in_features: int,
    mlp_ratio: float,
    act_layer=nn.GELU,
    drop: float = 0.0,
    is_complex: bool = False,
    complex_activation: str = "ComplexGELU1d",
    mlp_type: str = "mlp",
    swiglu_hidden_ratio: Optional[float] = None,
    swiglu_multiple_of: Optional[int] = None,
    complex_swiglu_gate: str = "magnitude_silu",
) -> nn.Module:
    """Build the block FFN.

    ``mlp_type='mlp'`` intentionally returns the original Mlp with the original
    hidden-width formula, preserving baseline module structure and behavior.
    """
    if mlp_type == "mlp":
        return Mlp(
            in_features=in_features,
            hidden_features=int(in_features * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            is_complex=is_complex,
            complex_activation=complex_activation,
        )
    if mlp_type == "swiglu":
        if swiglu_hidden_ratio is None:
            hidden_features = int(in_features * mlp_ratio * 2 / 3)
        else:
            hidden_features = int(in_features * swiglu_hidden_ratio)
        hidden_features = _round_up_to_multiple(hidden_features, swiglu_multiple_of)
        return SwiGLU(
            in_features=in_features,
            hidden_features=hidden_features,
            drop=drop,
            is_complex=is_complex,
            complex_swiglu_gate=complex_swiglu_gate,
        )
    raise ValueError(f"Unsupported mlp_type={mlp_type!r}; expected 'mlp' or 'swiglu'")


# --------------------------------------------------------------------------
# SwinTransformerBlock
# --------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """Swin Transformer V2 block with configurable residual norm placement.

    Even-indexed blocks within a stage use W-MSA (shift_size = 0);
    odd-indexed blocks use SW-MSA (shift_size = window_size//2).

    Args:
        dim: Token channel dimension.
        input_resolution: (H, W) spatial grid for this stage.
        num_heads: Number of attention heads.
        window_size: Local window size — int for square, (wh, ww) for rect.
            Clamped per-dimension to input_resolution when the grid is smaller.
        shift_size: Cyclic shift offset — 0 = W-MSA, window_size // 2 = SW-MSA.
            Int or (sh, sw) tuple; stored as tuple after per-dim collapse guard.
        mlp_ratio: FFN hidden-dim multiplier.
        qkv_bias: Learnable bias on Q and V projections.
        drop: Dropout rate on FFN outputs and projections.
        attn_drop: Dropout rate on attention weights.
        drop_path: Stochastic depth rate for this block.
        act_layer: FFN activation. Default: nn.GELU.
        norm_placement: ``"res_post"`` preserves the current Swin V2
            branch-output norm. ``"pre"`` runs the ablation with pre-norm
            residual branches.
        pretrained_window_size: Window size used in pre-training (Log-CPB normalisation).
        fused_window_process: Use fused CUDA kernel for roll+partition (CUDA only).
        complex_activation: Name of the phase-equivariant complex activation used in Mlp
            when is_complex=True. Default ``"ComplexGELU1d"``. Passed to eulero.nn registry.
    """

    def __init__(
        self,
        dim: int,
        input_resolution,
        num_heads: int,
        window_size: Union[int, Tuple[int, int]] = 7,
        shift_size: Union[int, Tuple[int, int]] = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        pretrained_window_size: int = 0,
        fused_window_process: bool = False,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
        mlp_type: str = "mlp",
        swiglu_hidden_ratio: Optional[float] = None,
        swiglu_multiple_of: Optional[int] = None,
        complex_swiglu_gate: str = "magnitude_silu",
        attention_variant: str = "baseline",
        xsa_eps: float = 1.0e-6,
        xsa_strength: float = 1.0,
        norm_placement: str = "res_post",
    ) -> None:

        super().__init__()
        if norm_placement not in {"res_post", "pre"}:
            raise ValueError(
                f"Unsupported norm_placement={norm_placement!r}; expected 'res_post' or 'pre'"
            )
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.is_complex = is_complex
        self.mlp_type = mlp_type
        self.attention_variant = attention_variant
        self.norm_placement = norm_placement

        # Fused CUDA kernels (roll + window_partition in one pass) work for both real
        # and complex inputs. For complex64, forward() reinterprets the tensor as
        # float32 (B,H,W,2C) before calling the kernel and casts back to complex64
        # afterward — safe because the kernel is pure memory scatter/gather with no
        # arithmetic, and complex64 is stored as interleaved float32 pairs in memory.
        self.fused_window_process = fused_window_process and FUSED_WINDOW_AVAILABLE

        # Per-dimension collapse guard: clamp each window dim independently when the
        # grid is smaller than the window (e.g. at the deepest encoder/decoder stage).
        wh, ww = to_2tuple(window_size)
        sh, sw = to_2tuple(shift_size)
        H, W = self.input_resolution
        if H <= wh:
            sh, wh = 0, H
        if W <= ww:
            sw, ww = 0, W
        self.window_size: Tuple[int, int] = (wh, ww)   # always stored as tuple
        self.shift_size: Tuple[int, int] = (sh, sw)    # always stored as tuple
        assert 0 <= sh < wh and 0 <= sw < ww, "shift_size must be in [0, window_size) per dimension"

        self.norm1 = make_norm(dim, is_complex)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            is_complex=is_complex,
            attention_variant=attention_variant,
            xsa_eps=xsa_eps,
            xsa_strength=xsa_strength,
        )
        # ComplexSafeDropPath when is_complex=True, timm DropPath otherwise.
        # Both are nn.Identity when drop_path == 0.
        self.drop_path = make_drop_path(drop_path, is_complex)
        self.norm2 = make_norm(dim, is_complex)
        self.mlp = make_mlp(
            in_features=dim,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop,
            is_complex=is_complex,
            complex_activation=complex_activation,
            mlp_type=mlp_type,
            swiglu_hidden_ratio=swiglu_hidden_ratio,
            swiglu_multiple_of=swiglu_multiple_of,
            complex_swiglu_gate=complex_swiglu_gate,
        )

        # We will compute the SW-MSA mask dynamically and cache it in forward()
        self._attn_mask_cache = {}

        # Variable-length inference: multi-phase attention on collapsed blocks.
        # None = disabled (default) → forward() is bit-identical to the original.
        # Set via c_vae.swin.varlen.enable_varlen(); see varlen.py for the rationale.
        self.varlen: Optional["VarlenConfig"] = None   # multi-phase config, or None when off
        self._varlen_weight_cache = {}                 # (W, device, dtype) → (P, W) combination weights

    def _varlen_active(self, time_tokens: int) -> bool:
        """True when multi-phase attention should replace the single-phase branch.

        Only fires when a config is attached *and* the runtime time axis is longer
        than the window: at the training length the window already spans the whole
        axis, so every phase would degenerate and the extra passes would only add
        wrap-around artefacts. Keeping this guard makes every variant bit-identical
        to the baseline at the training resolution.
        """
        return self.varlen is not None and time_tokens > self.window_size[1]

    def _multiphase_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention branch at several grid phases and combine per token.

        The deepest encoder/decoder stage has its shift frozen to 0 by the collapse
        guard above, so at inference on audio longer than the training segment the
        window partition tiles the time axis into disjoint blocks with nothing to
        bridge the seams. Re-running the same branch with the grid translated by
        each phase gives every token a choice of contexts; the per-token weights
        (see varlen.phase_weights) favour the phase where the token sits farthest
        from its attention group's edge.

        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C)
        """
        from .varlen import phase_weights

        H = self.input_resolution[0]
        B, L, C = x.shape                                              # (B, H*W, C)
        W = L // H
        cfg = self.varlen

        cache_key = (W, x.device, x.dtype)
        weights = self._varlen_weight_cache.get(cache_key)
        if weights is None:
            weights = phase_weights(W, self.window_size[1], cfg).to(x.device, x.dtype)  # (P, W)
            self._varlen_weight_cache[cache_key] = weights

        acc = None
        for i, phase in enumerate(cfg.phases):
            y = self._attention_branch(x, shift_size=(0, phase))       # (B, H*W, C)
            y = y.view(B, H, W, C) * weights[i].view(1, 1, W, 1)       # (B, H, W, C)
            acc = y if acc is None else acc + y                        # (B, H, W, C)
        return acc.view(B, L, C)                                       # (B, H*W, C)

    def _attention_branch(
        self,
        x: torch.Tensor,
        shift_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Apply W-MSA or SW-MSA to a normalized or raw residual branch.

        Args:
            x: (B, H*W, C)
            shift_size: Overrides ``self.shift_size`` for this call only. Used by
                multi-phase variable-length inference to sweep the grid phase
                without mutating module state. None = use the block's own shift.

        Returns:
            (B, H*W, C)
        """
        H = self.input_resolution[0]
        B, L, C = x.shape                                              # (B, H*W, C)
        assert L % H == 0, f"Token count {L} not divisible by H={H}"
        W = L // H

        wh, ww = self.window_size
        sh, sw = self.shift_size if shift_size is None else shift_size

        pad_l = pad_t = 0
        pad_r = (ww - W % ww) % ww
        pad_b = (wh - H % wh) % wh

        Hp = H + pad_b
        Wp = W + pad_r

        # cached attention mask uses -100.0 for cross-region pairs so softmax -> 0 after exp.
        if max(sh, sw) > 0:
            # Cache the mask already ON the target device, keyed by (grid, device):
            # the SW-MSA mask is constant per grid, so building + H2D-copying it on
            # every forward was a needless per-shifted-block transfer (a sync point).
            # Keying on device keeps DDP/multi-device correct (each replica caches its
            # own). This mask only feeds the softmax in WindowAttention and never the
            # fused CUDA window kernels, so caching it cannot affect them.
            # The shift is part of the key: the mask depends on it, and multi-phase
            # inference calls this method with several shifts on the same block.
            cache_key = (Hp, Wp, sh, sw, x.device)
            attn_mask = self._attn_mask_cache.get(cache_key)
            if attn_mask is None:
                img_mask = torch.zeros((1, Hp, Wp, 1))
                h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
                w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, wh * ww)
                mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = (
                    mask.masked_fill(mask != 0, -100.0)
                        .masked_fill(mask == 0, 0.0)
                        .to(x.device)
                )
                self._attn_mask_cache[cache_key] = attn_mask
        else:
            attn_mask = None

        x = x.view(B, H, W, C)                                        # (B, H, W,  C)
        if pad_r > 0 or pad_b > 0:
            import torch.nn.functional as F
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # Cyclic shift + window partition
        if max(sh, sw) > 0:
            # not cuda
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, wh, ww, C)
            # cuda — complex64: reinterpret as float32 (B,Hp,Wp,2C), run kernel, cast back.
            # Safe because the kernel is pure memory scatter/gather (no arithmetic) and
            # complex64 is stored as interleaved float32 pairs, so Re/Im pairing is preserved.
            elif self.is_complex:
                x_f = x.view(torch.float32)                                  # (B, Hp, Wp, 2C)
                out_f = WindowProcess.apply(x_f, B, Hp, Wp, 2 * C, -sh, -sw, wh, ww)
                x_windows = out_f.view(torch.complex64)                      # (nW*B, wh, ww, C)
            # cuda — real: kernel supports float32/float16 but not bfloat16; cast if needed
            else:
                orig_dtype = x.dtype
                if orig_dtype == torch.bfloat16:
                    x_windows = WindowProcess.apply(x.float(), B, Hp, Wp, C, -sh, -sw, wh, ww).bfloat16()
                else:
                    x_windows = WindowProcess.apply(x, B, Hp, Wp, C, -sh, -sw, wh, ww)

        else:
            x_windows = window_partition(x, self.window_size)         # (nW*B, wh, ww, C)

        x_windows = x_windows.view(-1, wh * ww, C)                    # (nW*B, wh*ww, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)      # (nW*B, wh*ww, C)

        # Reverse window partition + reverse cyclic shift
        attn_windows = attn_windows.view(-1, wh, ww, C)
        if max(sh, sw) > 0:
            # not cuda
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)
                x = torch.roll(shifted_x, shifts=(sh, sw), dims=(1, 2))
            # cuda — complex64 view trick (same rationale as forward path above)
            elif self.is_complex:
                aw_f = attn_windows.view(torch.float32)                      # (nW*B, wh, ww, 2C)
                out_f = WindowProcessReverse.apply(aw_f, B, Hp, Wp, 2 * C, sh, sw, wh, ww)
                x = out_f.view(torch.complex64)                              # (B, Hp, Wp, C)
            # cuda — real: kernel supports float32/float16 but not bfloat16; cast if needed
            else:
                orig_dtype = attn_windows.dtype
                if orig_dtype == torch.bfloat16:
                    x = WindowProcessReverse.apply(attn_windows.float(), B, Hp, Wp, C, sh, sw, wh, ww).bfloat16()
                else:
                    x = WindowProcessReverse.apply(attn_windows, B, Hp, Wp, C, sh, sw, wh, ww)
        else:
            x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)
            
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H * W, C)                                       # (B, H*W, C)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply W-MSA or SW-MSA + FFN.

        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C)
        """

        # Multi-phase only when explicitly enabled AND the time axis is longer than
        # the window; otherwise this is the original single-phase branch, unchanged.
        attn_branch = (
            self._multiphase_attention
            if self._varlen_active(x.shape[1] // self.input_resolution[0])
            else self._attention_branch
        )

        if self.norm_placement == "pre":
            x = x + self.drop_path(attn_branch(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))           # (B, H*W, C)
        else:
            # Res-Post-LN (Swin V2): norm applied to branch output, residual added after
            x = x + self.drop_path(self.norm1(attn_branch(x)))        # (B, H*W, C)
            x = x + self.drop_path(self.norm2(self.mlp(x)))           # (B, H*W, C)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, "
            f"norm_placement={self.norm_placement}"
        )
