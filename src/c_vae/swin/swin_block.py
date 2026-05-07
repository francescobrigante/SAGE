# ===============================================================
# Swin Transformer V2: atomic computation block.
# Contains Mlp (2-layer FFN) and SwinTransformerBlock
# (W-MSA / SW-MSA + FFN with pre-norm residuals).
# Optional fused CUDA window kernels are loaded at import time.
# ===============================================================

from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.layers import to_2tuple

from ar_spectra.utils.console import ok, warn
from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.blocks.activations import get_activation
from .windowing import window_partition, window_reverse
from .attention import WindowAttention
from .utils import make_norm, make_drop_path

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


# --------------------------------------------------------------------------
# SwinTransformerBlock
# --------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """Swin Transformer V2 block: pre-norm W-MSA or SW-MSA + FFN with residuals.

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
        norm_layer: Normalisation class. Default: nn.LayerNorm.
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
    ) -> None:

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.is_complex = is_complex

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
        )
        # ComplexSafeDropPath when is_complex=True, timm DropPath otherwise.
        # Both are nn.Identity when drop_path == 0.
        self.drop_path = make_drop_path(drop_path, is_complex)
        self.norm2 = make_norm(dim, is_complex)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            is_complex=is_complex,
            complex_activation=complex_activation,
        )

        # Pre-compute SW-MSA attention mask once and register as a non-parameter buffer.
        # The mask uses -100.0 for cross-region pairs so softmax -> 0 after exp.
        if max(self.shift_size) > 0:
            H, W = self.input_resolution
            wh, ww = self.window_size
            sh, sw = self.shift_size
            img_mask = torch.zeros((1, H, W, 1))                       # (1, H, W, 1)
            h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
            w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # (nW, wh, ww, 1)
            mask_windows = mask_windows.view(-1, wh * ww)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply W-MSA or SW-MSA + FFN with pre-norm residual connections.

        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape                                              # (B, H*W, C)
        assert L == H * W, f"token count {L} != H*W = {H*W}"

        wh, ww = self.window_size
        sh, sw = self.shift_size

        shortcut = x
        x = x.view(B, H, W, C)                                        # (B, H, W,  C)

        # Cyclic shift + window partition
        if max(self.shift_size) > 0:
            # not cuda
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, wh, ww, C)
            # cuda — complex64: reinterpret as float32 (B,H,W,2C), run kernel, cast back.
            # Safe because the kernel is pure memory scatter/gather (no arithmetic) and
            # complex64 is stored as interleaved float32 pairs, so Re/Im pairing is preserved.
            elif self.is_complex:
                x_f = x.view(torch.float32)                                  # (B, H, W, 2C)
                out_f = WindowProcess.apply(x_f, B, H, W, 2 * C, -sh, -sw, wh, ww)
                x_windows = out_f.view(torch.complex64)                      # (nW*B, wh, ww, C)
            # cuda — real float32: pass tensor directly
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -sh, -sw, wh, ww)

        else:
            x_windows = window_partition(x, self.window_size)         # (nW*B, wh, ww, C)

        x_windows = x_windows.view(-1, wh * ww, C)                    # (nW*B, wh*ww, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)      # (nW*B, wh*ww, C)

        # Reverse window partition + reverse cyclic shift
        attn_windows = attn_windows.view(-1, wh, ww, C)
        if max(self.shift_size) > 0:
            # not cuda
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
                x = torch.roll(shifted_x, shifts=(sh, sw), dims=(1, 2))
            # cuda — complex64 view trick (same rationale as forward path above)
            elif self.is_complex:
                aw_f = attn_windows.view(torch.float32)                      # (nW*B, wh, ww, 2C)
                out_f = WindowProcessReverse.apply(aw_f, B, H, W, 2 * C, sh, sw, wh, ww)
                x = out_f.view(torch.complex64)                              # (B, H, W, C)
            # cuda — real float32
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, sh, sw, wh, ww)
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        x = x.view(B, H * W, C)                                       # (B, H*W, C)

        # Res-Post-LN (Swin V2): norm applied to branch output, residual added after
        x = shortcut + self.drop_path(self.norm1(x))                  # (B, H*W, C)
        x = x + self.drop_path(self.norm2(self.mlp(x)))               # (B, H*W, C)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )
