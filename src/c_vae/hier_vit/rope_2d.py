# ===============================================================
# 2-D Rotary Position Embedding (RoPE) for hierarchical ViT.
#
# Splits head_dim in half: the first half carries the y-axis (freq)
# rotation, the second half carries the x-axis (time) rotation.
# Each half is a standard 1-D RoPE with its own base θ.
#
# Designed for spectrograms: theta_y (freq) defaults lower than
# theta_x (time) because the freq axis has finer local periodicity.
#
# For complex tokens, RoPE is applied independently to .real and
# .imag — the rotation is a phase-equivariant operation in the
# real representation, leaving the Hermitian inner-product score
# of an attention layer still relative-position-aware.
# ===============================================================

from typing import Tuple

import torch
import torch.nn as nn


def _precompute_cos_sin(
    grid_size: Tuple[int, int],
    head_dim: int,
    theta_y: float,
    theta_x: float,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables of shape ``(H*W, head_dim)``.

    head_dim is split in half: indices ``[0, D/2)`` rotate by y position,
    ``[D/2, D)`` by x position. Within each half, consecutive pairs
    ``(2k, 2k+1)`` share the same frequency band ``θ_k``.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2-D RoPE")

    H, W = grid_size
    d_half = head_dim // 2                                          # dims per axis
    n_freq = d_half // 2                                            # rotation pairs per axis

    # Per-axis frequency bands θ_k = base^{-2k/d_half}  for k = 0..n_freq-1
    k = torch.arange(0, n_freq, dtype=dtype)
    freqs_y = 1.0 / (theta_y ** (2.0 * k / d_half))                 # (n_freq,)
    freqs_x = 1.0 / (theta_x ** (2.0 * k / d_half))                 # (n_freq,)

    pos_y = torch.arange(H, dtype=dtype)                            # (H,)
    pos_x = torch.arange(W, dtype=dtype)                            # (W,)

    # Outer products: angle[i, k] = pos_i * freq_k
    angles_y = torch.outer(pos_y, freqs_y)                          # (H, n_freq)
    angles_x = torch.outer(pos_x, freqs_x)                          # (W, n_freq)

    angles_y = angles_y.unsqueeze(1).expand(H, W, n_freq)           # (H, W, n_freq)
    angles_x = angles_x.unsqueeze(0).expand(H, W, n_freq)           # (H, W, n_freq)

    # Each rotation acts on a pair (x_{2k}, x_{2k+1}); repeat the angle
    # twice so cos/sin can be broadcast element-wise over head_dim.
    cos_y = angles_y.cos().repeat_interleave(2, dim=-1)             # (H, W, d_half)
    sin_y = angles_y.sin().repeat_interleave(2, dim=-1)             # (H, W, d_half)
    cos_x = angles_x.cos().repeat_interleave(2, dim=-1)             # (H, W, d_half)
    sin_x = angles_x.sin().repeat_interleave(2, dim=-1)             # (H, W, d_half)

    cos = torch.cat([cos_y, cos_x], dim=-1)                         # (H, W, head_dim)
    sin = torch.cat([sin_y, sin_x], dim=-1)                         # (H, W, head_dim)
    return cos.reshape(H * W, head_dim), sin.reshape(H * W, head_dim)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x0, x1, x2, x3, ...]`` to ``[-x1, x0, -x3, x2, ...]`` along the last dim."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)


def _apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a real tensor ``x`` of shape ``(..., N, D)``."""
    return x * cos + _rotate_pairs(x) * sin


class RoPE2D(nn.Module):
    """Per-stage 2-D RoPE module.

    Stores ``cos``/``sin`` as non-persistent buffers so they move with the
    module (``.cuda()`` etc.) but are not saved in the state dict — they are
    fully determined by ``grid_size`` and ``head_dim``.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        head_dim: int,
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.head_dim = head_dim
        cos, sin = _precompute_cos_sin(self.grid_size, head_dim, theta_y, theta_x)
        self.register_buffer("cos", cos, persistent=False)              # (N, D)
        self.register_buffer("sin", sin, persistent=False)              # (N, D)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q`` and ``k`` in place of position.

        Args:
            q, k: ``(B, num_heads, N, head_dim)`` real or complex.

        Returns:
            Rotated ``(q, k)`` of the same shape and dtype.
        """
        cos = self.cos.to(q.real.dtype if q.is_complex() else q.dtype)
        sin = self.sin.to(q.real.dtype if q.is_complex() else q.dtype)

        if q.is_complex():
            q_r = _apply_rope_real(q.real, cos, sin)
            q_i = _apply_rope_real(q.imag, cos, sin)
            k_r = _apply_rope_real(k.real, cos, sin)
            k_i = _apply_rope_real(k.imag, cos, sin)
            return torch.complex(q_r, q_i), torch.complex(k_r, k_i)

        return _apply_rope_real(q, cos, sin), _apply_rope_real(k, cos, sin)

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, head_dim={self.head_dim}"
