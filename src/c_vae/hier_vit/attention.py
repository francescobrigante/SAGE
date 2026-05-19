# ===============================================================
# Global multi-head self-attention with 2-D RoPE.
#
# Real path  (is_complex=False): standard scaled dot-product attention.
# Complex path (is_complex=True): Hermitian inner-product score
#     Re(Q K^H) = Q.real @ K.real^T + Q.imag @ K.imag^T
#     followed by softmax; attn @ V is split re/im (same pattern as
#     Swin V2 WindowAttention).
#
# No windowing, no relative position bias MLP — position information
# is delivered exclusively through RoPE applied to Q and K.
# ===============================================================

import math
from typing import Tuple

import torch
import torch.nn as nn

from ar_spectra.blocks.conv.normed import NormLinear
from c_vae.hier_vit.rope_2d import RoPE2D


class GlobalAttention(nn.Module):
    """Global self-attention over the full token grid with 2-D RoPE.

    Args:
        dim: Token channel dimension. Must be divisible by ``num_heads``.
        grid_size: ``(H, W)`` token grid this stage operates on. Used to
            precompute RoPE tables; ``H*W`` must equal the sequence length.
        num_heads: Number of attention heads. ``dim/num_heads`` must be
            divisible by 4 (RoPE 2-D pair split).
        qkv_bias: Add learnable Q and V biases (K bias = 0, Swin V2 style).
        attn_drop: Dropout on attention weights.
        proj_drop: Dropout on output projection.
        is_complex: If True, run the complex path with Hermitian scoring.
        theta_y: RoPE base for the y (freq) axis.
        theta_x: RoPE base for the x (time) axis.
    """

    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_complex: bool = False,
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by 4 for 2-D RoPE")

        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_complex = is_complex
        self.scale = 1.0 / math.sqrt(head_dim)

        # QKV projection: NormLinear is mathematically nn.Linear when norm='none' (default);
        # is_complex=True sets dtype=complex64 on the weight, producing complex output.
        self.qkv = NormLinear(dim, dim * 3, bias=False, is_complex=is_complex)

        _bias_dtype = torch.complex64 if is_complex else torch.float32
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
            self.v_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = RoPE2D(self.grid_size, head_dim, theta_y=theta_y, theta_x=theta_x)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
        self.proj = NormLinear(dim, dim, bias=True, is_complex=is_complex)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, C)`` with ``N = H*W``.

        Returns:
            ``(B, N, C)``.
        """
        B, N, C = x.shape

        qkv = self.qkv(x)                                                   # (B, N, 3C)
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
            qkv = qkv + qkv_bias
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                     # each (B, H, N, D)

        q, k = self.rope(q, k)

        if self.is_complex:
            # Hermitian inner product Re(Q K^H); 2 real SGEMM instead of 1 ZGEMM.
            attn = (q.real @ k.real.mT + q.imag @ k.imag.mT) * self.scale    # (B, H, N, N) real
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale                    # (B, H, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if self.is_complex:
            # attn is real, V is complex → split re/im to dodge mixed-dtype matmul.
            # bf16-mixed autocast leaves matmul output in bf16; torch.complex needs fp32.
            out_re = (attn @ v.real).to(torch.float32)
            out_im = (attn @ v.imag).to(torch.float32)
            x = torch.complex(out_re, out_im)                                # (B, H, N, D)
        else:
            x = attn @ v                                                     # (B, H, N, D)

        x = x.transpose(1, 2).reshape(B, N, C)                               # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, grid_size={self.grid_size}, "
            f"num_heads={self.num_heads}, is_complex={self.is_complex}"
        )
