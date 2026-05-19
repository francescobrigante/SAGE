# ===============================================================
# Pre-norm ViT block: GlobalAttention + Mlp with residuals.
#
#   y = x + DropPath(Attn(LN(x)))
#   z = y + DropPath(Mlp(LN(y)))
#
# Differs from SwinTransformerBlock (which uses Swin V2 post-norm
# cosine attention) — this is canonical pre-norm ViT.
#
# Mlp is reused from c_vae.swin.swin_block (already is_complex-aware
# with ComplexGELU1d phase-equivariant activation).
# ===============================================================

from typing import Tuple

import torch
import torch.nn as nn

from c_vae.swin.swin_block import Mlp
from c_vae.swin.utils import make_norm, make_drop_path
from c_vae.hier_vit.attention import GlobalAttention


class HierViTBlock(nn.Module):
    """Single pre-norm ViT block with global attention + RoPE 2-D.

    Args:
        dim: Token channel dimension.
        grid_size: ``(H, W)`` for this stage's RoPE tables.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden-dim expansion ratio.
        qkv_bias: Add learnable Q and V biases (K = 0).
        drop: Dropout rate on FFN/projection.
        attn_drop: Dropout rate on attention weights.
        drop_path: Stochastic-depth rate.
        is_complex: Switch the whole block to complex64 dtype.
        complex_activation: Phase-equivariant activation name for the
            Mlp when ``is_complex=True``.
        theta_y / theta_x: RoPE bases for freq / time axes.
    """

    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.is_complex = is_complex

        self.norm1 = make_norm(dim, is_complex)
        self.attn = GlobalAttention(
            dim=dim,
            grid_size=self.grid_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_complex=is_complex,
            theta_y=theta_y,
            theta_x=theta_x,
        )
        self.drop_path = make_drop_path(drop_path, is_complex)
        self.norm2 = make_norm(dim, is_complex)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            is_complex=is_complex,
            complex_activation=complex_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, N=H*W, C)`` → same shape."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, grid_size={self.grid_size}, is_complex={self.is_complex}"
