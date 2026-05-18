# ===============================================================
# HierViTStage: a stack of HierViTBlocks followed by an optional
# spatial resampler (PatchMerging for encoder, PatchExpand for
# decoder, or None at the deepest stage).
# ===============================================================

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from c_vae.hier_vit.block import HierViTBlock


class HierViTStage(nn.Module):
    """One stage = ``depth`` HierViTBlocks at fixed grid + optional resampler.

    Args:
        dim: Channel dimension at this stage (the resampler may change it for
            the next stage).
        input_resolution: ``(H, W)`` token grid for the blocks.
        depth: Number of blocks.
        num_heads: Attention heads per block.
        mlp_ratio: FFN hidden expansion.
        qkv_bias: Add Q/V biases (K = 0).
        drop: FFN/projection dropout.
        attn_drop: Attention-weight dropout.
        drop_path: Per-block stochastic-depth rates (list of length ``depth``).
        downsample: Class (e.g. ``PatchMerging``) instantiated with
            ``(input_resolution=input_resolution, dim=dim, is_complex=is_complex)``
            applied AFTER the blocks, or ``None`` for no resampling.
        upsample: Mirror of ``downsample`` — class instantiated with the
            same signature applied AFTER the blocks. Use exactly one of
            ``downsample`` / ``upsample`` or neither.
        is_complex: Switch the whole stage to complex dtype.
        complex_activation: Phase-equivariant activation name for the Mlp.
        theta_y / theta_x: RoPE bases.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        downsample: Optional[Type[nn.Module]] = None,
        upsample: Optional[Type[nn.Module]] = None,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__()
        if downsample is not None and upsample is not None:
            raise ValueError("downsample and upsample are mutually exclusive")

        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.is_complex = is_complex

        if isinstance(drop_path, (int, float)):
            drop_path_list = [float(drop_path)] * depth
        else:
            drop_path_list = list(drop_path)
            if len(drop_path_list) != depth:
                raise ValueError(
                    f"drop_path length {len(drop_path_list)} != depth {depth}"
                )

        self.blocks = nn.ModuleList([
            HierViTBlock(
                dim=dim,
                grid_size=self.input_resolution,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_list[i],
                is_complex=is_complex,
                complex_activation=complex_activation,
                theta_y=theta_y,
                theta_x=theta_x,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.resampler = downsample(
                input_resolution=self.input_resolution,
                dim=dim,
                is_complex=is_complex,
            )
        elif upsample is not None:
            self.resampler = upsample(
                input_resolution=self.input_resolution,
                dim=dim,
                is_complex=is_complex,
            )
        else:
            self.resampler = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.resampler is not None:
            x = self.resampler(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"depth={self.depth}, is_complex={self.is_complex}"
        )
