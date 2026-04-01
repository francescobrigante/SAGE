# ===============================================================
# Swin Transformer V2: stage constructor.
#
# SwinStage stacks N SwinTransformerBlocks (alternating W-MSA /
# SW-MSA) and applies an optional spatial downsampler at the end.
# ===============================================================

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .swin_block import SwinTransformerBlock


class SwinStage(nn.Module):
    """One hierarchical stage of Swin Transformer V2.

    Stacks depth SwinTransformerBlock modules with alternating W-MSA / SW-MSA,
    optionally followed by a spatial downsampler (e.g. PatchMerging in the encoder).

    Typical encoder layout with depths=[2, 2, 4, 2]:

        SwinStage 0 →  2 blocks  grid (256, 32)  dim= 48  → PatchMerging ↓
        SwinStage 1 →  2 blocks  grid (128, 16)  dim= 96  → PatchMerging ↓
        SwinStage 2 →  4 blocks  grid  (64,  8)  dim=192  → PatchMerging ↓
        SwinStage 3 →  2 blocks  grid  (32,  4)  dim=384  (no downsample)

    Args:
        dim: Token channel dimension for this stage.
        input_resolution: (H, W) spatial grid at stage input.
        depth: Number of SwinTransformerBlock modules to stack.
        num_heads: Number of attention heads.
        window_size: Local attention window size.
        mlp_ratio: FFN hidden-dim expansion ratio.
        qkv_bias: Learnable bias on Q and V projections.
        drop: FFN and projection dropout rate.
        attn_drop: Attention weight dropout rate.
        drop_path: Stochastic depth rate(s); scalar or list of length depth.
        norm_layer: Normalisation class. Default: nn.LayerNorm.
        downsample: Downsampler class instantiated as
            downsample(input_resolution, dim=dim, norm_layer=norm_layer),
            or None for no spatial change (decoder stages, last encoder stage).
        use_checkpoint: Gradient checkpointing to trade compute for VRAM.
        pretrained_window_size: Pre-training window size for Log-CPB normalisation.
        fused_window_process: Use fused CUDA window kernel (CUDA only).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
        pretrained_window_size: int = 0,
        fused_window_process: bool = False,
    ) -> None:
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint   # gradient checkpointing flag

        # Even index = W-MSA (no shift), odd index = SW-MSA (cyclic shift window_size//2)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                fused_window_process=fused_window_process,
            )
            for i in range(depth)
        ])

        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all blocks then optionally downsample.

        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C) if no downsample, else (B, H/2 * W/2, 2C).
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # (B, H*W, C)
            else:
                x = blk(x)                          # (B, H*W, C)
                
        if self.downsample is not None:
            x = self.downsample(x)                  # (B, H/2*W/2, 2C)
            
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def _init_respostnorm(self) -> None:
        """Zero-init post-norm residual scaling (Swin V2 training stabilisation).

        Zeroing norm weights at init means residuals start as identity mappings,
        stabilising deep-network training. Called once after init_swin_weights.
        """
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


# Backwards-compatible alias — existing code that imports BasicLayer still works
BasicLayer = SwinStage
