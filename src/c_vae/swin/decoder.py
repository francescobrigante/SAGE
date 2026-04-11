# ====================================================================
# SwinDecoder: Swin Transformer V2 decoder, symmetric inverse 
# of SwinEncoder.
# Accepts (B, latent_channels, H_lat*W_lat) float32 latent 
# sequence and reconstructs (B, in_channels, 1024, 128)
# ====================================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.models.implementations.abstract_ae import AbstractDecoder
from .swin_stage import SwinStage
from .patches import PatchExpand, PatchUnembed
from .utils import init_swin_weights, make_norm


class SwinDecoder(AbstractDecoder):
    """
    Architecture (embed_dim=48, depths=[2,4,2,2], patch_size=4, latent_channels=64):

        Input (latent z)              (B, latent_channels=64, H*W_lat=128)
        
                                                H*W,    C       grid
        Transpose + Linear(64->384)         (B, 128,  384)    ( 32,  4)
        Stage 4:  2x SwinBlock              (B, 128,  384)    ( 32,  4)
            └── PatchExpand                 (B, 512,  192)    ( 64,  8)
        Stage 3:  4x SwinBlock              (B, 512,  192)    ( 64,  8)
            └── PatchExpand                 (B,2048,   96)    (128, 16)
        Stage 2:  2x SwinBlock              (B,2048,   96)    (128, 16)
            └── PatchExpand                 (B,8192,   48)    (256, 32)
        Stage 1:  2x SwinBlock              (B,8192,   48)    (256, 32)  (no expand)
        LayerNorm                           (B,8192,   48)    (256, 32)
        PatchUnembed  ConvTranspose2d       (B,   4, 1024, 128)

    Args:
        channels: Latent channel count (= latent_channels). Stored as self.channels
            via AbstractDecoder.
        in_channels: Output channel count of the reconstruction (4 = stereo CAC STFT).
        embed_dim: Smallest token dimension, used by the last stage and PatchUnembed.
        depths: SwinTransformerBlock count per stage (reversed encoder depths).
        num_heads: Attention heads per stage (reversed encoder heads).
        window_size: Local attention window size; must divide all stage grids.
        patch_size: ConvTranspose2d kernel/stride for PatchUnembed.
        mlp_ratio: FFN hidden-dim expansion ratio.
        drop_rate: Dropout rate for FFN and projections.
        attn_drop_rate: Dropout rate applied to attention weights.
        drop_path_rate: Peak stochastic-depth rate, linearly distributed across blocks.
        use_checkpoint: Enable gradient checkpointing to save VRAM.
    """

    def __init__(
        self,
        *,
        channels: int,               # latent_channels (e.g. 64)
        in_channels: int = 2,        # reconstruction output channels (stereo STFT)
        embed_dim: int = 48,         # smallest channel width (last stage + PatchUnembed)
        depths: List[int] = (2, 4, 2, 2),
        num_heads: List[int] = (24, 12, 6, 3),
        window_size: int = 8,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        is_complex: bool = False,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)

        self.input_size = channels          # alias for AutoEncoder dimension-check compatibility
        self.embed_dim = embed_dim          # smallest channel width
        self.depths = list(depths)          # blocks per stage, e.g. [2,4,2,2]
        self.num_heads = list(num_heads)    # attention heads per stage
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_stages = len(depths)       # 4

        # Stage channel dims
        # e.g. [384, 192, 96, 48] for embed_dim=48, num_stages=4
        stage_dims: List[int] = [
            embed_dim * (2 ** (self.num_stages - 1 - i))
            for i in range(self.num_stages)
        ]

        # Stage grid resolutions
        _freq_size = 1024
        _time_size = 128
        base_h = _freq_size // patch_size // (2 ** (self.num_stages - 1))   # 32
        base_w = _time_size // patch_size // (2 ** (self.num_stages - 1))   # 4
        # e.g. stage_resolutions = [(32,4), (64,8), (128,16), (256,32)]
        stage_resolutions: List[Tuple[int, int]] = [
            (base_h * (2 ** i), base_w * (2 ** i))
            for i in range(self.num_stages)
        ]

        self._stage_dims = stage_dims
        self._stage_resolutions = stage_resolutions

        # Input projection: latent_channels → largest stage dim
        # (B, T, latent_channels=64) → (B, T, stage_dims[0]=384)
        self.input_proj = NormLinear(channels, stage_dims[0], is_complex=is_complex)

        # Stochastic depth
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Swin stages + PatchExpand upsamplers
        self.stages = nn.ModuleList()
        self.patch_expands = nn.ModuleList()
        block_idx = 0
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=stage_dims[i],
                input_resolution=stage_resolutions[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx: block_idx + depths[i]],
                norm_layer=nn.LayerNorm,
                downsample=None,     # no spatial downsampling in the decoder
                use_checkpoint=use_checkpoint,
                is_complex=is_complex,
            )
            self.stages.append(stage)
            block_idx += depths[i]

            if i < self.num_stages - 1:
                self.patch_expands.append(
                    PatchExpand(
                        input_resolution=stage_resolutions[i],
                        dim=stage_dims[i],
                        norm_layer=nn.LayerNorm,
                        is_complex=is_complex,
                    )
                )

        # Final norm + PatchUnembed
        self.norm = make_norm(stage_dims[-1], is_complex)
        self.patch_unembed = PatchUnembed(
            input_resolution=stage_resolutions[-1],   # (256, 32) final token grid
            embed_dim=stage_dims[-1],                 # 48
            out_channels=in_channels,                 # 4 (stereo CAC STFT)
            patch_size=patch_size,                    # 4
            is_complex=is_complex,
        )

        # Weight init
        self.apply(init_swin_weights)
        for stage in self.stages:
            stage._init_respostnorm()

    def forward(self, x: torch.Tensor,encoder_info: Optional[Dict] = None) -> torch.Tensor:
        """Decode a latent sequence back to a stereo STFT spectrogram.

        Args:
            x: (B, latent_channels, H_lat * W_lat) = (B, 64, 128)
            encoder_info: unused; accepted for interface compatibility.

        Returns:
            STFT (B, in_channels, F=1024, T=128) float32 reconstruction.
        """
        # (B, latent_chan, H * W) -> (B, H * W, latent_chan)
        x = x.transpose(1, 2)                           # (B, H*W=128,  C=64)

        # [ALTERNATIVE] Input from 2D spatial map (B, C, H=32, W=4):
        # x = x.flatten(2).transpose(1, 2)              # (B, 128,  64)

        x = self.input_proj(x)                          # (B, 128, 384)

        # 4 Swin stages with PatchExpand between them
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < self.num_stages - 1:
                x = self.patch_expands[i](x)
        # Stage 4  (384, grid 32*4):               (B,  128, 384)
        # └── PatchExpand → (192, grid 64*8):      (B,  512, 192)
        # Stage 3  (192, grid 64*8):               (B,  512, 192)
        # └── PatchExpand → ( 96, grid 128*16):    (B, 2048,  96)
        # Stage 2  ( 96, grid 128*16):             (B, 2048,  96)
        # └── PatchExpand → ( 48, grid 256*32):    (B, 8192,  48)
        # Stage 1  ( 48, grid 256*32):             (B, 8192,  48)  (no expand)

        x = self.norm(x)                                # (B, 8192,  C=48)
        x = self.patch_unembed(x)                       # (B, in_channels, F=1024, T=128)

        return x