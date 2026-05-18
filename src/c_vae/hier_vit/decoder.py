# ===============================================================
# HierViTDecoder: symmetric inverse of HierViTEncoder.
# Global-attention ViT stages with PatchExpand upsampling and
# 2-D RoPE at every stage.
# ===============================================================

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import to_2tuple

from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.models.implementations.abstract_ae import AbstractDecoder
from c_vae.swin.patches import PatchExpand, PatchUnembed
from c_vae.swin.utils import init_swin_weights, make_norm
from c_vae.hier_vit.stage import HierViTStage


class HierViTDecoder(AbstractDecoder):
    """Hierarchical ViT decoder mirroring :class:`HierViTEncoder`.

    Architecture (default mirroring depths=[2,6,2,2], patch_size=(16,4)):

        Input (latent z)              (B, latent_channels,  N_lat = 32)
        Transpose + Linear(C → 512)   (B,    32,  512)       (  8,  4)
        Stage 4  2× HierViTBlock      (B,    32,  512)       (  8,  4)
            └── PatchExpand           (B,   128,  256)       ( 16,  8)
        Stage 3  6× HierViTBlock      (B,   128,  256)       ( 16,  8)
            └── PatchExpand           (B,   512,  128)       ( 32, 16)
        Stage 2  2× HierViTBlock      (B,   512,  128)       ( 32, 16)
            └── PatchExpand           (B,  2048,   64)       ( 64, 32)
        Stage 1  2× HierViTBlock      (B,  2048,   64)       ( 64, 32)
        LayerNorm                     (B,  2048,   64)
        PatchUnembed ConvT k=(16,4)   (B, in_channels, 1024, 128)

    Args:
        channels: Latent channel count (the bottleneck's ``latent_channels``).
        in_channels: Output spectrogram channels.
        embed_dim: Smallest channel width (last decoder stage = first encoder).
        depths: Block counts per stage — should be the encoder's depths reversed.
        num_heads: Heads per stage — encoder heads reversed.
        patch_size: ConvTranspose2d kernel/stride for PatchUnembed; must
            match the encoder's PatchEmbed for shape inversion.
        mlp_ratio / drop_rate / attn_drop_rate / drop_path_rate: as encoder.
        is_complex: Switch to complex64 dtype.
        complex_activation: Phase-equivariant Mlp activation.
        theta_y / theta_x: RoPE bases.
    """

    def __init__(
        self,
        *,
        channels: int,
        in_channels: int = 2,
        embed_dim: int = 64,
        depths: List[int] = (2, 6, 2, 2),
        num_heads: List[int] = (32, 16, 8, 4),
        patch_size: Union[int, Tuple[int, int]] = (16, 4),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)

        self.input_size = channels                                 # for AutoEncoder dim-check
        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        ps_h, ps_w = to_2tuple(patch_size)
        self.patch_size: Tuple[int, int] = (ps_h, ps_w)
        self.num_stages = len(depths)

        stage_dims: List[int] = [
            embed_dim * (2 ** (self.num_stages - 1 - i))
            for i in range(self.num_stages)
        ]

        _freq_size = 1024
        _time_size = 128
        base_h = _freq_size // ps_h // (2 ** (self.num_stages - 1))
        base_w = _time_size // ps_w // (2 ** (self.num_stages - 1))
        stage_resolutions: List[Tuple[int, int]] = [
            (base_h * (2 ** i), base_w * (2 ** i))
            for i in range(self.num_stages)
        ]

        self._stage_dims = stage_dims
        self._stage_resolutions = stage_resolutions

        self.input_proj = NormLinear(channels, stage_dims[0], is_complex=is_complex)

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        block_idx = 0
        for i in range(self.num_stages):
            stage = HierViTStage(
                dim=stage_dims[i],
                input_resolution=stage_resolutions[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx: block_idx + depths[i]],
                upsample=PatchExpand if i < self.num_stages - 1 else None,
                is_complex=is_complex,
                complex_activation=complex_activation,
                theta_y=theta_y,
                theta_x=theta_x,
            )
            self.stages.append(stage)
            block_idx += depths[i]

        self.norm = make_norm(stage_dims[-1], is_complex)
        self.patch_unembed = PatchUnembed(
            input_resolution=stage_resolutions[-1],
            embed_dim=stage_dims[-1],
            out_channels=in_channels,
            patch_size=(ps_h, ps_w),
            is_complex=is_complex,
        )

        self.apply(init_swin_weights)

    def forward(
        self, x: torch.Tensor, encoder_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """Decode a latent sequence back to a spectrogram.

        Args:
            x: ``(B, latent_channels, N_lat)``.
            encoder_info: unused — kept for interface compatibility.
        """
        x = x.transpose(1, 2)                                      # (B, N_lat, latent_channels)
        x = self.input_proj(x)                                     # (B, N_lat, stage_dims[0])
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.patch_unembed(x)                                  # (B, in_channels, 1024, 128)
        return x
