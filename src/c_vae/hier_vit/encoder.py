# ===============================================================
# HierViTEncoder: hierarchical ViT for STFT spectrograms.
#
# Like SwinEncoder, but blocks use GLOBAL attention with 2-D RoPE
# instead of W-MSA/SW-MSA with Log-CPB. Same hierarchical structure
# (PatchEmbed → N stages with PatchMerging between them).
#
# Designed for the spectrogram aspect ratio 1024×128 with a
# rectangular stem patch (default 16×4) that brings the first stage
# grid to a manageable (64, 32) = 2048 tokens — global attention is
# affordable at every stage.
# ===============================================================

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import to_2tuple

from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.models.implementations.abstract_ae import AbstractEncoder
from c_vae.swin.patches import PatchEmbed, PatchMerging
from c_vae.swin.utils import init_swin_weights, make_norm, random_patch_mask
from c_vae.hier_vit.stage import HierViTStage


class HierViTEncoder(AbstractEncoder):
    """Hierarchical ViT encoder with global attention + 2-D RoPE.

    Architecture (default: embed_dim=64, depths=[2,2,6,2], patch_size=(16,4)):

                                            C,    F,    T      grid
        Input                         (B,   2, 1024,  128)
        PatchEmbed Conv2d k=(16,4)    (B,  2048,   64)        ( 64, 32)
        Stage 1  2× HierViTBlock      (B,  2048,   64)        ( 64, 32)
            └── PatchMerging          (B,   512,  128)        ( 32, 16)
        Stage 2  2× HierViTBlock      (B,   512,  128)        ( 32, 16)
            └── PatchMerging          (B,   128,  256)        ( 16,  8)
        Stage 3  6× HierViTBlock      (B,   128,  256)        ( 16,  8)
            └── PatchMerging          (B,    32,  512)        (  8,  4)
        Stage 4  2× HierViTBlock      (B,    32,  512)        (  8,  4)
        LayerNorm + Linear(512→dim)   (B,    32,  out_chan)
        Transpose                     (B,  out_chan,   32)

    Args:
        in_channels: Spectrogram channels (2 = stereo complex STFT,
            4 = stereo CAC float). Aliased to ``input_size``.
        embed_dim: Stage-0 channel width. Doubles at each PatchMerging.
        depths: Number of HierViTBlocks per stage.
        num_heads: Attention heads per stage.
        patch_size: Kernel/stride of the PatchEmbed Conv2d. Tuple
            ``(ph, pw)`` for rectangular patches (spectrogram-friendly).
        dimension: Encoder output channels =
            ``parameters_to_predict × latent_channels``.
        mlp_ratio: FFN hidden expansion.
        drop_rate / attn_drop_rate / drop_path_rate: regularisation rates.
        is_complex: Switch the whole encoder to complex64 dtype.
        complex_activation: Phase-equivariant activation name for the Mlp.
        abs_pos_embed: Add a learnable absolute pos embedding at PatchEmbed
            (RoPE already provides position info; off by default).
        mim_mask_ratio: SimMIM-style patch dropout at training only.
        theta_y / theta_x: RoPE bases for freq / time axes. Default 1000 /
            10000 — freq has finer local periodicity than time.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        embed_dim: int = 64,
        depths: List[int] = (2, 2, 6, 2),
        num_heads: List[int] = (4, 8, 16, 32),
        patch_size: Union[int, Tuple[int, int]] = (16, 4),
        dimension: int = 24,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
        abs_pos_embed: bool = False,
        mim_mask_ratio: float = 0.0,
        theta_y: float = 1000.0,
        theta_x: float = 10000.0,
    ) -> None:
        super().__init__(input_size=in_channels, is_complex=is_complex)

        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.dimension = dimension
        self.num_stages = len(depths)
        self.mim_mask_ratio = float(mim_mask_ratio)

        ps_h, ps_w = to_2tuple(patch_size)
        self.patch_size: Tuple[int, int] = (ps_h, ps_w)

        self.downsampling_ratio: Tuple[int, int] = (
            ps_h * (2 ** (self.num_stages - 1)),
            ps_w * (2 ** (self.num_stages - 1)),
        )

        # STFT freq is cropped 1025 → 1024 in forward (Nyquist bin always 0).
        self._freq_size: int = 1024
        self._time_size: int = 128

        self.patch_embed = PatchEmbed(
            img_size=(self._freq_size, self._time_size),
            patch_size=(ps_h, ps_w),
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            is_complex=is_complex,
            abs_pos_embed=abs_pos_embed,
        )

        freq_g = self._freq_size // ps_h
        time_g = self._time_size // ps_w
        stage_resolutions: List[Tuple[int, int]] = []
        for _ in range(self.num_stages):
            stage_resolutions.append((freq_g, time_g))
            freq_g //= 2
            time_g //= 2
        stage_dims: List[int] = [embed_dim * (2 ** i) for i in range(self.num_stages)]

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
                downsample=PatchMerging if i < self.num_stages - 1 else None,
                is_complex=is_complex,
                complex_activation=complex_activation,
                theta_y=theta_y,
                theta_x=theta_x,
            )
            self.stages.append(stage)
            block_idx += depths[i]

        self._final_dim: int = stage_dims[-1]
        self._final_resolution: Tuple[int, int] = stage_resolutions[-1]

        self.norm = make_norm(self._final_dim, is_complex)
        self.head = NormLinear(self._final_dim, dimension, is_complex=is_complex)

        # init_swin_weights initialises both real and complex Linear layers and
        # standard LayerNorms — works as-is for our blocks.
        self.apply(init_swin_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encode a stereo STFT spectrogram to a 1-D latent sequence.

        Args:
            x: ``(B, in_channels, F, T)`` — F may be 1025 (Nyquist bin
                gets cropped here).

        Returns:
            ``(latents, {"feature_shape": (H_lat, W_lat)})`` with
            ``latents.shape == (B, dimension, H_lat * W_lat)``.
        """
        x = x[..., :self._freq_size, :]                            # (B, C, 1024, T)
        if self.training and self.mim_mask_ratio > 0.0:
            x = random_patch_mask(x, self.patch_size, self.mim_mask_ratio)
        x = self.patch_embed(x)                                    # (B, N0, embed_dim)
        for stage in self.stages:
            x = stage(x)                                           # progressively halves N, doubles C
        x = self.norm(x)                                           # (B, N_final, final_dim)
        x = self.head(x)                                           # (B, N_final, dimension)
        x = x.transpose(1, 2)                                      # (B, dimension, N_final)

        H_lat, W_lat = self._final_resolution
        return x, {"feature_shape": (H_lat, W_lat)}
