# ===============================================================
# SwinDecoder: exact symmetric inverse of SwinEncoder.
# Accepts (B, latent_channels, 32, 4) float32 and reconstructs
# (B, 2, 1024, 128) via PatchExpand upsampling + SwinTransformerBlocks.
# PatchExpand is Phase-2-local; replaced by PatchUnmergingLinearComplex in Phase 4.
# ===============================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ar_spectra.models.implementations.abstract_ae import AbstractDecoder
from c_vae.swin.swin_transformer_v2 import (
    BasicLayer,
    init_swin_weights,
)


# --------------------------------------------------------------------------- #
# PatchExpand — exact inverse of PatchMerging                                 #
# --------------------------------------------------------------------------- #

class PatchExpand(nn.Module):
    """Pixel-shuffle upsample: the spatial inverse of ``PatchMerging``.

    ``PatchMerging``  : ``(B, H*W, C)``   → concat 2×2 → Linear(4C→2C) → ``(B, H/2*W/2, 2C)``
    ``PatchExpand``   : ``(B, H*W, 2C)``  → Linear(2C→4C) → 2×2 scatter  → ``(B, 4*H*W, C)``

    The 2×2 scatter is implemented as a view + permute that places the 4
    sub-channels at their correct spatial positions (pixel-shuffle in 2-D).

    Args:
        input_resolution: ``(H, W)`` of the input token grid (before expansion).
        dim: Input channel dimension (``= 2C``); output dim will be ``C = dim // 2``.
        norm_layer: Normalization applied to the output tokens.

    .. note::
        Phase 2 local implementation. In Phase 4 this will be replaced by
        ``PatchUnmergingLinearComplex`` from ``ar_spectra.blocks.complex_patch_merging``
        (real variant thereof).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution  # (H, W) before expansion
        self.dim = dim                             # input channels (2C)
        self.out_dim = dim // 2                   # output channels (C)

        # Maps 2C → 4C; after 2×2 scatter each position carries C channels
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, H*W, 2C)``
        Returns:
            x: ``(B, 4*H*W, C)`` = ``(B, 2H*2W, C)``
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W,   f"token count {L} != H*W = {H*W}"
        assert C == self.dim, f"channel dim {C} != expected {self.dim}"

        x = self.expand(x)                               # (B, H*W, 4C)  — Linear(2C→4C)
        x = x.view(B, H, W, 2 * C)                      # (B, H, W,  4C)
        x = x.view(B, H, W, 2, 2, self.out_dim)         # (B, H, W,  2, 2, C)  — split into 2×2 sub-pixels
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, H, 2,  W, 2, C)  — interleave rows and cols
        x = x.view(B, 2 * H, 2 * W, self.out_dim)       # (B, 2H, 2W, C)
        x = x.reshape(B, 4 * H * W, self.out_dim)        # (B, 4H*W,   C)
        x = self.norm(x)                                 # (B, 4H*W,   C)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}->{self.out_dim}"


# --------------------------------------------------------------------------- #
# SwinDecoder                                                                  #
# --------------------------------------------------------------------------- #

class SwinDecoder(AbstractDecoder):
    """Swin Transformer V2 decoder — exact spatial and channel-wise inverse of ``SwinEncoder``.

    Symmetry guarantees:
      - ``PatchExpand`` is the arithmetic inverse of ``PatchMerging``: each expands
        spatial resolution by 2× and halves channel count.
      - ``PatchUnembed`` (``ConvTranspose2d``) is the inverse of ``PatchEmbed`` (``Conv2d``),
        identical kernel and stride.
      - Depth sequence ``[2, 4, 2, 2]`` is the reversed encoder ``[2, 2, 4, 2]``.
      - Channel dims ``[384, 192, 96, 48]`` are the reversed encoder ``[48, 96, 192, 384]``.
      - No skip connections — intentional: the bottleneck must encode all information
        for the downstream diffusion model.

    Architecture (embed_dim=48, depths=[2,4,2,2], patch_size=4, latent_channels=64):

        Input          (B,  64,  32,   4)  sampled latent z
        Flatten+proj   Linear(64→384)      (B, 128, 384)   grid ( 32,  4)
        Stage 4'       2× SwinBlock        (B, 128, 384)   grid ( 32,  4)
        PatchExpand 3  Linear(384→4×192)   (B, 512, 192)   grid ( 64,  8)
        Stage 3'       4× SwinBlock        (B, 512, 192)   grid ( 64,  8)
        PatchExpand 2  Linear(192→4×96)    (B,2048,  96)   grid (128, 16)
        Stage 2'       2× SwinBlock        (B,2048,  96)   grid (128, 16)
        PatchExpand 1  Linear( 96→4×48)    (B,8192,  48)   grid (256, 32)
        Stage 1'       2× SwinBlock        (B,8192,  48)   grid (256, 32)
        LayerNorm                          (B,8192,  48)
        Reshape                            (B,  48, 256, 32)
        PatchUnembed   ConvTranspose2d     (B,   2,1024,128)  ← decoder output

    Args:
        channels: Latent channel count (= ``latent_channels``); the decoder's
            input will have this many channels. Stored as ``self.channels``
            via ``AbstractDecoder``.
        in_channels: Output channel count of the reconstruction (2 = stereo STFT).
        embed_dim: Smallest token dimension, used by the last stage and ``PatchUnembed``.
        depths: ``SwinTransformerBlock`` count per stage (reversed encoder depths).
        num_heads: Attention heads per stage (reversed encoder heads).
        window_size: Local attention window size; must divide all stage grids.
        patch_size: ``ConvTranspose2d`` kernel/stride for ``PatchUnembed``.
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
    ) -> None:
        super().__init__(channels=channels, is_complex=False)

        self.input_size = channels          # alias for AutoEncoder dimension-check compatibility
        self.embed_dim = embed_dim          # smallest channel width
        self.depths = list(depths)          # blocks per stage, e.g. [2,4,2,2]
        self.num_heads = list(num_heads)    # attention heads per stage
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_stages = len(depths)       # 4

        # ------------------------------------------------------------------ #
        # Stage channel dims (large → small, mirroring encoder in reverse)   #
        # stage_dims[i] = embed_dim * 2^(num_stages - 1 - i)                 #
        # = [384, 192, 96, 48] for embed_dim=48, num_stages=4                #
        # ------------------------------------------------------------------ #
        stage_dims: List[int] = [
            embed_dim * (2 ** (self.num_stages - 1 - i))
            for i in range(self.num_stages)
        ]

        # ------------------------------------------------------------------ #
        # Stage grid resolutions (small → large, inverse of encoder)         #
        # base = (freq_size // patch_size // 2^(num_stages-1), ...)           #
        # = (1024 // 4 // 8, 128 // 4 // 8) = (32, 4)                        #
        # ------------------------------------------------------------------ #
        _freq_size = 1024
        _time_size = 128
        base_h = _freq_size // patch_size // (2 ** (self.num_stages - 1))   # 32
        base_w = _time_size // patch_size // (2 ** (self.num_stages - 1))   # 4
        stage_resolutions: List[Tuple[int, int]] = [
            (base_h * (2 ** i), base_w * (2 ** i))
            for i in range(self.num_stages)
        ]
        # stage_resolutions = [(32,4), (64,8), (128,16), (256,32)]

        # Store for use in forward
        self._stage_dims = stage_dims
        self._stage_resolutions = stage_resolutions

        # ------------------------------------------------------------------ #
        # Input projection: latent_channels → largest stage dim              #
        # (B, H_lat*W_lat, latent_channels) → (B, H_lat*W_lat, stage_dims[0])
        # ------------------------------------------------------------------ #
        self.input_proj = nn.Linear(channels, stage_dims[0])  # 64 → 384

        # ------------------------------------------------------------------ #
        # Stochastic depth — linear schedule 0 → drop_path_rate              #
        # ------------------------------------------------------------------ #
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ------------------------------------------------------------------ #
        # Stages (BasicLayer, no downsample) + PatchExpand between them      #
        # Pattern: Stage_i → PatchExpand_i → Stage_{i+1} → ... → Stage_{n-1}#
        # ------------------------------------------------------------------ #
        self.stages = nn.ModuleList()
        self.patch_expands = nn.ModuleList()
        block_idx = 0
        for i in range(self.num_stages):
            stage = BasicLayer(
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
            )
            self.stages.append(stage)
            block_idx += depths[i]

            # PatchExpand after every stage except the last
            if i < self.num_stages - 1:
                self.patch_expands.append(
                    PatchExpand(
                        input_resolution=stage_resolutions[i],
                        dim=stage_dims[i],
                        norm_layer=nn.LayerNorm,
                    )
                )
        # patch_expands = [Expand(32,4  384→192),
        #                  Expand(64,8  192→96),
        #                  Expand(128,16  96→48)]

        # ------------------------------------------------------------------ #
        # Output head                                                         #
        # ------------------------------------------------------------------ #
        # LayerNorm on the final token dim (embed_dim = 48)
        self.norm = nn.LayerNorm(stage_dims[-1])

        # ConvTranspose2d(embed_dim → in_channels, k=patch_size, s=patch_size)
        # (B, 48, 256, 32) → (B, 2, 1024, 128)
        self.patch_unembed = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=patch_size, stride=patch_size
        )

        # ------------------------------------------------------------------ #
        # Weight initialisation                                               #
        # ------------------------------------------------------------------ #
        self.apply(init_swin_weights)
        for stage in self.stages:
            stage._init_respostnorm()

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        encoder_info: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Decode a latent token map back to a stereo STFT spectrogram.

        Args:
            x: ``(B, latent_channels, H_lat, W_lat)`` = ``(B, 64, 32, 4)``
            encoder_info: unused; accepted for interface compatibility.

        Returns:
            ``(B, 2, 1024, 128)`` float32 reconstruction.
        """
        B = x.shape[0]

        # --- Flatten spatial dims + project to largest stage dim -----------
        x = x.flatten(2).transpose(1, 2)               # (B, 128, 64)   — H_lat*W_lat tokens
        x = self.input_proj(x)                          # (B, 128, 384)  — project latent_channels → 384

        # --- Hierarchical upsampling stages --------------------------------
        # Each stage runs SwinBlocks; all but the last are followed by
        # PatchExpand (doubles the spatial grid, halves channel dim).
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < self.num_stages - 1:
                x = self.patch_expands[i](x)
        # After stage 0 (dim=384, grid 32×4):    (B,  128, 384)  grid ( 32,  4)
        # After expand 0 (384→192, →64×8):       (B,  512, 192)  grid ( 64,  8)
        # After stage 1 (dim=192, grid 64×8):    (B,  512, 192)  grid ( 64,  8)
        # After expand 1 (192→96, →128×16):      (B, 2048,  96)  grid (128, 16)
        # After stage 2 (dim=96, grid 128×16):   (B, 2048,  96)  grid (128, 16)
        # After expand 2 (96→48, →256×32):       (B, 8192,  48)  grid (256, 32)
        # After stage 3 (dim=48, grid 256×32):   (B, 8192,  48)  grid (256, 32)

        # --- Normalise + reshape to 2-D -------------------------------------
        x = self.norm(x)                                # (B, 8192,  48)
        H_out, W_out = self._stage_resolutions[-1]      # 256, 32
        x = x.transpose(1, 2)                           # (B, 48, 8192)
        x = x.reshape(B, self._stage_dims[-1], H_out, W_out)  # (B, 48, 256, 32)

        # --- PatchUnembed: ConvTranspose2d(48→2, k=4, s=4) -----------------
        x = self.patch_unembed(x)                       # (B, 2, 1024, 128)

        return x
