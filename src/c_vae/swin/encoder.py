# ===============================================================
# SwinEncoder: Swin Transformer V2 encoder adapted for audio STFT.
# Accepts (B, 2, F, T) float32 stereo STFT (F may be 1025),
# crops to 1024 freq bins, and outputs (B, ptp×lc, 32, 4).
# ===============================================================

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ar_spectra.models.implementations.abstract_ae import AbstractEncoder
from c_vae.swin.swin_transformer_v2 import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
    init_swin_weights,
)


class SwinEncoder(AbstractEncoder):
    """Swin Transformer V2 encoder for audio STFT spectrograms (Exp 0 — real-valued).

    Differences from the original ImageNet SwinTransformerV2:
      - ``in_chans=2`` (stereo float STFT) instead of 3 (RGB)
      - ``window_size=8`` (divides all stage grids) instead of 7 (ImageNet 224px)
      - Freq crop 1025 → 1024 applied at the forward boundary before PatchEmbed
      - No classification head; instead a Linear projection to ``dimension`` channels
        followed by a reshape to a 2-D spatial latent map

    Architecture (embed_dim=48, depths=[2,2,4,2], patch_size=4):

        Input          (B,   2, 1024, 128)  after freq crop
        PatchEmbed     (B,   48, 256, 32)   → tokens (B, 8192, 48)
        Stage 1        2x SwinBlock          (B, 8192,  48)  grid (256, 32)
        PatchMerging   → 2x↓                 (B, 2048,  96)  grid (128, 16)
        Stage 2        2x SwinBlock          (B, 2048,  96)  grid (128, 16)
        PatchMerging   → 2x↓                 (B,  512, 192)  grid  (64,  8)
        Stage 3        4x SwinBlock          (B,  512, 192)  grid  (64,  8)
        PatchMerging   → 2x↓                 (B,  128, 384)  grid  (32,  4)
        Stage 4        2x SwinBlock          (B,  128, 384)  grid  (32,  4)
        LayerNorm + Linear(384→dimension)    (B,  128, 128)
        Reshape                              (B, 128, 32, 4)  ← encoder output

    Args:
        in_channels: Input spectrogram channels (2 = stereo STFT). Aliased to
            ``input_size`` for the ``AbstractEncoder`` contract.
        embed_dim: Base token dimension at stage 0. Doubles at each PatchMerging.
        depths: Number of ``SwinTransformerBlock``s per stage.
        num_heads: Number of self-attention heads per stage.
        window_size: Local attention window size. Must divide every stage grid
            dimension. Default 8 works for the (1024, 128) STFT grid.
        patch_size: Kernel/stride of the ``PatchEmbed`` Conv2d.
        dimension: Total encoder output channels = ``parameters_to_predict × latent_channels``.
        mlp_ratio: FFN hidden-dim expansion ratio.
        drop_rate: Dropout rate for FFN and output projections.
        attn_drop_rate: Dropout rate applied to attention weights.
        drop_path_rate: Peak stochastic-depth rate; linearly distributed across all blocks.
        use_checkpoint: Enable gradient checkpointing to trade speed for VRAM.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        embed_dim: int = 48,
        depths: List[int] = (2, 2, 4, 2),
        num_heads: List[int] = (3, 6, 12, 24),
        window_size: int = 8,
        patch_size: int = 4,
        dimension: int = 128,       # parameters_to_predict × latent_channels
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        # AbstractEncoder stores input_size (= in_channels) and is_complex
        super().__init__(input_size=in_channels, is_complex=False)

        self.embed_dim = embed_dim          # base channel width (doubles per merge)
        self.depths = list(depths)          # blocks per stage, e.g. [2, 2, 4, 2]
        self.num_heads = list(num_heads)    # attention heads per stage
        self.window_size = window_size      # local window size (must divide all grids)
        self.patch_size = patch_size        # PatchEmbed Conv2d kernel/stride
        self.dimension = dimension          # total output channels (ptp × lc)
        self.num_stages = len(depths)       # number of hierarchical stages (4)

        # Total spatial downsampling:
        #   patch_size (4) × 2^(num_stages-1 merge steps) (8) = 32
        self.downsampling_ratio: int = patch_size * (2 ** (self.num_stages - 1))

        # STFT dimensions after freq crop; must be divisible by patch_size
        self._freq_size: int = 1024  # 1025 Nyquist bin is always zeroed, safe to drop
        self._time_size: int = 128

        # ------------------------------------------------------------------ #
        # PatchEmbed                                                           #
        # Conv2d(in_channels, embed_dim, k=patch_size, s=patch_size)          #
        # (B, 2, 1024, 128) → flatten → (B, 8192, 48)  grid (256, 32)        #
        # ------------------------------------------------------------------ #
        self.patch_embed = PatchEmbed(
            img_size=(self._freq_size, self._time_size),
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        # Compute per-stage input grid dimensions and channel widths.
        # Stage i receives tokens on a grid of size (freq_grid>>i, time_grid>>i).
        # PatchMerging (applied at the END of stages 0..num_stages-2) halves the grid
        # and doubles the channels, so stage i sees dim = embed_dim * 2^i.
        freq_g = self._freq_size // patch_size   # 256 after PatchEmbed
        time_g = self._time_size // patch_size   # 32  after PatchEmbed
        stage_resolutions: List[Tuple[int, int]] = []
        for _ in range(self.num_stages):
            stage_resolutions.append((freq_g, time_g))
            freq_g //= 2
            time_g //= 2
        # stage_resolutions = [(256,32), (128,16), (64,8), (32,4)]

        stage_dims: List[int] = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        # stage_dims = [48, 96, 192, 384]

        # Stochastic depth: linear schedule 0 → drop_path_rate across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ------------------------------------------------------------------ #
        # Hierarchical stages                                                  #
        # ------------------------------------------------------------------ #
        self.stages = nn.ModuleList()
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
                # PatchMerging after every stage except the last
                downsample=PatchMerging if i < self.num_stages - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)
            block_idx += depths[i]

        # Grid and channel dim seen at the OUTPUT of the final stage (no merge)
        self._final_dim: int = stage_dims[-1]                   # 384
        self._final_resolution: Tuple[int, int] = stage_resolutions[-1]  # (32, 4)

        # ------------------------------------------------------------------ #
        # Output head                                                          #
        # LayerNorm → Linear(384 → dimension) → reshape to (B, dim, H, W)    #
        # ------------------------------------------------------------------ #
        self.norm = nn.LayerNorm(self._final_dim)
        self.head = nn.Linear(self._final_dim, dimension)

        # ------------------------------------------------------------------ #
        # Weight initialisation                                                #
        # ------------------------------------------------------------------ #
        # Truncated-normal for Linear, ones/zeros for LayerNorm
        self.apply(init_swin_weights)
        # Post-norm residual scaling: zero init on every SwinBlock's norms
        # (from original Swin V2; stabilises training at depth)
        for stage in self.stages:
            stage._init_respostnorm()

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encode a stereo STFT spectrogram to a 2-D latent token map.

        Args:
            x: ``(B, 2, F, T)`` float32, F can be 1025 (raw STFT output).

        Returns:
            latents: ``(B, dimension, H_lat, W_lat)`` = ``(B, 128, 32, 4)``
            info: ``{"feature_shape": (H_lat, W_lat)}`` consumed by AutoEncoder
                  to populate the decoder's ``feature_shape`` attribute.
        """
        B = x.shape[0]

        # --- Freq crop ---------------------------------------------------
        # 1025 → 1024: the Nyquist bin (index 1024) carries no energy after the
        # pipeline's align_freq_bins() and is safe to discard.
        # 1024 = 2^10 is optimal for Swin's powers-of-2 hierarchical downsampling.
        x = x[..., :self._freq_size, :]                          # (B, 2, 1024, 128)

        # --- Patch embedding ---------------------------------------------
        # Conv2d(2→48, k=4, s=4) + flatten spatial dims
        x = self.patch_embed(x)                                  # (B, 8192, 48)

        # --- Hierarchical Swin stages ------------------------------------
        # Each stage runs SwinTransformerBlocks; all but the last also apply
        # PatchMerging at the end (halves grid, doubles channels).
        for stage in self.stages:
            x = stage(x)
        # After stage 1 + PatchMerging: (B, 2048,  96)  grid (128, 16)
        # After stage 2 + PatchMerging: (B,  512, 192)  grid  (64,  8)
        # After stage 3 + PatchMerging: (B,  128, 384)  grid  (32,  4)
        # After stage 4 (no merge):     (B,  128, 384)  grid  (32,  4)

        # --- Normalise + project -----------------------------------------
        x = self.norm(x)                                         # (B, 128, 384)
        x = self.head(x)                                         # (B, 128, dimension)

        # --- Reshape to 2-D spatial map ----------------------------------
        H_lat, W_lat = self._final_resolution                    # 32, 4
        x = x.transpose(1, 2)                                    # (B, dimension, 128)
        x = x.reshape(B, self.dimension, H_lat, W_lat)           # (B, 128, 32, 4)

        return x, {"feature_shape": (H_lat, W_lat)}
