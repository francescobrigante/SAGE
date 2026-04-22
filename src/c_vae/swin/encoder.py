# ===============================================================
# SwinEncoder: Swin Transformer V2 encoder adapted for audio STFT.
# Accepts (B, in_channels, F, T) float32 stereo STFT 
# and outputs (B, out_channels, H_lat*W_lat).
# ===============================================================

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.models.implementations.abstract_ae import AbstractEncoder
from .swin_stage import SwinStage
from .patches import PatchEmbed, PatchMerging
from .utils import init_swin_weights, make_norm


class SwinEncoder(AbstractEncoder):
    """Swin Transformer V2 encoder for audio STFT spectrograms (Exp 0 — real-valued).

    Differences from the original ImageNet SwinTransformerV2:
      - in_chans=2 (stereo float STFT) instead of 3 (RGB)
      - window_size=8 (divides all stage grids) instead of 7 (ImageNet 224px)
      - Freq crop 1025 -> 1024 applied at the forward boundary before PatchEmbed
      - No classification head; instead a Linear projection to dimension channels
        followed by a transpose to a 1-D latent sequence (B, dimension, H_lat*W_lat)

    Architecture (embed_dim=48, depths=[2,2,4,2], patch_size=4):

                                            C,    F,    T      grid
        Input                         (B,   4, 1024,  128)
        
        PatchEmbed  Conv2d(4->48,k=4) (B, F*T=8192,   C=48) (256, 32)
        Stage 1  2x SwinBlock         (B, 8192,   48)       (256, 32)
            └── PatchMerging          (B, 2048,  96)        (128, 16)
        Stage 2  2x SwinBlock         (B, 2048,   96)       (128, 16)
            └── PatchMerging          (B,  512, 192)        (64,  8)
        Stage 3  4x SwinBlock         (B,  512,  192)       (64,  8)
            └── PatchMerging          (B,  128, 384)        (32,  4)
        Stage 4  2x SwinBlock         (B,  128,  384)       (32,  4)
            └── (no downsample)
        LayerNorm + Linear(384→dim)   (B,  128,  out_chan)  (32,  4)
        Transpose                     (B,  out_chan,  128) = encoder output

    Args:
        in_channels: Input spectrogram channels (2 = stereo STFT). Aliased to
            input_size for the AbstractEncoder contract.
        embed_dim: Base token dimension at stage 0. Doubles at each PatchMerging.
        depths: Number of SwinTransformerBlocks per stage.
        num_heads: Number of self-attention heads per stage.
        window_size: Local attention window size. Must divide every stage grid
            dimension. Default 8 works for the (1024, 128) STFT grid.
        patch_size: Kernel/stride of the PatchEmbed Conv2d.
        dimension: Total encoder output channels = parameters_to_predict * latent_channels.
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
        dimension: int = 128,       # parameters_to_predict * latent_channels
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
        is_complex: bool = False,
        complex_activation: str = "ComplexGELU1d",
    ) -> None:

        super().__init__(input_size=in_channels, is_complex=is_complex)

        self.embed_dim = embed_dim                          # base channel width (doubles per merge)
        self.depths = list(depths)                          # blocks per stage, e.g. [2, 2, 4, 2]
        self.num_heads = list(num_heads)                    # attention heads per stage
        self.window_size = window_size                      # local window size (must divide all grids)
        self.patch_size = patch_size                        # PatchEmbed Conv2d kernel/stride
        self.dimension = dimension                          # total output channels
        self.num_stages = len(depths)                       # number of hierarchical stages (4)
        self.fused_window_process = fused_window_process    # use fused CUDA kernel (CUDA only)

        # Total spatial downsampling:
        #   patch_size (=4) × 2^(merge steps = num_stages (=4) - 1 ) (=8) = 32
        self.downsampling_ratio: int = patch_size * (2 ** (self.num_stages - 1))

        # STFT dimensions after freq crop; must be divisible by patch_size
        self._freq_size: int = 1024  # 1025 Nyquist bin is always zeroed, safe to drop
        self._time_size: int = 128

        # ------------------------------ PatchEmbed ------------------------------
        # (B, 2, 1024, 128) -> Conv2d and flatten -> (B, 8192, 48)  grid (256, 32)
        # ------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(
            img_size=(self._freq_size, self._time_size),
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            is_complex=is_complex,
        )

        # Compute per-stage input grid dimensions (H,W) and channel widths
        freq_g = self._freq_size // patch_size   # 256 after PatchEmbed
        time_g = self._time_size // patch_size   # 32  after PatchEmbed
        
        # e.g. stage_resolutions = [(256,32), (128,16), (64,8), (32,4)]
        stage_resolutions: List[Tuple[int, int]] = []
        for _ in range(self.num_stages):
            stage_resolutions.append((freq_g, time_g))
            freq_g //= 2
            time_g //= 2
            
        # e.g. stage_dims = [48, 96, 192, 384]
        stage_dims: List[int] = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # Stochastic depth: linear schedule 0 to drop_path_rate across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # --------------------------- Swin Stages ---------------------------
        # Stacks of SwinTransformerBlocks with alternating W-MSA / SW-MSA,
        # followed by PatchMerging spatial downsamplers in all but the 
        # last stage.
        # -------------------------------------------------------------------
        self.stages = nn.ModuleList()
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
                # PatchMerging after every stage except the last
                downsample=PatchMerging if i < self.num_stages - 1 else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                is_complex=is_complex,
                complex_activation=complex_activation,
            )
            self.stages.append(stage)
            block_idx += depths[i]

        # Grid and channel dim seen at the OUTPUT of the final stage (no merge)
        self._final_dim: int = stage_dims[-1]                            # 384
        self._final_resolution: Tuple[int, int] = stage_resolutions[-1]  # (32, 4)

        # Final norm + projection to latent dimension
        self.norm = make_norm(self._final_dim, is_complex)
        self.head = NormLinear(self._final_dim, dimension, is_complex=is_complex)

        # Weight init
        self.apply(init_swin_weights)
        for stage in self.stages:
            stage._init_respostnorm()


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encode a stereo STFT spectrogram to a 1-D latent sequence.

        Args:
            x: (B, in_channels, F, T) float32, F can be 1025 (raw STFT output).

        Returns:
            latents: (B, dimension, H_lat*W_lat) = (B, 128, 128)
            info: {"feature_shape": (H_lat, W_lat)} passed to the decoder
                  to allow optional reconstruction of the 2-D spatial layout.
        """

        # freq crop from 1025 to 1024
        x = x[..., :self._freq_size, :]        # (B, 2, 1024, 128)
        # PatchEmbed
        x = self.patch_embed(x)                # (B, 8192, 48)
        # 4 Swin Stages 
        for stage in self.stages:
            x = stage(x)
        # Stage 1 + PatchMerging:                 (B, 2048, 96)  grid (128, 16)
        # Stage 2 + PatchMerging:                 (B,  512, 192)  grid  (64,  8)
        # Stage 3 + PatchMerging:                 (B,  128, 384)  grid  (32,  4)
        # Stage 4 (no merge):                     (B,  128, 384)  grid  (32,  4)

        x = self.norm(x)                        # (B, 128, 384)
        x = self.head(x)                        # (B, 128, out_channels)

        x = x.transpose(1, 2)                   # (B, out_channels, 128=H*W latent)
        
        H_lat, W_lat = self._final_resolution   # 32, 4

        # [OPTIONAL] Reshape back to 2D spatial grid
        # x = x.reshape(B, self.dimension, H_lat, W_lat)         # (B, dimension, H=32, W=4)

        return x, {"feature_shape": (H_lat, W_lat)} # (B, out_channels, 128=H*W latent)