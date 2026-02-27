from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_kl import Decoder as HFDecoderKL
from diffusers.models.autoencoders.autoencoder_kl import Encoder as HFEncoderKL

from ar_spectra.models.autoencoders.abstract_ae import AbastractDecoder, AbstractEncoder


class HFAutoencoderKLEncoder(AbstractEncoder):
    """Wrapper around diffusers' ``Encoder`` (+quant_conv) exposing the repo's encoder contract.

    The module outputs concatenated ``[mu | logvar]`` so that the built-in
    :class:`ar_spectra.models.bottlenecks.VAEBottleneck` can sample latents.
    """

    def __init__(
        self,
        *,
        input_size: int,
        is_complex: bool,
        latent_channels: int = 4,
        down_block_types: Sequence[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        block_out_channels: Sequence[int] = (128, 128, 256, 256),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__(input_size=input_size, is_complex=is_complex)

        self.latent_channels = int(latent_channels)
        self.dimension = 2 * self.latent_channels  # used by AutoEncoder sanity checks
        self.scaling_factor = float(scaling_factor)

        # Rough stride accounting for AutoEncoder.infer_downsampling_ratio
        self.ratios = [{"stride": 2} for _ in range(max(len(block_out_channels) - 1, 0))]

        self.encoder = HFEncoderKL(
            in_channels=input_size,
            out_channels=self.latent_channels,
            down_block_types=tuple(down_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = nn.Conv2d(2 * self.latent_channels, 2 * self.latent_channels, kernel_size=1)

    def output_size(self) -> int:
        return self.dimension

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Optional[Tuple[int, int]]]]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        self.last_feature_shape = mean.shape[-2:]
        info = {"feature_shape": self.last_feature_shape, "scaling_factor": self.scaling_factor}
        return torch.cat([mean, logvar], dim=1), info


class HFAutoencoderKLDecoder(AbastractDecoder):
    """Decoder companion for :class:`HFAutoencoderKLEncoder`.

    Expects sampled latents with ``latent_channels`` channels. The usual SD
    scaling factor is applied before feeding latents to the HF decoder.
    """

    def __init__(
        self,
        *,
        channels: int,
        is_complex: bool,
        latent_channels: int = 4,
        up_block_types: Sequence[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Sequence[int] = (128, 128, 256, 256),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        use_post_quant_conv: bool = True,
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)
        self.input_size = int(latent_channels)
        self.scaling_factor = float(scaling_factor)

        self.post_quant_conv: nn.Module
        if use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(self.input_size, self.input_size, kernel_size=1)
        else:
            self.post_quant_conv = nn.Identity()

        self.decoder = HFDecoderKL(
            in_channels=self.input_size,
            out_channels=channels,
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

    def forward(self, latents: torch.Tensor, encoder_info: Optional[Dict] = None) -> torch.Tensor:
        z = latents / self.scaling_factor if self.scaling_factor != 0 else latents
        z = self.post_quant_conv(z)
        return self.decoder(z)
