# =============================================================================
# HuggingFace Diffusers DCAE wrappers tailored for Eulero's dimension and config requirements.
# =============================================================================
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_dc import Decoder as HFDecoderDC
from diffusers.models.autoencoders.autoencoder_dc import Encoder as HFEncoderDC

from ar_spectra.models.implementations.abstract_ae import AbstractDecoder, AbstractEncoder


class HFAutoencoderDCEncoder(AbstractEncoder):
    """Diffusers DCAE encoder adapted to emit VAE moments.

    We add a lightweight ``1x1`` projection to generate ``[mu | logvar]`` so
    that the existing :class:`VAEBottleneck` can be used during training.
    
    Handles odd frequency dimensions (e.g., 1025 from STFT) by stripping the
    Nyquist bin (last bin) before encoding. The Nyquist bin typically contains
    negligible energy and can be safely zero-padded in the decoder.
    """

    def __init__(
        self,
        *,
        input_size: int,
        is_complex: bool,
        latent_channels: int = 32,
        attention_head_dim: int = 32,
        encoder_block_types: Union[str, Sequence[str]] = "ResBlock",
        encoder_block_out_channels: Sequence[int] = (128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block: Sequence[int] = (2, 2, 2, 3, 3, 3),
        encoder_qkv_multiscales: Sequence[Tuple[int, ...]] = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "pixel_unshuffle",
        encoder_out_shortcut: bool = True,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__(input_size=input_size, is_complex=is_complex)

        self.latent_channels = int(latent_channels)
        self.dimension = 2 * self.latent_channels
        self.scaling_factor = float(scaling_factor)

        # DCAE compresses spatial dims by 2^(n-1)
        self.ratios = [{"stride": 2} for _ in range(max(len(encoder_block_out_channels) - 1, 0))]

        self.encoder = HFEncoderDC(
            in_channels=input_size,
            latent_channels=self.latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
            out_shortcut=encoder_out_shortcut,
        )
        self.moments = nn.Conv2d(self.latent_channels, 2 * self.latent_channels, kernel_size=1)

    def output_size(self) -> int:
        return self.dimension

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Optional[Tuple[int, int]]]]:
        # x shape: [B, C, F, T] where F is frequency bins
        original_freq_bins = x.shape[-2]
        stripped_nyquist = False
        
        # If frequency dimension is odd, strip Nyquist bin (last bin) to make it even
        # This is needed for pixel_unshuffle downsampling
        # Nyquist bin typically contains ~0 energy due to anti-aliasing filters
        if original_freq_bins % 2 == 1:
            x = x[..., :-1, :]  # Remove Nyquist bin (last index)
            stripped_nyquist = True
        
        h = self.encoder(x)
        stats = self.moments(h)
        mean, logvar = torch.chunk(stats, 2, dim=1)

        self.last_feature_shape = mean.shape[-2:]
        info = {
            "feature_shape": self.last_feature_shape,
            "scaling_factor": self.scaling_factor,
            "original_freq_bins": original_freq_bins,
            "stripped_nyquist": stripped_nyquist,
        }
        return torch.cat([mean, logvar], dim=1), info


class HFAutoencoderDCDecoder(AbstractDecoder):
    """Decoder companion for :class:`HFAutoencoderDCEncoder`.

    Mirrors the original DCAE decoder; latents are optionally rescaled to
    preserve HF's variance convention.
    
    Handles restoration of Nyquist bin when the encoder stripped it due to odd
    frequency dimensions. The Nyquist bin is zero-padded since it typically
    contains negligible energy.
    """

    def __init__(
        self,
        *,
        channels: int,
        is_complex: bool,
        latent_channels: int = 32,
        attention_head_dim: int = 32,
        decoder_block_types: Union[str, Sequence[str]] = "ResBlock",
        decoder_block_out_channels: Sequence[int] = (128, 256, 512, 512, 1024, 1024),
        decoder_layers_per_block: Sequence[int] = (3, 3, 3, 3, 3, 3),
        decoder_qkv_multiscales: Sequence[Tuple[int, ...]] = ((), (), (), (5,), (5,), (5,)),
        upsample_block_type: str = "pixel_shuffle",
        decoder_norm_types: Union[str, Sequence[str]] = "rms_norm",
        decoder_act_fns: Union[str, Sequence[str]] = "silu",
        decoder_in_shortcut: bool = True,
        decoder_conv_act_fn: str = "relu",
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)
        self.input_size = int(latent_channels)
        self.scaling_factor = float(scaling_factor)

        self.decoder = HFDecoderDC(
            in_channels=channels,
            latent_channels=self.input_size,
            attention_head_dim=attention_head_dim,
            block_type=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
            in_shortcut=decoder_in_shortcut,
            conv_act_fn=decoder_conv_act_fn,
        )
        # Note: Nyquist bin restoration uses simple zero-padding (no learned params)
        # since the Nyquist bin typically contains negligible energy

    def forward(self, latents: torch.Tensor, encoder_info: Optional[Dict] = None) -> torch.Tensor:
        z = latents / self.scaling_factor if self.scaling_factor != 0 else latents
        decoded = self.decoder(z)
        
        # Restore Nyquist bin if it was stripped during encoding
        # Zero-padding is appropriate since Nyquist typically has ~0 energy
        if encoder_info is not None and encoder_info.get("stripped_nyquist", False):
            original_freq_bins = encoder_info.get("original_freq_bins")
            if original_freq_bins is not None and decoded.shape[-2] == original_freq_bins - 1:
                # Append zero-valued Nyquist bin at the end
                nyquist_bin = torch.zeros_like(decoded[..., :1, :])
                decoded = torch.cat([decoded, nyquist_bin], dim=-2)
        
        return decoded
