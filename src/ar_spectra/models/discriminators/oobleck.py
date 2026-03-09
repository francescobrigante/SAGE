
# =============================================================================
# Interface and implementation of discriminators derived from Oobleck for high audio fidelity.
# =============================================================================

from .types import BANDS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as tp
from typing import List, Tuple
from functools import reduce
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class OobleckDiscriminator(nn.Module):

    def __init__(
            self,
            in_channels=1,
            ):
        super().__init__()

        multi_scale_discriminator = MultiScaleDiscriminator(
            in_channels=in_channels,
            n_scales=3,
        )

        multi_period_discriminator = MultiPeriodDiscriminator(
            in_channels=in_channels,
            periods=[2, 3, 5, 7, 11]
        )

        # multi_resolution_discriminator = MultiScaleSTFTDiscriminator(
        #     filters=32,
        #     in_channels = in_channels,
        #     out_channels = 1,
        #     n_ffts = [2048, 1024, 512, 256, 128],
        #     hop_lengths = [512, 256, 128, 64, 32],
        #     win_lengths = [2048, 1024, 512, 256, 128]
        # )

        self.multi_discriminator = MultiDiscriminator(
            [multi_scale_discriminator, multi_period_discriminator], #, multi_resolution_discriminator],
            ["reals", "fakes"]
        )

    def loss(self, reals, fakes):
        inputs = {
            "reals": reals,
            "fakes": fakes,
        }

        inputs = self.multi_discriminator(inputs)

        scores_real = inputs["score_reals"]
        scores_fake = inputs["score_fakes"]

        features_real = inputs["features_reals"]
        features_fake = inputs["features_fakes"]

        dis_loss, gen_loss = get_hinge_losses(scores_real, scores_fake)
         
        feature_matching_distance = torch.tensor(0.)

        for _, (scale_real, scale_fake) in enumerate(zip(features_real, features_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda real, fake: abs(real - fake).mean(),
                    scale_real,
                    scale_fake,
                )) / len(scale_real)
            
        return dis_loss, gen_loss, feature_matching_distance

class MPD(nn.Module):
    def __init__(self, period, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv2d

        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(channels, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv1d

        self.convs = nn.ModuleList(
            [
                WNConv1d(channels, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
        channels: int = 1
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        from dac.model.discriminator import WNConv2d
        from audiotools import STFTParams

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        self.channels = channels

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        from audiotools import AudioSignal
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b ch f t c -> (b ch) c t f", ch=self.channels)
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap
