
# =============================================================================
# Discriminators and loss functions derived from the Descript Audio Codec (DAC) framework.
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

class DACDiscriminator(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p, channels=channels) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate, channels=channels) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands, channels=channels) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [checkpoint(d,x) for d in self.discriminators]
        return fmaps

class DACGANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, use_hinge: bool = False, **discriminator_kwargs):
        super().__init__()
        self.use_hinge = use_hinge
        self.discriminator = DACDiscriminator(**discriminator_kwargs)

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += (
                F.relu(x_fake[-1]).mean() +
                F.relu(1 - x_real[-1]).mean()
            ) if self.use_hinge else (
                (x_fake[-1] ** 2).mean() +
                ((1 - x_real[-1]) ** 2).mean()
            )
        loss_d /= len(d_fake)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += (
                F.relu(1 - x_fake[-1]).mean()
                if self.use_hinge else
                ((1 - x_fake[-1]) ** 2).mean()
            )

        n_discriminators = len(d_fake)
        loss_feature = 0
        for i in range(n_discriminators):
            # Average over N model layers (except for the last item, which is logits).
            n_layers = len(d_fake[i]) - 1
            loss_feature += sum(map(
                lambda j: F.l1_loss(d_fake[i][j], d_real[i][j].detach()),
                range(n_layers)
            )) / n_layers

        # Average over K discriminators.
        loss_feature = loss_feature / n_discriminators

        loss_g /= len(d_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance
