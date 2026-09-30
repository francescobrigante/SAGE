
# =============================================================================
# HIL discriminator family (HILDiscriminator, MultiFilterBankDiscriminator,
# FilterBankDiscriminator, ChromaDiscriminator).
# Ported from stable-audio-tools. Uses PQMF subband analysis + conv2d stacks.
# =============================================================================

import math
import typing as tp
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

from sage.nn.discriminators.types import get_hinge_losses, get_relativistic_losses, get_sigmoid_relgan_losses
from sage.nn.discriminators.pretransforms import fold_channels_into_batch, unfold_channels_from_batch


# ---------------------------------------------------------------------------
# PQMF (Pseudo-Quadrature Mirror Filter Bank)
# ---------------------------------------------------------------------------

def _design_prototype_filter(taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0) -> np.ndarray:
    """Kaiser-windowed prototype filter for PQMF."""
    from scipy.signal.windows import kaiser
    assert taps % 2 == 0, 'taps must be even'
    assert 0.0 < cutoff_ratio < 1.0, 'cutoff_ratio must be in (0, 1)'
    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1)
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (n - 0.5 * taps)) / (np.pi * (n - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio
    w = kaiser(taps + 1, beta)
    return h_i * w


class PQMF(nn.Module):
    """
    Pseudo-Quadrature Mirror Filter Bank for sub-band analysis.

    Args:
        subbands: number of sub-bands
        taps: prototype filter length
        cutoff_freq: prototype filter cutoff ratio
        beta: Kaiser window beta
    """

    def __init__(self, subbands: int = 4, taps: int = 62, cutoff_freq: float = 0.142, beta: float = 9.0):
        super().__init__()
        h_proto = torch.from_numpy(
            _design_prototype_filter(taps, cutoff_freq, beta)
        ).float().unsqueeze(0)

        k = torch.arange(subbands, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(taps + 1, dtype=torch.float32).unsqueeze(0)
        filt = 2.0 * h_proto * torch.cos(
            (2 * k + 1) * np.pi / (2 * subbands) * (n - taps / 2)
            + (-1) ** k * np.pi / 4
        ).unsqueeze(1) * subbands ** 0.5

        self.taps = taps
        self.subbands = subbands
        self.register_buffer('pqmf_filter', filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.analysis(x)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] → [B, subbands, T//subbands]."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, self.pqmf_filter, stride=self.subbands, padding=self.taps // 2)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, subbands, T//subbands] → [B, 1, T]."""
        return F.conv_transpose1d(
            x, self.pqmf_filter,
            stride=self.subbands,
            padding=self.taps // 2,
            output_padding=self.subbands - 1,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


# ---------------------------------------------------------------------------
# FilterBankDiscriminator (PQMF + Conv2d stack)
# ---------------------------------------------------------------------------

class FilterBankDiscriminator(nn.Module):
    """
    One filter-bank discriminator branch.

    For period=1 skips PQMF and treats the signal as a single subband.
    For period>1 applies PQMF with `period` subbands.
    Uses a Conv2d stack over (subband × time) feature maps.
    """

    def __init__(
        self,
        period: int,
        taps: int = 0,
        beta: float = 0.0,
        cutoff_freq: float = 0.0,
        kernel_sizes: tp.List[int] = (5, 5, 5, 5, 5),
        strides: tp.List[int] = (3, 3, 3, 3, 3),
        channels: tp.List[int] = (32, 128, 256, 512, 1024, 1024),
        norm: str = 'weight_norm',
        in_channels: int = 1,
    ):
        super().__init__()
        self.period = period
        self.in_channels = in_channels

        if period == 1:
            self.pqmf = nn.Identity()
        else:
            assert taps > 0 and beta > 0.0 and cutoff_freq > 0.0
            self.pqmf = PQMF(subbands=period, taps=taps, beta=beta, cutoff_freq=cutoff_freq)

        norm_f = weight_norm if norm == 'weight_norm' else torch.nn.utils.spectral_norm

        c_in = in_channels
        self.convs = nn.ModuleList()
        for ch, s, k in zip(channels, strides, kernel_sizes):
            conv = nn.Conv2d(c_in, ch, (1, k), (1, s), padding=(0, _get_padding(k)))
            self.convs.append(norm_f(conv))
            c_in = ch

        self.conv_post = norm_f(nn.Conv2d(c_in, 1, (1, 3), 1, padding=(0, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = fold_channels_into_batch(x)                                      # (B*C, T)
        x = self.pqmf(x)                                                     # (B*C, subbands, T')
        x = unfold_channels_from_batch(x, self.in_channels)                 # (B, C, subbands, T')
        if self.period == 1:
            x = x.unsqueeze(2)                                               # (B, C, 1, T)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1)
        return x, fmap


# ---------------------------------------------------------------------------
# MultiFilterBankDiscriminator
# ---------------------------------------------------------------------------

class MultiFilterBankDiscriminator(nn.Module):
    """Six-period PQMF filter-bank discriminator (periods [1,2,3,5,7,11])."""

    def __init__(
        self,
        periods: tp.List[int] = (1, 2, 3, 5, 7, 11),
        taps: int = 256,
        beta: float = 8.0,
        cutoff_freqs: tp.List[float] = (0.0, 0.253881, 0.170546, 0.103881, 0.075310, 0.049338),
        kernel_sizes: tp.List[int] = (5, 5, 5, 5, 5),
        strides: tp.List[int] = (3, 3, 3, 3, 1),
        channels: tp.List[int] = (32, 128, 512, 1024, 1024),
        norm: str = 'weight_norm',
        in_channels: int = 1,
        **kwargs,
    ):
        assert len(strides) == len(channels) == len(kernel_sizes)
        super().__init__()
        self.discriminators = nn.ModuleList([
            FilterBankDiscriminator(
                p, taps=taps, beta=beta, cutoff_freq=c,
                kernel_sizes=kernel_sizes, strides=strides,
                channels=channels, norm=norm, in_channels=in_channels,
            )
            for p, c in zip(periods, cutoff_freqs)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List, List]:
        logits, fmaps = [], []
        for disc in self.discriminators:
            y_d, fmap = disc(x)
            logits.append(y_d)
            fmaps.append(fmap)
        return logits, fmaps


# ---------------------------------------------------------------------------
# ChromaDiscriminator
# ---------------------------------------------------------------------------

class ChromaDiscriminator(nn.Module):
    """
    Chroma spectrogram + Conv2d stack discriminator.
    Uses torchaudio.prototype.transforms.ChromaSpectrogram.
    """

    def __init__(
        self,
        n_chroma: int,
        sample_rate: int,
        kernel_sizes: tp.List[int] = (5, 5, 5, 5, 5),
        strides: tp.List[int] = (3, 3, 3, 3, 3),
        channels: tp.List[int] = (32, 128, 256, 512, 1024, 1024),
        norm: str = 'weight_norm',
        in_channels: int = 1,
        **kwargs,
    ):
        super().__init__()
        from torchaudio.prototype.transforms import ChromaSpectrogram
        self.chroma = ChromaSpectrogram(
            sample_rate=sample_rate, n_fft=4096, n_chroma=n_chroma, normalized=True)
        self.in_channels = in_channels

        norm_f = weight_norm if norm == 'weight_norm' else torch.nn.utils.spectral_norm

        c_in = in_channels
        self.convs = nn.ModuleList()
        for ch, s, k in zip(channels, strides, kernel_sizes):
            conv = nn.Conv2d(c_in, ch, (1, k), (1, s), padding=(0, _get_padding(k)))
            self.convs.append(norm_f(conv))
            c_in = ch
        self.conv_post = norm_f(nn.Conv2d(c_in, 1, (1, 3), 1, padding=(0, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = fold_channels_into_batch(x)                                      # (B*C, T)
        x = self.chroma(x)                                                   # (B*C, n_chroma, M)
        x = unfold_channels_from_batch(x, self.in_channels)                 # (B, C, n_chroma, M)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1)
        return x, fmap


# ---------------------------------------------------------------------------
# HILDiscriminator
# ---------------------------------------------------------------------------

class HILDiscriminator(nn.Module):
    """
    Hybrid-Input-Level discriminator: MS-STFT + MultiFilterBank + optional Chroma.

    Exposes the standard .loss(reals, fakes) → (dis_loss, adv_loss, fm_dist) API.

    Args:
        in_channels: audio channels
        normalize_losses: normalise FM distance by signal magnitude
        loss_type: 'hinge' | 'rpgan' | 'sigmoid_rpgan'
        n_chroma: if > 0, add a ChromaDiscriminator branch
        sample_rate: required for ChromaDiscriminator
    """

    def __init__(
        self,
        in_channels: int = 2,
        normalize_losses: bool = False,
        loss_type: tp.Literal['hinge', 'rpgan', 'sigmoid_rpgan'] = 'rpgan',
        n_chroma: int = 0,
        sample_rate: int = 44100,
        *args,
        **kwargs,
    ):
        super().__init__()
        from sage.nn.discriminators.encodec import MultiScaleSTFTDiscriminator
        self.stft_discriminators = MultiScaleSTFTDiscriminator(
            *args, in_channels=in_channels, **kwargs)
        self.fb_discriminators = MultiFilterBankDiscriminator(
            *args, in_channels=in_channels, **kwargs)

        if n_chroma > 0:
            self.chroma_discriminator = ChromaDiscriminator(
                n_chroma=n_chroma, sample_rate=sample_rate,
                in_channels=in_channels, **kwargs)

        self.normalize_losses = normalize_losses
        self.loss_type = loss_type
        self.fm_reduction = (
            (lambda x, y: (x - y).abs().mean() / (x.abs().mean() + 1e-3))
            if normalize_losses else
            (lambda x, y: (x - y).abs().mean())
        )

    def forward(self, x: torch.Tensor) -> Tuple[List, List]:
        logits, features = self.stft_discriminators(x)
        lg_fb, fm_fb = self.fb_discriminators(x)
        logits.extend(lg_fb)
        features.extend(fm_fb)
        if hasattr(self, 'chroma_discriminator'):
            lg_ch, fm_ch = self.chroma_discriminator(x)
            logits.append(lg_ch)
            features.append(fm_ch)
        return logits, features

    def loss(
        self,
        reals: torch.Tensor,
        fakes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (dis_loss, adv_loss, feature_matching_distance)."""
        fm_dist = torch.tensor(0., device=reals.device)
        dis_loss = torch.tensor(0., device=reals.device)
        adv_loss = torch.tensor(0., device=reals.device)

        logits_real, feats_real = self.forward(reals)
        logits_fake, feats_fake = self.forward(fakes)

        for i, (sf_real, sf_fake) in enumerate(zip(feats_real, feats_fake)):
            if len(sf_real) > 0:
                fm_dist = fm_dist + sum(
                    map(self.fm_reduction, sf_real, sf_fake)
                ) / len(sf_real)

            if self.loss_type == 'hinge':
                _dis, _adv = get_hinge_losses(logits_real[i], logits_fake[i])
            elif self.loss_type == 'rpgan':
                _dis, _adv = get_relativistic_losses(logits_real[i], logits_fake[i])
            elif self.loss_type == 'sigmoid_rpgan':
                _dis, _adv = get_sigmoid_relgan_losses(logits_real[i], logits_fake[i])
            else:
                raise ValueError(f'Unknown loss_type: {self.loss_type!r}')

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        fm_dist = torch.nan_to_num(fm_dist, nan=0., posinf=0.)
        n = max(len(logits_real), 1)
        return dis_loss / n, adv_loss / n, fm_dist / n
