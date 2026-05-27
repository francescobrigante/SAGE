
# =============================================================================
# Transformer-based discriminators ported from stable-audio-tools:
# TransformerDiscriminator, TransformerMultiSTFTDiscriminator,
# TransformerMultiPatchedDiscriminator, TransformerMultiWaveletDiscriminator,
# and the top-level MultiTransformerDiscriminator with .loss() API.
# =============================================================================

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pretransforms import ComplexSTFTPretransform, PatchedPretransform, WaveletPretransform, ChromaPretransform
from ...blocks.transformer_sat import TransformerResamplingBlock
from .types import get_hinge_losses, get_relativistic_losses, get_sigmoid_relgan_losses


# ---------------------------------------------------------------------------
# Dimension heuristic
# ---------------------------------------------------------------------------

def get_transformer_dim(
    c: int,
    *,
    k: float = 16.0,
    min_d: int = 192,
    max_d: int = 512,
    head_dim: int = 64,
) -> int:
    """Return transformer d_model as a multiple of head_dim via saturating √-law."""
    d_raw = k * math.sqrt(max(1.0, float(c)))
    q = int(math.floor(d_raw / head_dim + 0.5))
    q_min = math.ceil(min_d / head_dim)
    q_max = math.floor(max_d / head_dim)
    q = max(q_min, min(q_max, q))
    return q * head_dim


# ---------------------------------------------------------------------------
# Single TransformerDiscriminator branch
# ---------------------------------------------------------------------------

class TransformerDiscriminator(nn.Module):
    """
    One discriminator branch: pretransform → TransformerResamplingBlock → Conv1d head.

    Returns (logits [B, 1, T'], fmaps list).
    """

    def __init__(
        self,
        in_dim: int,
        transformer_dim: int,
        stride: int,
        pretransform: nn.Module,
        sliding_window: tp.List[int] = (1, 1),
        depth: int = 3,
        checkpointing: bool = False,
        differential: bool = True,
        max_depth_feature: int = 2,
        ff_mult: float = 1.5,
        dyt: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.pretransform = pretransform
        self.max_depth_feature = max_depth_feature
        self.discriminator = TransformerResamplingBlock(
            in_channels=in_dim,
            out_channels=transformer_dim,
            stride=stride,
            sliding_window=list(sliding_window),
            transformer_depth=depth,
            checkpointing=checkpointing,
            differential=differential,
            ff_mult=ff_mult,
            dyt=dyt,
            dim_heads=64,
        )
        self.decision_map = nn.Conv1d(transformer_dim, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        # Pad to pretransform's downsampling_ratio
        dr = self.pretransform.downsampling_ratio
        if x.shape[-1] % dr != 0:
            pad = ((x.shape[-1] // dr) + 1) * dr - x.shape[-1]
            x = F.pad(x, (0, pad), mode='constant')

        x = self.pretransform(x)                                             # (B, encoded_ch, M)
        x, fmap = self.discriminator(x, return_features=True)               # (B, d_model, M//stride)
        logits = self.decision_map(x)                                        # (B, 1, M//stride)
        return logits, fmap[:min(len(fmap), self.max_depth_feature)]


# ---------------------------------------------------------------------------
# Multi-scale STFT transformer discriminator
# ---------------------------------------------------------------------------

class TransformerMultiSTFTDiscriminator(nn.Module):
    """Three STFT-pretransform TransformerDiscriminators at n_ffts=[4096,1024,128]."""

    def __init__(
        self,
        in_channels: int = 1,
        n_ffts: tp.List[int] = (4096, 1024, 128),
        depths: tp.List[int] = (3, 3, 3),
        strides: tp.List[int] = (2, 32, 2),
        sliding_window_widths: tp.List[int] = (2, 4, 3),
        **kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i, n_fft in enumerate(n_ffts):
            pt = ComplexSTFTPretransform(
                in_channels, n_fft,
                ema_flatten=False, use_compander=False,
                demodulate=True, center=False,
            )
            d_model = get_transformer_dim(pt.encoded_channels)
            disc = TransformerDiscriminator(
                pt.encoded_channels, d_model, strides[i], pt,
                sliding_window=[sliding_window_widths[i], sliding_window_widths[i]],
                depth=depths[i], **kwargs,
            )
            self.discriminators.append(disc)

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for disc in self.discriminators:
            l, f = disc(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# ---------------------------------------------------------------------------
# Multi-scale patched transformer discriminator
# ---------------------------------------------------------------------------

class TransformerMultiPatchedDiscriminator(nn.Module):
    """Three patch-pretransform TransformerDiscriminators at patch_sizes=[29,443,953]."""

    def __init__(
        self,
        in_channels: int = 1,
        patch_sizes: tp.List[int] = (29, 443, 953),
        depths: tp.List[int] = (3, 3, 3),
        strides: tp.List[int] = (2, 16, 2),
        sliding_window_widths: tp.List[int] = (4, 8, 1),
        **kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i, ps in enumerate(patch_sizes):
            pt = PatchedPretransform(in_channels, ps)
            d_model = get_transformer_dim(pt.encoded_channels)
            disc = TransformerDiscriminator(
                pt.encoded_channels, d_model, strides[i], pt,
                sliding_window=[sliding_window_widths[i], sliding_window_widths[i]],
                depth=depths[i], **kwargs,
            )
            self.discriminators.append(disc)

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for disc in self.discriminators:
            l, f = disc(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# ---------------------------------------------------------------------------
# Multi-scale wavelet transformer discriminator
# ---------------------------------------------------------------------------

class TransformerMultiWaveletDiscriminator(nn.Module):
    """Three wavelet-pretransform TransformerDiscriminators at levels=[4,8,10]."""

    def __init__(
        self,
        in_channels: int = 1,
        levels: tp.List[int] = (4, 8, 10),
        depths: tp.List[int] = (3, 3, 3),
        strides: tp.List[int] = (2, 32, 2),
        sliding_window_widths: tp.List[int] = (3, 8, 1),
        **kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i, lvl in enumerate(levels):
            pt = WaveletPretransform(in_channels, lvl)
            d_model = get_transformer_dim(pt.encoded_channels)
            disc = TransformerDiscriminator(
                pt.encoded_channels, d_model, strides[i], pt,
                sliding_window=[sliding_window_widths[i], sliding_window_widths[i]],
                depth=depths[i], **kwargs,
            )
            self.discriminators.append(disc)

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for disc in self.discriminators:
            l, f = disc(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# ---------------------------------------------------------------------------
# Multi-scale chroma transformer discriminator
# ---------------------------------------------------------------------------

class TransformerMultiChromaDiscriminator(nn.Module):
    """Three chroma-pretransform TransformerDiscriminators at octave centres [1, 5, 9].

    Each branch uses a ChromaPretransform with a different (ctroct, octwidth) pair,
    giving the discriminator pitch-class sensitivity at low, mid, and high octaves.
    Matches SAME Config-2 (transformer discriminator) architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        centres: tp.List[float] = (1.0, 5.0, 9.0),       # octave centres
        octwidths: tp.List[float] = (1.0, 1.5, 1.0),      # Gaussian octave widths
        n_chroma: int = 64,
        sample_rate: int = 44100,
        depths: tp.List[int] = (3, 3, 3),
        strides: tp.List[int] = (16, 8, 2),
        sliding_window_widths: tp.List[int] = (3, 3, 3),
        **kwargs,
    ):
        super().__init__()
        assert len(centres) == len(octwidths) == len(depths) == len(strides) == len(sliding_window_widths)
        self.discriminators = nn.ModuleList()
        for i, (ctroct, octwidth) in enumerate(zip(centres, octwidths)):
            pt = ChromaPretransform(
                in_channels, n_chroma=n_chroma, sample_rate=sample_rate,
                ctroct=ctroct, octwidth=octwidth,
            )
            d_model = get_transformer_dim(pt.encoded_channels)
            disc = TransformerDiscriminator(
                pt.encoded_channels, d_model, strides[i], pt,
                sliding_window=[sliding_window_widths[i], sliding_window_widths[i]],
                depth=depths[i], **kwargs,
            )
            self.discriminators.append(disc)

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for disc in self.discriminators:
            l, f = disc(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# ---------------------------------------------------------------------------
# Top-level MultiTransformerDiscriminator
# ---------------------------------------------------------------------------

class MultiTransformerDiscriminator(nn.Module):
    """
    Container for STFT + patched + mfb (PQMF or wavelet) + chroma discriminator branches.
    Exposes the standard .loss(reals, fakes) → (dis_loss, adv_loss, fm_dist) API.

    Branches are enabled via their respective `*_kwargs` dicts with an
    ``enabled`` key (default True for patched, False for others).
    Mirrors the SAT MultiTransformerDiscriminator interface exactly.

    Args:
        in_channels: audio channels (1=mono, 2=stereo)
        normalize_losses: normalise FM distance by signal magnitude
        loss_type: 'hinge' | 'rpgan' | 'sigmoid_rpgan'
        stft_kwargs: forwarded to TransformerMultiSTFTDiscriminator; needs enabled=True
        patched_kwargs: forwarded to TransformerMultiPatchedDiscriminator; enabled by default
        mfb_kwargs: filter-bank branch; needs enabled=True.
            use_HIL=True  → convolutional MultiFilterBankDiscriminator (PQMF, SAME Config-2)
            use_HIL=False → TransformerMultiWaveletDiscriminator (wavelet TRB)
        chroma_kwargs: forwarded to TransformerMultiChromaDiscriminator; needs enabled=True
    """

    def __init__(
        self,
        in_channels: int = 2,
        normalize_losses: bool = False,
        loss_type: tp.Literal['hinge', 'rpgan', 'sigmoid_rpgan'] = 'rpgan',
        stft_kwargs: dict = {},
        patched_kwargs: dict = {},
        mfb_kwargs: dict = {},
        chroma_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__()

        if stft_kwargs.pop('enabled', False):
            self.stft_discriminators = TransformerMultiSTFTDiscriminator(
                in_channels=in_channels, **stft_kwargs)

        if patched_kwargs.pop('enabled', True):
            self.patched_discriminators = TransformerMultiPatchedDiscriminator(
                in_channels=in_channels, **patched_kwargs)

        if mfb_kwargs.pop('enabled', False):
            use_hil = mfb_kwargs.pop('use_HIL', False)
            if use_hil:
                # Convolutional PQMF filter-bank (SAT HIL, SAME Config-2 "retained PQMF")
                from .hil import MultiFilterBankDiscriminator
                self.fb_discriminators = MultiFilterBankDiscriminator(
                    in_channels=in_channels, **mfb_kwargs)
            else:
                # Transformer-based wavelet discriminator
                self.fb_discriminators = TransformerMultiWaveletDiscriminator(
                    in_channels=in_channels, **mfb_kwargs)

        if chroma_kwargs.pop('enabled', False):
            self.chroma_discriminators = TransformerMultiChromaDiscriminator(
                in_channels=in_channels, **chroma_kwargs)

        self.normalize_losses = normalize_losses
        self.loss_type = loss_type
        self.fm_reduction = (
            (lambda x, y: (x - y).abs().mean() / (x.abs().detach().mean() + 1e-3))
            if normalize_losses else
            (lambda x, y: (x - y).abs().mean())
        )
        self.last_disc_loss = 1.0  # EMA for noise injection when disc is too strong

    def forward(self, x: torch.Tensor) -> tp.Tuple[tp.List, tp.List]:
        # Slight noise injection when discriminator is dominating
        if self.last_disc_loss < 0.1:
            x = x + torch.randn_like(x) * x.std() * 10.0 * (0.1 - self.last_disc_loss)
        logits, features = [], []
        if hasattr(self, 'stft_discriminators'):
            lg, fm = self.stft_discriminators(x)
            logits.extend(lg); features.extend(fm)
        if hasattr(self, 'patched_discriminators'):
            lg, fm = self.patched_discriminators(x)
            logits.extend(lg); features.extend(fm)
        if hasattr(self, 'fb_discriminators'):
            lg, fm = self.fb_discriminators(x)
            logits.extend(lg); features.extend(fm)
        if hasattr(self, 'chroma_discriminators'):
            lg, fm = self.chroma_discriminators(x)
            logits.extend(lg); features.extend(fm)
        return logits, features

    def loss(
        self,
        reals: torch.Tensor,
        fakes: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GAN losses.

        Returns:
            dis_loss: discriminator loss (averaged across scales)
            adv_loss: generator adversarial loss (averaged across scales)
            feature_matching_distance: feature matching loss (averaged across scales)
        """
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

            dis_loss = dis_loss + torch.nan_to_num(_dis, nan=0., posinf=0.)
            adv_loss = adv_loss + torch.nan_to_num(_adv, nan=0., posinf=0.)

        fm_dist = torch.nan_to_num(fm_dist, nan=0., posinf=0.)
        n = max(len(logits_real), 1)
        self.last_disc_loss = 0.99 * self.last_disc_loss + 0.01 * dis_loss.item() / n
        return dis_loss / n, adv_loss / n, fm_dist / n
