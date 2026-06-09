
# =============================================================================
# Discriminators and loss functions derived from the Descript Audio Codec (DAC) framework.
# =============================================================================

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .types import (
    BANDS,
    get_hinge_losses,
    get_relativistic_losses,
    get_sigmoid_relgan_losses,
)
from .oobleck import MPD, MSD, MRD

class DACDiscriminator(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
        use_checkpoint: bool = False,
        preprocess: bool = True,
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
        self.use_checkpoint = use_checkpoint
        # When False, skip DC-removal + per-sample peak-normalize (it alters loudness and
        # can interfere with SI-SDR scaling). Default True = faithful to the DAC paper.
        self.preprocess_enabled = preprocess
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
        if self.preprocess_enabled:
            x = self.preprocess(x)
        if self.use_checkpoint:
            fmaps = [checkpoint(d, x, use_reentrant=False) for d in self.discriminators]
        else:
            fmaps = [d(x) for d in self.discriminators]
        return fmaps

def _lsgan_losses(scores_real, scores_fake):
    """Least-squares GAN losses — the original DAC default (``use_hinge=False``)."""
    dis_loss = (scores_fake ** 2).mean() + ((1 - scores_real) ** 2).mean()
    gen_loss = ((1 - scores_fake) ** 2).mean()
    return dis_loss, gen_loss


# `hinge`/`rpgan`/`sigmoid_relgan` reuse the shared helpers in `types.py` (same family the
# Encodec/Transformer discriminators use), so all critics stay on one loss vocabulary.
# `lsgan` is kept inline to preserve the original DAC least-squares behaviour bit-for-bit.
_LOSS_DISPATCH = {
    "hinge": get_hinge_losses,
    "rpgan": get_relativistic_losses,
    "sigmoid_relgan": get_sigmoid_relgan_losses,
    "lsgan": _lsgan_losses,
}


class DACGANLoss(nn.Module):
    """
    DAC discriminator (MPD + complex multi-band MRD) wrapped behind a unified
    ``loss(reals, fakes) -> (dis_loss, adv_loss, feature_matching_distance)`` interface
    that mirrors ``EncodecDiscriminator.loss``. This lets it drop into the engine's single
    forward + phase-alternation step exactly like the other GAN critics, and exposes the
    relativistic (``rpgan``) formulation needed to compare disc *architecture* against N7
    at a fixed loss formulation.

    Args:
        loss_type: adversarial formulation — ``hinge`` | ``rpgan`` | ``sigmoid_relgan`` | ``lsgan``.
        use_hinge: legacy flag. If set, overrides ``loss_type`` (``True`` ⇒ ``hinge``,
            ``False`` ⇒ ``lsgan``, the original default).
        normalize_losses: if True, normalize the feature-matching distance by the real
            feature magnitude (EnCodec-style), otherwise plain L1.
    """

    def __init__(
        self,
        loss_type: tp.Literal["hinge", "rpgan", "sigmoid_relgan", "lsgan"] = "hinge",
        use_hinge: tp.Optional[bool] = None,
        normalize_losses: bool = False,
        **discriminator_kwargs,
    ):
        super().__init__()
        # Backward-compat: original API used `use_hinge` (False ⇒ LSGAN default).
        if use_hinge is not None:
            loss_type = "hinge" if use_hinge else "lsgan"
        if loss_type not in _LOSS_DISPATCH:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Choose from {list(_LOSS_DISPATCH)}."
            )
        self.loss_type = loss_type
        self._loss_fn = _LOSS_DISPATCH[loss_type]
        self.normalize_losses = normalize_losses
        self.fm_reduction = (
            (lambda x, y: (x - y).abs().mean() / (x.abs().mean() + 1e-3))
            if normalize_losses
            else (lambda x, y: (x - y).abs().mean())
        )
        self.discriminator = DACDiscriminator(**discriminator_kwargs)

    def forward(self, fake, real):
        return self.discriminator(fake), self.discriminator(real)

    def _feature_matching(self, fr, fg):
        """L1 (optionally normalized) over all feature maps except the final logit map."""
        n_layers = len(fg) - 1
        if n_layers <= 0:
            return torch.zeros((), device=fg[-1].device)
        return sum(self.fm_reduction(fr[j], fg[j]) for j in range(n_layers)) / n_layers

    def loss(self, reals, fakes):
        """Unified adversarial + feature-matching loss (one forward of reals and fakes).

        Returns ``(dis_loss, adv_loss, feature_matching_distance)``, each a scalar averaged
        over the K sub-discriminators. ``d_*[i]`` is a list of feature maps whose last
        element ``[-1]`` is the logit map; ``[:-1]`` are intermediate features.
        """
        d_real = self.discriminator(reals)
        d_fake = self.discriminator(fakes)

        dis_loss = torch.zeros((), device=reals.device)
        adv_loss = torch.zeros((), device=reals.device)
        fm_loss = torch.zeros((), device=reals.device)

        for fr, fg in zip(d_real, d_fake):
            _dis, _adv = self._loss_fn(fr[-1], fg[-1])     # (scores_real, scores_fake)
            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv
            fm_loss = fm_loss + self._feature_matching(fr, fg)

        k = len(d_fake)
        return dis_loss / k, adv_loss / k, fm_loss / k

    def discriminator_loss(self, fake, real):
        """Discriminator-only update (fakes detached). Respects ``loss_type``."""
        d_fake = self.discriminator(fake.clone().detach())
        d_real = self.discriminator(real)
        loss_d = torch.zeros((), device=real.device)
        for fr, fg in zip(d_real, d_fake):
            _dis, _ = self._loss_fn(fr[-1], fg[-1])
            loss_d = loss_d + _dis
        return loss_d / len(d_fake)

    def generator_loss(self, fake, real):
        """Generator adversarial + feature-matching loss. Respects ``loss_type``."""
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        loss_g = torch.zeros((), device=real.device)
        loss_feature = torch.zeros((), device=real.device)
        for fr, fg in zip(d_real, d_fake):
            _, _adv = self._loss_fn(fr[-1], fg[-1])
            loss_g = loss_g + _adv
            loss_feature = loss_feature + self._feature_matching(fr, fg)
        k = len(d_fake)
        return loss_g / k, loss_feature / k
