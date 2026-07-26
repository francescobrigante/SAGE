# =============================================================================
# WavTokenizer-full composite discriminator (Vocos-MPD + Vocos legacy mag-MRD + DAC)
# and its GAN loss wrapper WavTokenizerGANLoss.
# =============================================================================

import typing as tp
import torch
import torch.nn as nn
from einops import rearrange

from .types import (
    get_hinge_losses,
    get_relativistic_losses,
    get_sigmoid_relgan_losses,
)
from .dac import DACDiscriminator
from .vocos_legacy import MultiPeriodDiscriminator, MultiResolutionDiscriminator


class WavTokenizerDiscriminator(nn.Module):
    """
    Composite discriminator for WavTokenizer-full ablation:
      1. Vocos MPD (waveform period view)
      2. Vocos legacy MRD (magnitude STFT view, phase-blind)
      3. DACDiscriminator (MPD + complex multi-band MRD)
    
    Inputs can be mono (B, T) or stereo/multi-channel (B, C, T). For the Vocos
    sub-discriminators, channel dimensions are folded into the batch dimension
    prior to computation.
    """

    def __init__(
        self,
        channels: int = 1,
        sample_rate: int = 44100,
        periods: list = [2, 3, 5, 7, 11],
        resolutions: list = [(1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)],
        fft_sizes: list = [2048, 1024, 512],
        preprocess: bool = False,
        use_dac: bool = True,
        fold_lrms: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.sample_rate = sample_rate
        self.use_dac = use_dac
        self.fold_lrms = fold_lrms  # fold [M,S]=(L+R, L−R) into the batch → critics see L,R,M,S

        # 1. Vocos MPD (from vocos_legacy)
        self.vocos_mpd = MultiPeriodDiscriminator(periods=periods)

        # 2. Vocos legacy mag-MRD (from vocos_legacy)
        self.vocos_mrd = MultiResolutionDiscriminator(resolutions=resolutions)

        # 3. DACDiscriminator
        if self.use_dac:
            self.dac_disc = DACDiscriminator(
                channels=channels,
                periods=[],  # Set to empty to avoid duplicate MPD instances (covered by Vocos MPD)
                fft_sizes=fft_sizes,
                sample_rate=sample_rate,
                preprocess=preprocess,
            )

    def forward(self, x: torch.Tensor) -> tp.List[tp.List[torch.Tensor]]:
        if self.fold_lrms and x.ndim == 3 and x.shape[1] == 2:
            ms = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)  # (B, 2, T) = [M, S]
            x = torch.cat([x, ms], dim=0)                                     # (2B, 2, T) = [L,R | M,S]
        if x.ndim == 2:
            x_mono = x
            x_dac = x.unsqueeze(1)
        elif x.ndim == 3:
            if x.shape[1] == 1:
                x_mono = x.squeeze(1)
            else:
                x_mono = rearrange(x, "b c t -> (b c) t")
            x_dac = x
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.ndim}D tensor of shape {x.shape}")

        fmaps = []

        # Run Vocos MPD
        for d in self.vocos_mpd.discriminators:
            _, fmap = d(x_mono)
            fmaps.append(fmap)

        # Run Vocos legacy mag-MRD
        for d in self.vocos_mrd.discriminators:
            _, fmap = d(x_mono)
            fmaps.append(fmap)

        # Run DACDiscriminator
        if self.use_dac:
            dac_fmaps = self.dac_disc(x_dac)
            fmaps.extend(dac_fmaps)

        return fmaps


def _lsgan_losses(scores_real: torch.Tensor, scores_fake: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    dis_loss = (scores_fake ** 2).mean() + ((1 - scores_real) ** 2).mean()
    gen_loss = ((1 - scores_fake) ** 2).mean()
    return dis_loss, gen_loss


_LOSS_DISPATCH = {
    "hinge": get_hinge_losses,
    "rpgan": get_relativistic_losses,
    "sigmoid_relgan": get_sigmoid_relgan_losses,
    "lsgan": _lsgan_losses,
}


class WavTokenizerGANLoss(nn.Module):
    """
    GAN loss wrapper for WavTokenizerDiscriminator.
    Exposes the same interface as EncodecDiscriminator/DACGANLoss:
      .loss(reals, fakes) -> (dis_loss, adv_loss, feature_matching_distance)
    """

    def __init__(
        self,
        loss_type: tp.Literal["hinge", "rpgan", "sigmoid_relgan", "lsgan"] = "hinge",
        use_hinge: tp.Optional[bool] = None,
        normalize_losses: bool = False,
        **discriminator_kwargs,
    ):
        super().__init__()
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
        self.discriminator = WavTokenizerDiscriminator(**discriminator_kwargs)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> tp.Tuple[tp.List[tp.List[torch.Tensor]], tp.List[tp.List[torch.Tensor]]]:
        return self.discriminator(fake), self.discriminator(real)

    def _feature_matching(self, fr: tp.List[torch.Tensor], fg: tp.List[torch.Tensor]) -> torch.Tensor:
        n_layers = len(fg) - 1
        if n_layers <= 0:
            return torch.zeros((), device=fg[-1].device)
        return sum(self.fm_reduction(fr[j], fg[j]) for j in range(n_layers)) / n_layers

    def loss(self, reals: torch.Tensor, fakes: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_real = self.discriminator(reals)
        d_fake = self.discriminator(fakes)

        dis_loss = torch.zeros((), device=reals.device)
        adv_loss = torch.zeros((), device=reals.device)
        fm_loss = torch.zeros((), device=reals.device)

        for fr, fg in zip(d_real, d_fake):
            _dis, _adv = self._loss_fn(fr[-1], fg[-1])
            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv
            fm_loss = fm_loss + self._feature_matching(fr, fg)

        k = len(d_fake)
        return dis_loss / k, adv_loss / k, fm_loss / k

    def discriminator_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        d_fake = self.discriminator(fake.clone().detach())
        d_real = self.discriminator(real)
        loss_d = torch.zeros((), device=real.device)
        for fr, fg in zip(d_real, d_fake):
            _dis, _ = self._loss_fn(fr[-1], fg[-1])
            loss_d = loss_d + _dis
        return loss_d / len(d_fake)

    def generator_loss(self, fake: torch.Tensor, real: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
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
