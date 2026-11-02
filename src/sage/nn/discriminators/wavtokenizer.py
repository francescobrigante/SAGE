# =============================================================================
# WavTokenizer-full composite discriminator (Vocos-MPD + Vocos legacy mag-MRD + DAC)
# and its GAN loss wrapper WavTokenizerGANLoss.
# =============================================================================

import typing as tp
import torch
import torch.nn as nn
from einops import rearrange

from sage.nn.discriminators.types import (
    BANDS,
    get_hinge_losses,
    get_relativistic_losses,
    get_sigmoid_relgan_losses,
)
from sage.nn.discriminators.dac import DACDiscriminator, peak_normalize
from sage.nn.discriminators.vocos_legacy import MultiPeriodDiscriminator, MultiResolutionDiscriminator


# Stereo layout of each branch: "fold" = one channel per batch row (the paper's critic, with
# fold_lrms adding the M, S rows); "lr" / "lrms" = the channels enter the first layer together.
STEREO_MODES = {
    "mpd_stereo": ("fold", "lr", "lrms"),
    "mrd_stereo": ("fold", "lrms"),       # |M|, |S| add information only after the magnitude
    "dac_stereo": ("fold", "lr", "lrms"),
}
JOINT_CHANNELS = {"lr": 2, "lrms": 4}
PREPROCESS_GAINS = ("per_channel", "shared")


class WavTokenizerDiscriminator(nn.Module):
    """
    Composite discriminator for WavTokenizer-full ablation:
      1. Vocos MPD (waveform period view)
      2. Vocos legacy MRD (magnitude STFT view, phase-blind)
      3. DACDiscriminator (complex multi-band MRD; its MPD is off, covered by 1.)

    Inputs can be mono (B, T) or stereo/multi-channel (B, C, T). By default every branch
    folds the channels into the batch (with fold_lrms, the rows are L, R, M, S): no critic
    sees L and R together. The *_stereo keys make a branch joint instead: [L, R] ("lr") or
    [L, R, M, S] ("lrms", M, S = (L ± R) / 2) enter its first layer together.

    Args:
        channels: audio channels of the input.
        sample_rate: sample rate of the input (Hz).
        periods: periods of the Vocos MPD.
        resolutions: (n_fft, hop, win) of the Vocos magnitude MRD.
        fft_sizes: window lengths of the DAC complex MRDs.
        preprocess: DAC preprocess (DC removal + peak normalisation) before the DAC branch.
        use_dac: build the DAC branch.
        fold_lrms: add the [M, S] = [L + R, L - R] rows to the branches that fold.
        preprocess_gain: "per_channel" (DAC original: erases ILD and Side level) or "shared"
            (one gain per stereo example, taken on L and R; M, S derived after it).
        mpd_stereo, mrd_stereo, dac_stereo: layout of each branch (see STEREO_MODES).
        dac_bands: band edges of the DAC MRDs as fractions of Nyquist (None = BANDS).
        inplace_act: LeakyReLU in place in every branch (exact: same outputs and gradients, about
            half the activation memory kept for the backward).
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
        preprocess_gain: str = "per_channel",
        mpd_stereo: str = "fold",
        mrd_stereo: str = "fold",
        dac_stereo: str = "fold",
        dac_bands: tp.Optional[list] = None,
        inplace_act: bool = False,
    ):
        super().__init__()
        modes = {"mpd_stereo": mpd_stereo, "mrd_stereo": mrd_stereo, "dac_stereo": dac_stereo}
        for key, mode in modes.items():
            if mode not in STEREO_MODES[key]:
                raise ValueError(f"{key}={mode!r}: choose from {STEREO_MODES[key]}")
        if preprocess_gain not in PREPROCESS_GAINS:
            raise ValueError(f"preprocess_gain={preprocess_gain!r}: choose from {PREPROCESS_GAINS}")
        if any(m != "fold" for m in modes.values()) and channels != 2:
            raise ValueError(f"joint stereo critics need channels=2, got {channels}")
        if use_dac and dac_stereo != "fold" and preprocess and preprocess_gain == "per_channel":
            raise ValueError("a joint DAC critic needs preprocess_gain='shared' (or preprocess=false): "
                             "a per-channel peak normalisation erases the ILD before the fusion")
        self.channels = channels
        self.sample_rate = sample_rate
        self.use_dac = use_dac
        self.fold_lrms = fold_lrms  # fold [M,S]=(L+R, L−R) into the batch → critics see L,R,M,S
        self.preprocess = preprocess                # DAC preprocess, applied here (not inside DACDiscriminator)
        self.preprocess_gain = preprocess_gain      # per_channel | shared
        self.mpd_stereo = mpd_stereo                # layout of the Vocos MPD
        self.mrd_stereo = mrd_stereo                # layout of the Vocos magnitude MRD
        self.dac_stereo = dac_stereo                # layout of the DAC complex MRDs

        # 1. Vocos MPD (from vocos_legacy)
        self.vocos_mpd = MultiPeriodDiscriminator(periods=periods, in_channels=JOINT_CHANNELS.get(mpd_stereo, 1),
                                                  inplace_act=inplace_act)

        # 2. Vocos legacy mag-MRD (from vocos_legacy)
        self.vocos_mrd = MultiResolutionDiscriminator(resolutions=resolutions,
                                                      in_channels=JOINT_CHANNELS.get(mrd_stereo, 1),
                                                      inplace_act=inplace_act)

        # 3. DACDiscriminator
        if self.use_dac:
            dac_joint = dac_stereo != "fold"
            self.dac_disc = DACDiscriminator(
                channels=JOINT_CHANNELS[dac_stereo] if dac_joint else channels,
                periods=[],  # Set to empty to avoid duplicate MPD instances (covered by Vocos MPD)
                fft_sizes=fft_sizes,
                sample_rate=sample_rate,
                bands=BANDS if dac_bands is None else list(dac_bands),
                preprocess=False,  # applied by dac_input, where the gain can be shared
                joint=dac_joint,
            )
            if inplace_act:                                   # the LeakyReLU of DAC's WNConv2d blocks
                for m in self.dac_disc.modules():
                    if isinstance(m, nn.LeakyReLU):
                        m.inplace = True
        # number of sub-discriminators per branch, in the order of forward's output
        self.group_sizes = {"mpd": len(periods), "mrd": len(resolutions), "dac": len(fft_sizes) if use_dac else 0}
        self.inplace_act = inplace_act

    def _fold(self, x: torch.Tensor) -> torch.Tensor:
        """With fold_lrms on a stereo batch: (B, 2, T) → (2B, 2, T) = [L,R | M,S], M,S = L ± R."""
        if self.fold_lrms and x.ndim == 3 and x.shape[1] == 2:
            ms = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)  # (B, 2, T) = [M, S]
            x = torch.cat([x, ms], dim=0)                                     # (2B, 2, T) = [L,R | M,S]
        return x

    @staticmethod
    def _rows(x: torch.Tensor) -> torch.Tensor:
        """Folded (B', C, T) or (B, T) → one mono row per channel (B'·C, T)."""
        if x.ndim == 2:
            return x
        return x.squeeze(1) if x.shape[1] == 1 else rearrange(x, "b c t -> (b c) t")

    @staticmethod
    def _joint(x: torch.Tensor, mode: str) -> torch.Tensor:
        """(B, 2, T) → [L, R] or [L, R, M, S] with M, S = (L ± R) / 2."""
        if mode == "lr":
            return x                                                          # (B, 2, T)
        left, right = x[:, 0], x[:, 1]
        return torch.stack([left, right, (left + right) / 2, (left - right) / 2], dim=1)   # (B, 4, T)

    @staticmethod
    def _as_channels(x: torch.Tensor) -> torch.Tensor:
        """Mono (B, T) → (B, 1, T); (B, C, T) unchanged."""
        return x.unsqueeze(1) if x.ndim == 2 else x

    def dac_input(self, x: torch.Tensor, folded: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """What the DAC MRDs see: folded rows or joint channels, after the preprocess.

        folded: _as_channels(_fold(x)) when forward has already built it. Reusing it keeps the
        autograd graph of the default critic, and so the rounding of the generator gradient, as on main.
        """
        if self.dac_stereo != "fold":
            x = peak_normalize(x, shared=True) if self.preprocess else x      # (B, 2, T)
            return self._joint(x, self.dac_stereo)                            # (B, 2|4, T)
        if self.preprocess and self.preprocess_gain == "shared":
            x = peak_normalize(self._as_channels(x), shared=True)             # one gain per example, on L, R
            folded = None                                                     # M, S are derived after the gain
        if folded is None:
            folded = self._as_channels(self._fold(x))                         # (2B, 2, T) or (B, C, T)
        if self.preprocess and self.preprocess_gain == "per_channel":
            folded = peak_normalize(folded)                                   # per (row, channel), as DAC
        return folded

    def forward(self, x: torch.Tensor) -> tp.List[tp.List[torch.Tensor]]:
        if x.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.ndim}D tensor of shape {x.shape}")
        # Fold once, and build the DAC input before the Vocos branches, in main's order: the default
        # critic then sums the gradients reaching x in the same order, bit for bit.
        x_fold = self._fold(x)                                                # (2B, 2, T) or (B, C) / (B, C, T)
        x_dac = self._as_channels(x_fold)                                     # (2B, 2, T) or (B, C, T)
        x_rows = self._rows(x_fold)                                           # (rows, T), for the folded branches
        fmaps = []

        # Run Vocos MPD
        x_mpd = x_rows if self.mpd_stereo == "fold" else self._joint(x, self.mpd_stereo)   # (rows, T) | (B, 2|4, T)
        for d in self.vocos_mpd.discriminators:
            _, fmap = d(x_mpd)
            fmaps.append(fmap)

        # Run Vocos legacy mag-MRD
        x_mrd = x_rows if self.mrd_stereo == "fold" else self._joint(x, self.mrd_stereo)   # (rows, T) | (B, 4, T)
        for d in self.vocos_mrd.discriminators:
            _, fmap = d(x_mrd)
            fmaps.append(fmap)

        # Run DACDiscriminator
        if self.use_dac:
            dac_fmaps = self.dac_disc(self.dac_input(x, folded=x_dac))
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

    def dis_loss(self, reals: torch.Tensor, fakes: torch.Tensor) -> torch.Tensor:
        """The first output of loss() alone, for the discriminator step: the same forwards (real, then
        fake) and the same loss, without the adversarial and feature-matching terms of the generator,
        which the discriminator step drops (they would keep every feature-map difference in memory)."""
        d_real = self.discriminator(reals)
        d_fake = self.discriminator(fakes)
        dis_loss = torch.zeros((), device=reals.device)
        for fr, fg in zip(d_real, d_fake):
            dis_loss = dis_loss + self._loss_fn(fr[-1], fg[-1])[0]
        return dis_loss / len(d_fake)

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
