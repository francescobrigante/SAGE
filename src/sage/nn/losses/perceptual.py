# =============================================================================
# L_mel of the paper (eq. 2): multi-resolution log-mel + mel-magnitude L1
# (DAC-style, via audiotools).
# =============================================================================
import typing as tp
import audiotools
import torch
from torch import nn

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate: int,
        n_mels: tp.List[int],
        window_lengths: tp.List[int],
        loss_fn: tp.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        mel_fmin: tp.Optional[tp.List[float]] = None,
        mel_fmax: tp.Optional[tp.List[float]] = None,
        window_type: tp.Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [{"window_length": w, "hop_length": w // 4, "window_type": window_type} for w in window_lengths]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.pow = pow
        self.mel_fmin = mel_fmin if mel_fmin is not None else [0.0 for _ in range(len(window_lengths))]
        self.mel_fmax = mel_fmax if mel_fmax is not None else [None for _ in range(len(window_lengths))]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = audiotools.AudioSignal(x, self.sample_rate)
        y = audiotools.AudioSignal(y, self.sample_rate)
        loss = 0.0
        for n_mels, fmin, fmax, params in zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params):
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)
            loss += self.log_weight * self.loss_fn(x_mels.clamp(self.clamp_eps).pow(self.pow).log10(), y_mels.clamp(self.clamp_eps).pow(self.pow).log10())
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss
