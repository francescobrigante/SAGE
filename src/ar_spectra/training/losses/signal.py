# =============================================================================
# Auraloss-derived frequency-domain loss functions (modified for C-VAE).
# =============================================================================
# Copied and modified from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py under Apache License 2.0
# You can find the license at LICENSES/LICENSE_AURALOSS.txt

import torch
import numpy as np
from typing import List, Any
import scipy.signal

def normalized_complex_distance_loss(x, y, eps=1e-7):
    numerator = torch.nn.functional.l1_loss(x,y, reduction = 'none').abs()
    denominator =  0.5 * (x.abs() + y.abs()) + eps  # add epsilon for numerical stability
    return numerator / denominator

def apply_reduction(losses, reduction="none", retain_batch_dim=False):
    """Apply reduction to collection of losses."""
    dim = [-1, -2] if retain_batch_dim and len(losses.shape) == 3 else None
    if reduction == "mean":
        losses = losses.mean(dim = dim)
    elif reduction == "sum":
        losses = losses.sum(dim = dim)
    return losses

def get_window(win_type: str, win_length: int):
    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))
    return win

class SumAndDifference(torch.nn.Module):
    def __init__(self):
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")
        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)
        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class FIRFilter(torch.nn.Module):
    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997
            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
            DENs = np.polymul(np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

    def forward(self, input, target):
        input = torch.nn.functional.conv1d(input, self.fir.weight.data, padding=self.ntaps // 2)
        target = torch.nn.functional.conv1d(target, self.fir.weight.data, padding=self.ntaps // 2)
        return input, target

class SpectralConvergenceLoss(torch.nn.Module):
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()
    def forward(self, x_mag, y_mag):
        return (torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2]) / torch.norm(y_mag, p="fro", dim=[-1, -2])).unsqueeze(-1).unsqueeze(-1)

class STFTMagnitudeLoss(torch.nn.Module):
    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()
        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        retain_batch_dim: bool = False,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device
        self.retain_batch_dim = retain_batch_dim

        self.phs_used = bool(self.w_phs)
        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction if not self.retain_batch_dim else "none", distance=mag_distance, **kwargs)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction if not self.retain_batch_dim else "none", distance=mag_distance, **kwargs)

        if scale is not None:
            import librosa.filters
            if self.scale == "mel":
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)
            elif self.scale == "chroma":
                fb = librosa.filters.chroma(sr=sample_rate, n_fft=fft_size, n_chroma=n_bins)
            self.register_buffer("fb", fb)

        if self.perceptual_weighting:
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)

    def stft(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps))
        x_phs = x_stft if self.phs_used else None
        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()
        if self.perceptual_weighting:
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)
            self.prefilter.to(input.device)
            input, target = self.prefilter(input, target)
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        self.window = self.window.to(input.device)
        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = normalized_complex_distance_loss(x_phs,y_phs) if self.phs_used else 0.0

        loss = (self.w_sc * sc_mag_loss) + (self.w_log_mag * log_mag_loss) + (self.w_lin_mag * lin_mag_loss) + (self.w_phs * phs_loss)
        loss = apply_reduction(loss, reduction=self.reduction, retain_batch_dim=self.retain_batch_dim)

        if self.output == "loss": return loss
        return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: List[int] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.stft_losses = torch.nn.ModuleList()
        for i, (fs, ss, wl) in enumerate(zip(fft_sizes, hop_sizes, win_lengths)):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, scale, n_bins[i] if scale == "mel" and n_bins is not None else None, perceptual_weighting, scale_invariance, **kwargs)]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
        for f in self.stft_losses:
            if f.output == "full":
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1]); log_mag_loss.append(tmp_loss[2]); lin_mag_loss.append(tmp_loss[3]); phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)
        mrstft_loss /= len(self.stft_losses)
        if f.output == "loss": return mrstft_loss
        return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class SISDRLoss(torch.nn.Module):
    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target
        losses = 10 * torch.log10((target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses

class MelSTFTLoss(STFTLoss):
    def __init__(self, sample_rate, fft_size=1024, hop_size=256, win_length=1024, window="hann_window", w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0, w_phs=0.0, n_mels=128, **kwargs):
        super(MelSTFTLoss, self).__init__(fft_size, hop_size, win_length, window, w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, "mel", n_mels, **kwargs)
