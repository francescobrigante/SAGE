# =============================================================================
# Auraloss-derived frequency-domain loss functions (modified for C-VAE).
# =============================================================================
# Copied and modified from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py under Apache License 2.0
# You can find the license at LICENSES/LICENSE_AURALOSS.txt

import torch
import numpy as np
from typing import List, Any
import scipy.signal
import librosa.filters as librosa_filters

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
    except (AttributeError, TypeError):
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
    """
    Psychoacoustic prefilter (hp/fd/A-weight/K-weight) as a fixed FIR, stable for AMP/DDP.
    - Fixes K-shelf gain (→ k, not k^2).
    - Proper padding derived from kernel length.
    - Kernel registered as a buffer and cast to input dtype/device at runtime.
    """
    def __init__(self, filter_type="kw", coef=0.85, fs=44100, ntaps=257, pad_mode="reflect", ref_hz=1000.0, plot=False):
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.pad_mode = pad_mode
        self.ref_hz = ref_hz
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            taps = np.zeros(2, dtype=np.float64)  # length-2 pre-emphasis [1, -a]
            taps[0] = 1.0
            taps[1] = -self.coef
        elif filter_type == "fd":
            # simple 2-sample difference y[n] = x[n] - a x[n-2]
            taps = np.zeros(3, dtype=np.float64)
            taps[0] = 1.0
            taps[2] = -self.coef
        elif filter_type in {"aw", "kw"}:
            taps = self._design_weighting_fir(filter_type)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # normalise to unity gain at ref_hz
        if filter_type in {"aw", "kw"}:
            w = 2 * np.pi * self.ref_hz / self.fs
            n = np.arange(len(taps))
            H_ref = np.abs(np.sum(taps * np.exp(-1j * w * n)))
            if H_ref > 0:
                taps = taps / H_ref

        # register as buffer, not parameter
        k = torch.from_numpy(taps.astype(np.float32))[None, None, :]
        self.register_buffer("kernel", k, persistent=False)

    def _design_weighting_fir(self, which: str) -> np.ndarray:
        fs = self.fs
        ntaps = self.ntaps

        if which == "aw":
            f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
            A1000 = 1.9997  # dB
            NUMs = [(2*np.pi*f4)**2 * 10**(A1000/20), 0, 0, 0, 0]
            DENs = np.polymul([1, 4*np.pi*f4, (2*np.pi*f4)**2],
                              [1, 4*np.pi*f1, (2*np.pi*f1)**2])
            DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi*f3]),
                              [1, 2*np.pi*f2])
        elif which == "kw":
            # Stage 1: 2nd-order HP (critical damping)
            f_hp, Q_hp = 38.135, 0.5
            w_hp = 2*np.pi*f_hp
            NUM_hp = [1, 0, 0]                  # s^2
            DEN_hp = [1, w_hp/Q_hp, w_hp**2]    # s^2 + (w/Q)s + w^2

            # Stage 2: high-shelf (→ gain k at HF, 1 at LF)
            f_shelf, Q_shelf, G_shelf = 1681.974, 1.69, 4.0
            k = 10**(G_shelf/20.0)
            w_s = 2*np.pi*f_shelf
            NUM_shelf = [k, (k*w_s)/Q_shelf, w_s**2]
            DEN_shelf = [1,    w_s /Q_shelf, w_s**2]

            NUMs = np.polymul(NUM_hp, NUM_shelf)
            DENs = np.polymul(DEN_hp, DEN_shelf)
        else:
            raise RuntimeError

        # Bilinear to digital IIR
        b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

        # Endpoint-safe grid for firwin2
        freq = np.linspace(0.0, fs/2.0, num=8193, endpoint=True)  # Hz, exact 0 and fs/2
        _, H = scipy.signal.freqz(b, a, worN=freq, fs=fs)
        Hmag = np.abs(H)

        # FIR fit
        taps = scipy.signal.firwin2(ntaps, freq, Hmag, fs=fs)
        return taps

    def forward(self, input, target=None):
        B, C, T = input.shape
        x = input.reshape(B*C, 1, T)

        # ensure kernel is on the right device/dtype
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        pad = (k.shape[-1] - 1) // 2

        if self.pad_mode in {"reflect", "replicate", "constant"}:
            mode = self.pad_mode if self.pad_mode != "constant" else "constant"
            x = torch.nn.functional.pad(x, (pad, pad), mode=mode)
            y = torch.nn.functional.conv1d(x, k, padding=0)
        else:
            y = torch.nn.functional.conv1d(x, k, padding=pad)
            
        y = y.reshape(B, C, -1)
        
        if target is not None:
            B, C, T = target.shape
            t = target.reshape(B*C, 1, T)
            if self.pad_mode in {"reflect", "replicate", "constant"}:
                mode = self.pad_mode if self.pad_mode != "constant" else "constant"
                t = torch.nn.functional.pad(t, (pad, pad), mode=mode)
                y_t = torch.nn.functional.conv1d(t, k, padding=0)
            else:
                y_t = torch.nn.functional.conv1d(t, k, padding=pad)
            y_t = y_t.reshape(B, C, -1)
            return y, y_t
        return y

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
            if self.scale == "mel":
                fb = librosa_filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)
            elif self.scale == "chroma":
                fb = librosa_filters.chroma(sr=sample_rate, n_fft=fft_size, n_chroma=n_bins)
            self.register_buffer("fb", fb)

        if self.perceptual_weighting:
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)

    def stft(self, x):
        x_stft = torch.stft(x.float(), self.fft_size, self.hop_size, self.win_length, self.window.float(), return_complex=True)
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
