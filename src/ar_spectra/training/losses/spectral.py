import warnings
from typing import Any, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal

from ar_spectra.utils.spectral import to_complex_spectrogram
from .signal import FIRFilter
import numpy as np
import scipy.signal

def get_k_weight_curve(n_fft: int, fs: float) -> torch.Tensor:
    """Returns the K-weight frequency magnitude response curve for STFT bins."""
    f_hp, Q_hp = 38.135, 0.5
    w_hp = 2 * np.pi * f_hp
    NUM_hp = [1, 0, 0]                  
    DEN_hp = [1, w_hp/Q_hp, w_hp**2]    

    f_shelf, Q_shelf, G_shelf = 1681.974, 1.69, 4.0
    k = 10**(G_shelf/20.0)
    w_s = 2 * np.pi * f_shelf
    NUM_shelf = [k, (k*w_s)/Q_shelf, w_s**2]
    DEN_shelf = [1,    w_s /Q_shelf, w_s**2]

    NUMs = np.polymul(NUM_hp, NUM_shelf)
    DENs = np.polymul(DEN_hp, DEN_shelf)

    b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

    freqs = np.linspace(0.0, fs/2.0, num=n_fft // 2 + 1, endpoint=True)
    _, H = scipy.signal.freqz(b, a, worN=freqs, fs=fs)
    Hmag = np.abs(H)

    # Normalize roughly to 1 at 1kHz
    _, H_ref = scipy.signal.freqz(b, a, worN=[1000.0], fs=fs)
    Hmag = Hmag / np.abs(H_ref[0])

    return torch.from_numpy(Hmag.astype(np.float32))

class ComplexMSE(nn.Module):
    def __init__(
        self,
        *,
        p: float = 2.0,
        eps: float = 1e-7,
        reduction: str = "mean",
        dim: Optional[Sequence[int]] = None,
        keepdim: bool = False,
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}.")
        self.p = p
        self.eps = eps
        self.reduction = reduction
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self,
        S_hat: torch.Tensor,
        S: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        S_hat = to_complex_spectrogram(S_hat)
        S = to_complex_spectrogram(S)
        if S_hat.shape != S.shape:
            raise ValueError(f"Shape mismatch: {S_hat.shape} vs {S.shape}.")
        error_mag = (S_hat - S).abs()
        loss_tensor = error_mag.pow(self.p)

        if weight is not None:
            loss_tensor = loss_tensor * weight

        if self.reduction == "none":
            return loss_tensor
        reduce_dims = tuple(range(loss_tensor.ndim)) if self.dim is None else tuple(self.dim)
        if self.reduction == "mean":
            return loss_tensor.mean(dim=reduce_dims, keepdim=self.keepdim)
        else:  # sum
            return loss_tensor.sum(dim=reduce_dims, keepdim=self.keepdim)

class PerceptualComplexMSE(ComplexMSE):
    """
    ComplexMSE that weights the frequency bins using the K-weighting curve.
    Exactly identical to filtering the waveform if `power_norm_alpha` matches the VAE's compression.
    """
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, power_norm_alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        curve = get_k_weight_curve(n_fft=n_fft, fs=sample_rate)
        curve = curve.pow(power_norm_alpha)
        # Reshape to (1, 1, F, 1) to broadcast over (B, C, F, T)
        self.register_buffer("freq_weight", curve.view(1, 1, -1, 1))

    def forward(self, S_hat: torch.Tensor, S: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.freq_weight
        if weight is not None:
            w = w * weight
        return super().forward(S_hat, S, weight=w)

class PhaseCosineDistance(nn.Module):
    def __init__(
        self,
        *,
        energy_weighted: bool = True,
        energy_ref: str = "avg",
        energy_power: float = 1.0,
        eps: float = 1e-7,
        reduction: str = "mean",
        dim: Optional[Sequence[int]] = None,
        keepdim: bool = False,
    ):
        super().__init__()
        if energy_ref not in {"avg", "ref"}:
            raise ValueError(f"energy_ref must be 'avg' or 'ref', but got {energy_ref}.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}.")

        self.energy_weighted = energy_weighted
        self.energy_ref = energy_ref
        self.energy_power = energy_power
        self.eps = eps
        self.reduction = reduction
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self,
        S_hat: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        S_hat = to_complex_spectrogram(S_hat)
        S = to_complex_spectrogram(S)
        if S_hat.shape != S.shape:
            raise ValueError(f"Input shapes must match, but got {S_hat.shape} and {S.shape}.")
        if not (torch.is_complex(S_hat) and torch.is_complex(S)):
            raise TypeError("Input tensors S_hat and S must be complex-valued.")

        angle_hat = torch.angle(S_hat)
        angle_ref = torch.angle(S)
        loss_tensor = 1.0 - torch.cos(angle_hat - angle_ref)

        if self.energy_weighted:
            abs_hat = S_hat.abs()
            abs_ref = S.abs()
            if self.energy_ref == "avg":
                E = 0.5 * (abs_hat + abs_ref)
            else:
                E = abs_ref
            w = E.pow(self.energy_power)
            loss_tensor = loss_tensor * w

        if self.reduction == "none":
            return loss_tensor
        reduce_dims = tuple(range(loss_tensor.ndim)) if self.dim is None else tuple(self.dim)
        if self.reduction == "mean":
            return loss_tensor.mean(dim=reduce_dims, keepdim=self.keepdim)
        else:
            return loss_tensor.sum(dim=reduce_dims, keepdim=self.keepdim)

class ComplexSpectralConvergence(nn.Module):
    def __init__(self, *, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', but got {reduction}.")
        self.reduction = reduction
        self.eps = eps

    def forward(self, S_hat: torch.Tensor, S_gt: torch.Tensor) -> torch.Tensor:
        S_hat = to_complex_spectrogram(S_hat)
        S_gt = to_complex_spectrogram(S_gt)
        if S_hat.shape != S_gt.shape:
            raise ValueError(f"Input shapes must match, but got {S_hat.shape} and {S_gt.shape}.")
        if not (torch.is_complex(S_hat) and torch.is_complex(S_gt)):
            raise TypeError("Input tensors S_hat and S_gt must be complex-valued.")

        batch, channels, frequency, time = S_hat.shape
        S_hat = S_hat.reshape(batch*channels, frequency, time)
        S_gt = S_gt.reshape(batch*channels, frequency, time)

        diff_flat = (S_gt - S_hat).reshape(batch*channels, -1)

        # MPS-safe: torch.linalg.norm on complex is unsupported on MPS.
        # .abs() first is mathematically identical: ||z||₂ = ||z.abs()||₂ = sqrt(Σ|zᵢ|²)
        num = torch.linalg.norm(diff_flat.abs(), ord=2, dim=1)
        gt_flat = S_gt.reshape(batch*channels, -1)
        den = torch.linalg.norm(gt_flat.abs(), ord=2, dim=1).clamp_min(self.eps)
        sc = num / den

        if self.reduction == "none":
            return sc
        elif self.reduction == "sum":
            return sc.sum()
        elif self.reduction == "mean":
            return sc.mean()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

class SpectralContrastLoss(nn.Module):
    """Spectral contrast loss: symmetric, scale-invariant spectral distance.
    
    Computes the ratio ||x - y||_F / ||x + y||_F over the input magnitudes,
    where the numerator is the Frobenius norm of the difference and the
    denominator is the Frobenius norm of the sum.
    
    Adapted from stable-audio-tools.
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag, y_mag):
        # Allow complex spectrograms (take magnitude first)
        if torch.is_complex(x_mag):
            x_mag = x_mag.abs()
        if torch.is_complex(y_mag):
            y_mag = y_mag.abs()
            
        x_mag = x_mag.float()
        y_mag = y_mag.float()
        numerator = torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2])
        denominator = torch.norm(x_mag + y_mag, p="fro", dim=[-1, -2]).clamp_min(self.eps)
        return (numerator / denominator).mean()

class MultiResSpectralConvergence(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Optional[Sequence[int]] = (512, 1024, 2048),
        eps: float = 1e-6,
        window = torch.hann_window,
        *,
        apply_pre_transform: bool = False,
        pre_transform: Optional[Any] = None,
    ):
        super().__init__()
        if len(fft_sizes) != len(hop_sizes):
            raise ValueError("fft_sizes and hop_sizes must have the same length.")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.eps = eps
        self.window = window
        self.win_lengths = win_lengths if win_lengths is not None else fft_sizes
        if apply_pre_transform and pre_transform is None:
            warnings.warn(
                "apply_pre_transform=True but no pre_transform provided; disabling transform for MultiResSpectralConvergence.",
                RuntimeWarning,
            )
            apply_pre_transform = False
        self._apply_pre_transform = apply_pre_transform
        self._pre_transform = pre_transform
        self.sc_loss = ComplexSpectralConvergence(reduction="mean", eps=eps)

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win_length: int,
              window: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        Z = torch.stft(
            x.float(), n_fft=n_fft, hop_length=hop, win_length=win_length,
            window=window.float(), center=True, return_complex=True,
            pad_mode="reflect"
        )
        F, TT = Z.shape[-2:]
        return Z.view(B, C, F, TT)
        
    def forward(
            self,
            wav_hat: torch.Tensor,
            wav_gt: torch.Tensor,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        if wav_hat.dim() == 2:
            wav_hat = wav_hat.unsqueeze(1)
            wav_gt  = wav_gt.unsqueeze(1)
        assert wav_hat.shape == wav_gt.shape, "waveform shape mismatch"

        sc_vals = []
        for n_fft, hop, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            if callable(self.window):
                window = self.window(win_length, device=wav_hat.device, dtype=wav_hat.dtype)
            else:
                window = self.window.to(device=wav_hat.device, dtype=wav_hat.dtype)

            S_hat = self._stft(wav_hat, n_fft, hop, win_length, window)
            S_gt  = self._stft(wav_gt , n_fft, hop, win_length, window)

            if self._apply_pre_transform:
                S_hat = self._pre_transform.transform(S_hat)
                S_gt = self._pre_transform.transform(S_gt)
            sc = self.sc_loss(S_hat, S_gt)
            sc_vals.append(sc)

        sc_vals = torch.stack(sc_vals, dim=0)
        if reduction == "mean":
            return sc_vals.mean()
        elif reduction == "sum":
            return sc_vals.sum()
        elif reduction == "none":
            return sc_vals

class STFTConsistencyLoss(nn.Module):
    """
    Self-supervised STFT consistency loss: ||STFT(ISTFT(S_hat)) - S_hat||.
    Penalizes spectrograms that lie off the overlap-add manifold (valid STFT images).
    Off-manifold spectrograms cause audible static noise after ISTFT reconstruction.
    Does not require a target — purely self-supervised on the decoder output.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        center: bool = True,
        normalized: bool = False,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalized = normalized
        self.reduction = reduction
        self.eps = eps
        self.register_buffer("_window", torch.hann_window(win_length))

    def forward(self, S_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_hat: (B, C, F, T) complex spectrogram from the decoder.
        Returns:
            Scalar consistency loss.
        """
        B, C, F, T = S_hat.shape
        window = self._window.to(device=S_hat.device, dtype=torch.float32)

        S_flat = S_hat.reshape(B * C, F, T)                          # (B*C, F, T)

        # ISTFT: project onto waveform domain — this is the Griffin-Lim projection step
        length = (T - 1) * self.hop_length                           # original waveform length
        wav = torch.istft(
            S_flat.to(torch.complex64),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            length=length,
        )                                                              # (B*C, length)

        # STFT back: project onto the consistent spectrogram manifold
        S_cons = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            return_complex=True,
            pad_mode="reflect",
        )                                                              # (B*C, F, T)

        S_cons = S_cons.view(B, C, F, T).to(S_hat.dtype)

        diff = (S_cons - S_hat).abs()                                 # (B, C, F, T) real
        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        return diff


class MultiResolutionSpectrogramLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048, 4096),
        hop_sizes: Optional[Sequence[int]] = None,
        win_lengths: Optional[Sequence[int]] = None,
        window_fn = torch.hann_window,
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
        *,
        linear_mag: bool = False,
        log_mag: bool = False,
        factor_linear_mag: float = 1.0,
        factor_log_mag: float = 1.0,
        eps: float = 1e-6,
        eps_mag: float = 1e-6,
        reduction: str = "mean",
        return_details: bool = False,
        apply_pre_transform: bool = False,
        pre_transform: Optional[Any] = None,
        max_loss_clamp: float = 0.0,
        perceptual_weighting: bool = False,
        sample_rate: Optional[float] = None,
    ):
        super().__init__()
        self.max_loss_clamp = max_loss_clamp   # 0.0 = disabled; bounds explosive spikes from near-zero bins
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes if hop_sizes is not None else [n // 4 for n in fft_sizes]
        self.win_lengths = win_lengths if win_lengths is not None else fft_sizes

        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")

        if linear_mag and log_mag:
            raise ValueError("linear_mag and log_mag are mutually exclusive. Choose only one.")

        self.window_fn = window_fn
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.linear_mag = linear_mag
        self.log_mag = log_mag
        self.factor_linear_mag = factor_linear_mag
        self.factor_log_mag = factor_log_mag
        self.eps = eps
        self.eps_mag = eps_mag
        self.reduction = reduction
        self.return_details = return_details

        self.sc_loss = ComplexSpectralConvergence(reduction='mean', eps=eps)

        # A-weighting IIR pre-filter (same as SAO perceptual_weighting)
        if perceptual_weighting:
            if sample_rate is None:
                raise ValueError("sample_rate is required when perceptual_weighting=True")
            self._fir_filter: Optional[nn.Module] = FIRFilter(filter_type="aw", fs=int(sample_rate))
        else:
            self._fir_filter = None

        if apply_pre_transform and pre_transform is None:
            warnings.warn(
                "apply_pre_transform=True but no pre_transform provided; disabling transform for MultiResolutionSpectrogramLoss.",
                RuntimeWarning,
            )
            apply_pre_transform = False
        self._apply_pre_transform = apply_pre_transform and pre_transform is not None
        self._pre_transform = pre_transform if self._apply_pre_transform else None

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win_len: int, window: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        # cuFFT does not support BFloat16 — cast to float32 for STFT, then restore dtype
        Z = torch.stft(
            x_flat.float(), n_fft=n_fft, hop_length=hop, win_length=win_len,
            window=window.float(), center=True, return_complex=True, pad_mode="reflect"
        )  # stays complex64 — bfloat16 has no complex type, casting would silently discard imaginary
        _, F, TT = Z.shape
        return Z.view(B, C, F, TT)

    def forward(
        self,
        wav_hat: torch.Tensor,
        wav_gt: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if wav_hat.ndim == 2:
            wav_hat = wav_hat.unsqueeze(1)
        if wav_gt.ndim == 2:
            wav_gt = wav_gt.unsqueeze(1)
        if wav_hat.shape != wav_gt.shape:
            raise ValueError(f"Shape mismatch: {wav_hat.shape} vs {wav_gt.shape}")

        # A-weighting pre-filter — mirrors SAO's perceptual_weighting path exactly
        if self._fir_filter is not None:
            self._fir_filter.to(wav_hat.device)
            B, C, T = wav_hat.shape
            wav_hat_f, wav_gt_f = self._fir_filter(
                wav_hat.reshape(B * C, 1, T),
                wav_gt.reshape(B * C, 1, T),
            )
            wav_hat = wav_hat_f.reshape(B, C, -1)
            wav_gt = wav_gt_f.reshape(B, C, -1)

        losses_per_res = []
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = self.window_fn(win_len, device=wav_hat.device, dtype=wav_hat.dtype)

            S_hat = self._stft(wav_hat, n_fft, hop, win_len, window)
            S_gt = self._stft(wav_gt, n_fft, hop, win_len, window)

            if self._apply_pre_transform:
                S_hat = self._pre_transform.transform(S_hat)
                S_gt = self._pre_transform.transform(S_gt)

            loss_sc = self.sc_loss(S_hat, S_gt)
            loss_complex = (S_hat - S_gt).abs().mean()

            add_mag = 0.0
            if self.linear_mag or self.log_mag:
                mag_hat = S_hat.abs().clamp_min(self.eps_mag)
                mag_gt = S_gt.abs().clamp_min(self.eps_mag)
                if self.linear_mag:
                    lin_mag_loss = (mag_hat - mag_gt).abs().mean()
                    add_mag = add_mag + self.factor_linear_mag * lin_mag_loss
                if self.log_mag:
                    log_mag_hat = torch.log(mag_hat + self.eps_mag)
                    log_mag_gt = torch.log(mag_gt + self.eps_mag)
                    log_mag_loss = (log_mag_hat - log_mag_gt).abs().mean()
                    add_mag = add_mag + self.factor_log_mag * log_mag_loss

            total_res_loss = (
                self.factor_sc * loss_sc +
                self.factor_mag * loss_complex +
                add_mag
            )
            losses_per_res.append(total_res_loss)

        losses_per_res = torch.stack(losses_per_res)

        if self.reduction == "mean":
            total_loss = losses_per_res.mean()
        elif self.reduction == "sum":
            total_loss = losses_per_res.sum()
        elif self.reduction == "none":
            total_loss = losses_per_res
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        if self.max_loss_clamp > 0.0:
            total_loss = total_loss.clamp(max=self.max_loss_clamp)

        if self.return_details:
            return total_loss, losses_per_res
        return total_loss

class InstantaneousFrequencyGroupDelayLoss(nn.Module):
    def __init__(
        self,
        eps=1e-3,
        w_floor=1e-3,
        is_complex=False,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        # Multi-resolution extensions to match MRSTFTSame:
        fft_sizes: Optional[Sequence[int]] = None,
        overlap: float = 0.75,
        sample_rate: int = 44100,
        k_weighting: bool = False,
        ms_lr: bool = False,
        complex_distance: bool = False,
        ncd_eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.w_floor = w_floor
        self.is_complex = is_complex
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.fft_sizes = tuple(int(n) for n in fft_sizes) if fft_sizes is not None else None
        if self.fft_sizes is not None:
            self.hop_sizes = tuple(max(1, int(round(n * (1.0 - overlap)))) for n in self.fft_sizes)
            self.sample_rate = sample_rate
            self.k_weighting = k_weighting
            self.ms_lr = ms_lr
            self.complex_distance = complex_distance

            # Sub-modules
            self.base_ifgd_loss = InstantaneousFrequencyGroupDelayLoss(
                eps=eps, w_floor=w_floor, is_complex=True)
            self.ncd_loss = NormalizedComplexDistanceLoss(eps=ncd_eps, is_complex=True) if complex_distance else None

            for i, n_fft_val in enumerate(self.fft_sizes):
                self.register_buffer(f"_win_{i}", torch.hann_window(n_fft_val), persistent=False)
                if k_weighting:
                    curve = get_k_weight_curve(n_fft=n_fft_val, fs=sample_rate)
                    self.register_buffer(f"_kw_{i}", curve.view(1, 1, -1, 1), persistent=False)
        else:
            if not is_complex:
                self.register_buffer("_window", torch.hann_window(win_length))

    def _to_channels(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if self.ms_lr and wav.shape[1] == 2:
            left, right = wav[:, 0:1], wav[:, 1:2]
            mid, side = 0.5 * (left + right), 0.5 * (left - right)
            wav = torch.cat([left, right, mid, side], dim=1)
        return wav

    def _stft(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, C, N = x.shape
        n_fft = self.fft_sizes[i]
        window = getattr(self, f"_win_{i}").to(device=x.device, dtype=torch.float32)
        Z = torch.stft(
            x.reshape(B * C, N).float(), n_fft=n_fft, hop_length=self.hop_sizes[i],
            win_length=n_fft, window=window, center=True, return_complex=True,
            pad_mode="reflect",
        )
        Z = Z.view(B, C, Z.shape[-2], Z.shape[-1])
        if self.k_weighting:
            Z = Z * getattr(self, f"_kw_{i}").to(Z.device)
        return Z

    def forward(self, x_hat, x_gt):
        if getattr(self, "fft_sizes", None) is not None:
            if x_hat.shape != x_gt.shape:
                raise ValueError(f"Shape mismatch: {x_hat.shape} vs {x_gt.shape}")
            x = self._to_channels(x_hat)
            y = self._to_channels(x_gt)

            losses_per_res = []
            for i in range(len(self.fft_sizes)):
                X = self._stft(x, i)
                Y = self._stft(y, i)
                l_ifgd = self.base_ifgd_loss(X, Y)
                if self.ncd_loss is not None:
                    l_ifgd = l_ifgd + self.ncd_loss(X, Y)
                losses_per_res.append(l_ifgd)
            return torch.stack(losses_per_res).mean()

        # Original single-resolution path:
        if not self.is_complex:
            if x_hat.ndim == 2:
                x_hat = x_hat.unsqueeze(1)
                x_gt = x_gt.unsqueeze(1)
            B, C, T = x_hat.shape
            x_hat = x_hat.reshape(B*C, T)
            x_gt = x_gt.reshape(B*C, T)
            window = self._window.to(x_hat.device, dtype=x_hat.dtype)
            Xp = torch.stft(x_hat.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window.float(), center=True, return_complex=True, pad_mode="reflect")
            Xr = torch.stft(x_gt.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window.float(), center=True, return_complex=True, pad_mode="reflect")
            Xp = Xp.view(B, C, Xp.shape[-2], Xp.shape[-1])
            Xr = Xr.view(B, C, Xr.shape[-2], Xr.shape[-1])
        else:
            Xp = to_complex_spectrogram(x_hat)
            Xr = to_complex_spectrogram(x_gt)

        # ---- time increments (IF) ----
        Rt_p = Xp[..., :, 1:] * torch.conj(Xp[..., :, :-1])
        Rt_r = Xr[..., :, 1:] * torch.conj(Xr[..., :, :-1])
        denom_t_p = (Xp[..., :, 1:].abs() * Xp[..., :, :-1].abs()).clamp_min(self.eps)
        denom_t_r = (Xr[..., :, 1:].abs() * Xr[..., :, :-1].abs()).clamp_min(self.eps)
        Ut_p = Rt_p / denom_t_p
        Ut_r = Rt_r / denom_t_r
        wt = torch.sqrt(denom_t_p * denom_t_r).clamp_min(self.w_floor).detach()
        wt = wt / wt.mean().clamp_min(1e-7)
        Lt = (1.0 - (Ut_p * torch.conj(Ut_r)).real) * wt

        # ---- frequency increments (GD) ----
        Rf_p = Xp[..., 1:, :] * torch.conj(Xp[..., :-1, :])
        Rf_r = Xr[..., 1:, :] * torch.conj(Xr[..., :-1, :])
        denom_f_p = (Xp[..., 1:, :].abs() * Xp[..., :-1, :].abs()).clamp_min(self.eps)
        denom_f_r = (Xr[..., 1:, :].abs() * Xr[..., :-1, :].abs()).clamp_min(self.eps)
        Uf_p = Rf_p / denom_f_p
        Uf_r = Rf_r / denom_f_r
        wf = torch.sqrt(denom_f_p * denom_f_r).clamp_min(self.w_floor).detach()
        wf = wf / wf.mean().clamp_min(1e-7)
        Lf = (1.0 - (Uf_p * torch.conj(Uf_r)).real) * wf

        return Lt.mean() + Lf.mean()

class NormalizedComplexDistanceLoss(nn.Module):
    def __init__(self, eps=1e-5, is_complex=False, n_fft=2048, hop_length=512, win_length=2048):
        super().__init__()
        self.eps = eps
        self.is_complex = is_complex
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if not is_complex:
            self.register_buffer("_window", torch.hann_window(win_length))

    def forward(self, x_hat, x_gt):
        if not self.is_complex:
            if x_hat.ndim == 2:
                x_hat = x_hat.unsqueeze(1)
                x_gt = x_gt.unsqueeze(1)
            B, C, T = x_hat.shape
            x_hat = x_hat.reshape(B*C, T)
            x_gt = x_gt.reshape(B*C, T)
            window = self._window.to(x_hat.device, dtype=x_hat.dtype)
            Xp = torch.stft(x_hat.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window.float(), center=True, return_complex=True, pad_mode="reflect")
            Xr = torch.stft(x_gt.float(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window.float(), center=True, return_complex=True, pad_mode="reflect")
            Xp = Xp.view(B, C, Xp.shape[-2], Xp.shape[-1])
            Xr = Xr.view(B, C, Xr.shape[-2], Xr.shape[-1])
        else:
            Xp = to_complex_spectrogram(x_hat)
            Xr = to_complex_spectrogram(x_gt)

        numerator = (Xp - Xr).abs() ** 2
        return torch.log(numerator / numerator.std(dim=[-1, -2], keepdim=True).detach().clamp(min=self.eps) + 1).mean()


def adaptive_log_mag(x_mag: torch.Tensor, y_mag: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """SAME §3.1 (eq.4) adaptive, σ-normalized log-magnitude L1 distance.

    Replaces the fixed-``eps`` log compression of ``MultiResolutionSpectrogramLoss``
    with a per-channel data-adaptive floor ``σ = √(std(X)² + std(Y)²)`` (detached), so
    the term is invariant to a common scaling of the pair. Translated from auraloss'
    ``STFTMagnitudeLoss`` (stable-audio-tools).

    Args:
        x_mag: ``(B, C, F, T)`` magnitude spectrogram (prediction).
        y_mag: ``(B, C, F, T)`` magnitude spectrogram (target).
        eps: floor on each per-channel std, guards silent channels.

    Returns:
        Scalar L1 loss between the σ-normalized log-magnitudes.
    """
    log_eps = torch.sqrt(                                                  # (B, C, 1, 1) detached
        x_mag.std(dim=(-1, -2), keepdim=True).detach().clamp(min=eps) ** 2 +
        y_mag.std(dim=(-1, -2), keepdim=True).detach().clamp(min=eps) ** 2
    )
    return F.l1_loss(torch.log(x_mag / log_eps + 1.0),                     # log(X/σ + 1)
                     torch.log(y_mag / log_eps + 1.0))


class MRSTFTSame(nn.Module):
    """SAME §3.1 multi-resolution STFT reconstruction loss.

    For each FFT resolution, over both the L/R and (optionally) mid/side channel
    representations of K-weighted complex spectrograms, sums three bounded /
    scale-invariant terms and averages across resolutions:

      - spectral contrast (eq.3, bounded)            → ``SpectralContrastLoss``
      - adaptive σ-normalized log-magnitude (eq.4)   → ``adaptive_log_mag``
      - phase-aware IFGD: ``L_IFGD = L_IF + L_GD + L_cd`` (eq.5-7), i.e. the
        instantaneous-freq / group-delay cosine terms (``InstantaneousFrequency
        GroupDelayLoss``) **plus** the normalized complex-distance penalty
        (``NormalizedComplexDistanceLoss``, eq.7) — all three as in SAME.

    All sub-terms are scale-invariant, so the chosen mid/side normalization is
    irrelevant. Reuses the existing primitives; the only new piece is
    ``adaptive_log_mag``. Operates in the waveform domain
    (``forward(wav_hat, wav_gt)``), mirroring ``MultiResolutionSpectrogramLoss``.
    SAME uses this as the *sole* reconstruction loss (no spectrogram MSE).
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (32, 64, 128, 256, 512, 1024, 2048),  # 7 SAME resolutions
        overlap: float = 0.75,                       # hop = n_fft * (1 - overlap)
        sample_rate: int = 44100,                    # for the K-weight curve
        k_weighting: bool = True,                    # perceptually weight the magnitudes
        w_sc: float = 1.0,                           # spectral-contrast term weight
        w_lm: float = 1.0,                           # adaptive log-mag term weight
        w_ifgd: float = 1.0,                         # phase-aware (IF+GD+cd) term weight
        ms_lr: bool = True,                          # process mid/side in addition to L/R
        complex_distance: bool = True,               # include L_cd (eq.7) inside L_IFGD
        sc_eps: float = 1e-4,                        # SpectralContrast denominator floor
        lm_eps: float = 1e-4,                        # adaptive_log_mag std floor
        ifgd_eps: float = 1e-3,                      # IFGD phasor-denominator floor
        ifgd_w_floor: float = 1e-3,                  # IFGD energy-weight floor
        ncd_eps: float = 1e-5,                        # L_cd (eq.7) self-norm std floor
    ):
        super().__init__()
        self.fft_sizes = tuple(int(n) for n in fft_sizes)
        self.hop_sizes = tuple(max(1, int(round(n * (1.0 - overlap)))) for n in self.fft_sizes)
        self.sample_rate = sample_rate
        self.k_weighting = k_weighting
        self.w_sc = w_sc
        self.w_lm = w_lm
        self.w_ifgd = w_ifgd
        self.ms_lr = ms_lr                           # append mid/side to L/R channels when stereo
        self.lm_eps = lm_eps

        self.sc_loss = SpectralContrastLoss(eps=sc_eps)
        # is_complex=True → operates directly on the spectrograms we compute here (no inner STFT)
        self.ifgd_loss = InstantaneousFrequencyGroupDelayLoss(
            eps=ifgd_eps, w_floor=ifgd_w_floor, is_complex=True)
        # L_cd (eq.7): normalized complex-distance penalty — third term of SAME's L_IFGD
        self.ncd_loss = NormalizedComplexDistanceLoss(eps=ncd_eps, is_complex=True) if complex_distance else None

        # One Hann window + one K-weight curve per resolution (registered buffers → move with module)
        for i, n_fft in enumerate(self.fft_sizes):
            self.register_buffer(f"_win_{i}", torch.hann_window(n_fft), persistent=False)
            if k_weighting:
                curve = get_k_weight_curve(n_fft=n_fft, fs=sample_rate)   # (n_fft//2+1,)
                self.register_buffer(f"_kw_{i}", curve.view(1, 1, -1, 1), persistent=False)

    def _to_channels(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, C, N)`` → ``(B, C', N)``: append mid/side to L/R when stereo and enabled."""
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)                                        # (B, 1, N)
        if self.ms_lr and wav.shape[1] == 2:
            left, right = wav[:, 0:1], wav[:, 1:2]                        # (B, 1, N) each
            mid, side = 0.5 * (left + right), 0.5 * (left - right)        # (B, 1, N) each
            wav = torch.cat([left, right, mid, side], dim=1)             # (B, 4, N)
        return wav

    def _stft(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """K-weighted complex STFT at resolution ``i`` → ``(B, C, F, T)`` complex."""
        B, C, N = x.shape
        n_fft = self.fft_sizes[i]
        window = getattr(self, f"_win_{i}").to(device=x.device, dtype=torch.float32)
        Z = torch.stft(                                                  # (B*C, F, T) complex64
            x.reshape(B * C, N).float(), n_fft=n_fft, hop_length=self.hop_sizes[i],
            win_length=n_fft, window=window, center=True, return_complex=True,
            pad_mode="reflect",
        )
        Z = Z.view(B, C, Z.shape[-2], Z.shape[-1])                       # (B, C, F, T)
        if self.k_weighting:
            Z = Z * getattr(self, f"_kw_{i}").to(Z.device)              # scale magnitude, keep phase
        return Z

    def forward(self, wav_hat: torch.Tensor, wav_gt: torch.Tensor) -> torch.Tensor:
        if wav_hat.shape != wav_gt.shape:
            raise ValueError(f"Shape mismatch: {wav_hat.shape} vs {wav_gt.shape}")
        x = self._to_channels(wav_hat)                                   # (B, C', N)
        y = self._to_channels(wav_gt)                                    # (B, C', N)

        losses_per_res = []
        for i in range(len(self.fft_sizes)):
            X = self._stft(x, i)                                         # (B, C', F, T) complex
            Y = self._stft(y, i)                                         # (B, C', F, T) complex
            l_sc = self.sc_loss(X, Y)                                    # bounded, scale-invariant
            l_lm = adaptive_log_mag(X.abs(), Y.abs(), eps=self.lm_eps)   # σ-normalized log-mag
            l_ifgd = self.ifgd_loss(X, Y)                                # L_IF + L_GD (eq.5-6)
            if self.ncd_loss is not None:
                l_ifgd = l_ifgd + self.ncd_loss(X, Y)                    # + L_cd (eq.7)
            losses_per_res.append(self.w_sc * l_sc + self.w_lm * l_lm + self.w_ifgd * l_ifgd)
        return torch.stack(losses_per_res).mean()

