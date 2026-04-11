import warnings
from typing import Any, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from typing_extensions import Literal

from ar_spectra.utils.spectral import to_complex_spectrogram

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
    def __init__(self, *, reduction: str = "mean", eps : float = 1e-6):
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
        gt_flat = S_gt.reshape(batch*channels, -1)
        
        # MPS-safe: torch.linalg.norm on complex is unsupported on MPS.
        # .abs() first is mathematically identical: ||z||₂ = ||z.abs()||₂ = sqrt(Σ|zᵢ|²)
        num = torch.linalg.norm(diff_flat.abs(), ord=2, dim=1)
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
            x, n_fft=n_fft, hop_length=hop, win_length=win_length,
            window=window, center=True, return_complex=True,
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
        Z = torch.stft(
            x_flat, n_fft=n_fft, hop_length=hop, win_length=win_len,
            window=window, center=True, return_complex=True, pad_mode="reflect"
        )
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
