import warnings
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchaudio
from typing_extensions import Literal

def to_complex_spectrogram(X: torch.Tensor) -> torch.Tensor:
    """
    Convert a spectrogram tensor to complex dtype handling multiple layouts:
      - complex tensor (..., C, F, T) -> returned as-is
      - real tensor with RI on last dim (..., C, F, T, 2) -> view_as_complex
      - complex-as-channels (cac=True) from dataset:
            (B, 2C, F, T) or (2C, F, T) with order [c0_r, c0_i, c1_r, c1_i, ...]
        -> returns (B, C, F, T) or (C, F, T) complex
    """
    if torch.is_complex(X):
        return X
    if not X.is_floating_point():
        raise TypeError("Expected floating or complex tensor for spectrogram input.")

    # Case: real/imag in last dimension
    if X.ndim >= 1 and X.size(-1) == 2:
        return torch.view_as_complex(X.contiguous())

    # Case: complex-as-channels, shape (B?, 2C, F, T)
    if X.ndim == 4:
        B, C2, F, T = X.shape
        if C2 % 2 != 0:
            raise ValueError(f"Channel dimension must be even for complex-as-channels. Got {C2}.")
        C = C2 // 2
        Xv = X.reshape(B, C, 2, F, T)
        real = Xv[:, :, 0, :, :]
        imag = Xv[:, :, 1, :, :]
        return torch.complex(real, imag)
    elif X.ndim == 3:
        C2, F, T = X.shape
        if C2 % 2 != 0:
            raise ValueError(f"Channel dimension must be even for complex-as-channels. Got {C2}.")
        C = C2 // 2
        Xv = X.reshape(C, 2, F, T) 
        real = Xv[:, 0, :, :]
        imag = Xv[:, 1, :, :]
        return torch.complex(real, imag)
    raise ValueError("Unsupported spectrogram shape. Expected (..., C, F, T), (..., C, F, T, 2) or (B, 2C, F, T)/(2C, F, T).")



class ComplexMSE(nn.Module):
    """
    Compute a generalized L^p magnitude error on complex spectrograms without
    normalization or perceptual weighting.

    The loss is defined per element as |S_hat - S|^p and then reduced according
    to the selected reduction strategy.

    Parameters:
        p (float): Exponent applied to the absolute complex difference.
            p = 2.0 yields a mean squared magnitude error (MSE);
            p = 1.0 yields a mean absolute magnitude error (MAE).
        eps (float): Currently unused; retained for forward compatibility and
            interface consistency.
        reduction (str): Reduction mode: one of {'none', 'mean', 'sum'}.
        dim (Optional[Sequence[int]]): Dimensions over which to apply the
            reduction. If None, all dimensions are reduced.
        keepdim (bool): If True, retains reduced dimensions with length 1.

    Forward Parameters:
        S_hat (torch.Tensor): Predicted complex spectrogram or a real tensor
            encodable as complex (RI or complex-as-channels).
        S (torch.Tensor): Reference complex spectrogram with the same shape
            or layout as S_hat.
        weight (Optional[torch.Tensor]): Optional multiplicative weight
            broadcastable to the loss tensor shape.

    Returns:
        torch.Tensor: The reduced loss value if reduction != 'none',
        otherwise the per-element loss tensor.

    Raises:
        ValueError: If input shapes differ.
        ValueError: If an invalid reduction is specified.
    """
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
    r"""
    Computes a phase distance metric based on the cosine of the phase difference,
    designed for complex-valued spectrograms.

    This function measures the dissimilarity between the phases of two complex tensors,
    `S_hat` (prediction) and `S` (target). The core distance metric is `1 - cos(delta_angle)`,
    which naturally handles the periodic nature of angles and is bounded between [0, 2].
    A key feature is the energy-based weighting, which ensures that phase errors in
    perceptually insignificant (low-energy) bins are penalized less severely than those
    in high-energy bins.

    The per-bin loss `L` is defined as:
    $$
    L = w \cdot (1 - \cos(\phi_{\hat{S}} - \phi_S))
    $$
    where:
    - $\phi_{\hat{S}}$ and $\phi_S$ are the angles of `S_hat` and `S`, respectively.
    - `w` is the perceptual weight, typically derived from the magnitude of the spectrograms.
    If weighting is enabled, $w = E^{\text{energy\_power}}$, where `E` is the reference energy.

    Args:
        energy_weighted (bool, optional): If True, applies a perceptual weight based on the
            magnitude of the spectrogram bins. This is highly recommended for phase losses.
            Defaults to True.
        energy_ref (str, optional): The reference for calculating the energy weight `w`.
            Must be one of {'avg', 'ref'}.
            - 'avg': Uses the arithmetic mean of the magnitudes, `E = 0.5 * (|\hat{S}| + |S|)`.
            - 'ref': Uses the magnitude of the reference signal, `E = |S|`.
            Defaults to "avg".
        energy_power (float, optional): The exponent applied to the reference energy `E` to
            create the weight `w`. `1.0` provides linear weighting by energy. Defaults to 1.0.
        eps (float, optional): A small constant for numerical stability, primarily used if
            the reference energy calculation involves division (not the case here, but
            retained for consistency). Defaults to 1e-7.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
        dim (Optional[Sequence[int]], optional): The dimensions over which to reduce the
            loss. If None, reduces over all dimensions. Defaults to None.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not.
            Defaults to False.
    """
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
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', but got {reduction}.")

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
        # Convert to complex if inputs are RI or CAC from dataset
        S_hat = to_complex_spectrogram(S_hat)
        S = to_complex_spectrogram(S)
        # --- Input Validation ---
        if S_hat.shape != S.shape:
            raise ValueError(f"Input shapes must match, but got {S_hat.shape} and {S.shape}.")
        if not (torch.is_complex(S_hat) and torch.is_complex(S)):
            raise TypeError("Input tensors S_hat and S must be complex-valued.")

        # --- Core Phase Distance Calculation ---
        angle_hat = torch.angle(S_hat)
        angle_ref = torch.angle(S)

        # The cosine of the difference correctly handles angle wrapping
        loss_tensor = 1.0 - torch.cos(angle_hat - angle_ref)

        # --- Perceptual Energy Weighting ---
        if self.energy_weighted:
            abs_hat = S_hat.abs()
            abs_ref = S.abs()

            # Reference energy for weighting
            if self.energy_ref == "avg":
                E = 0.5 * (abs_hat + abs_ref)
            else:  # "ref"
                E = abs_ref

            # The weight is the energy raised to a power
            w = E.pow(self.energy_power)
            loss_tensor = loss_tensor * w

        # --- Final Reduction ---
        if self.reduction == "none":
            return loss_tensor

        reduce_dims = tuple(range(loss_tensor.ndim)) if self.dim is None else tuple(self.dim)
        if self.reduction == "mean":
            return loss_tensor.mean(dim=reduce_dims, keepdim=self.keepdim)
        else:  # "sum"
            return loss_tensor.sum(dim=reduce_dims, keepdim=self.keepdim)
        

class ComplexSpectralConvergence(nn.Module):
    
    def __init__(self, *, reduction: str = "mean", eps : float = 1e-7):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', but got {reduction}.")
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, S_hat: torch.Tensor, S_gt: torch.Tensor) -> torch.Tensor:
        # Convert to complex if inputs are RI or CAC from dataset
        S_hat = to_complex_spectrogram(S_hat)
        S_gt = to_complex_spectrogram(S_gt)

        # --- Input Validation ---
        if S_hat.shape != S_gt.shape:
            raise ValueError(f"Input shapes must match, but got {S_hat.shape} and {S_gt.shape}.")
        if not (torch.is_complex(S_hat) and torch.is_complex(S_gt)):
            raise TypeError("Input tensors S_hat and S_gt must be complex-valued.")
        
        # (Batch, channels, frequency, time) -> (batch*channels, frequency, time)
        batch, channels, frequency, time = S_hat.shape
        S_hat = S_hat.reshape(batch*channels, frequency, time)
        S_gt = S_gt.reshape(batch*channels, frequency, time)
        
        diff_flat = (S_gt - S_hat).reshape(batch*channels, -1)
        gt_flat = S_gt.reshape(batch*channels, -1)
        
        num = torch.linalg.norm(diff_flat, ord=2, dim=1)
        den = torch.linalg.norm(gt_flat, ord=2, dim=1).clamp_min(self.eps)
        
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
    """
    Implements a multi-resolution spectral convergence loss for
    complex spectrograms.

    This class calculates the spectral convergence loss across
    multiple FFT resolutions using specified window functions and
    computations for multi-resolution analysis. It is designed
    to operate on complex spectrograms derived from the input
    waveforms, comparing predicted and ground truth waveforms.
    The purpose is to provide a robust loss function for tasks
    such as speech synthesis or enhancement.

    :ivar n_ffts: Tuple of FFT sizes used for multi-resolution analysis.
    :type n_ffts: Sequence[int]
    :ivar hops: Tuple of hop sizes corresponding to each FFT size.
    :type hops: Sequence[int]
    :ivar eps: Small constant for numerical stability.
    :type eps: float
    """
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Optional[Sequence[int]] = (512, 1024, 2048),
        eps: float = 1e-7,
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
        x = x.reshape(B * C, T)                      # merge canali
        Z = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win_length,
            window=window, center=True, return_complex=True,
            pad_mode="reflect"
        )                                            # (B*C, F, T')
        F, TT = Z.shape[-2:]
        return Z.view(B, C, F, TT)                   # (B, C, F, T')
        
    def forward(
            self,
            wav_hat: torch.Tensor,
            wav_gt: torch.Tensor,
            reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        # shape -> (B, C, T)
        if wav_hat.dim() == 2:
            wav_hat = wav_hat.unsqueeze(1)
            wav_gt  = wav_gt.unsqueeze(1)
        assert wav_hat.shape == wav_gt.shape, "waveform shape mismatch"
        B, C, _ = wav_gt.shape

        sc_vals = []
        for n_fft, hop, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # build window on the right device/dtype each time
            if callable(self.window):
                window = self.window(win_length, device=wav_hat.device, dtype=wav_hat.dtype)
            else:
                window = self.window.to(device=wav_hat.device, dtype=wav_hat.dtype)

            S_hat = self._stft(wav_hat, n_fft, hop, win_length, window)   # (B,C,F,T')
            S_gt  = self._stft(wav_gt , n_fft, hop, win_length, window)

            if self._apply_pre_transform:
                S_hat = self._pre_transform.transform(S_hat)
                S_gt = self._pre_transform.transform(S_gt)
            sc = self.sc_loss(S_hat, S_gt)  # scalar per-batch (mean)
            sc_vals.append(sc)

        sc_vals = torch.stack(sc_vals, dim=0)  # (R,)
        if reduction == "mean":
            return sc_vals.mean()
        elif reduction == "sum":
            return sc_vals.sum()
        elif reduction == "none":
            return sc_vals

class MultiResolutionSpectrogramLoss(nn.Module):
    """
    Multi-Resolution Spectrogram Loss (MR-Spec).

    This loss aggregates time–frequency discrepancies between predicted and reference
    waveforms across multiple STFT resolutions. For each resolution r (n_fft_r, hop_r, win_r)
    the following scalar components are computed:

        1. Spectral Convergence (SC):
            SC_r = || S_r - Ŝ_r ||_F / ( || S_r ||_F + eps_sc )

        2. Complex L1 (time–frequency reconstruction term):
            L_complex_r = mean_{b,c,f,t} | Ŝ_r - S_r |

        3. Optional Linear Magnitude L1 (phase-discarded):
            L_linmag_r = mean_{b,c,f,t} | |Ŝ_r| - |S_r| |

        4. Optional Log-Magnitude L1 (phase-discarded, improves perceptual balance):
            L_logmag_r = mean_{b,c,f,t} | log(|Ŝ_r| + eps_mag) - log(|S_r| + eps_mag) |

    Total per-resolution loss:
        L_r = w_sc * SC_r
              + w_complex * L_complex_r
              + I_lin * w_lin * L_linmag_r
              + I_log * w_log * L_logmag_r

        where I_lin = 1 if linear_mag=True else 0,
              I_log = 1 if log_mag=True else 0,
              and not (linear_mag and log_mag) (mutually exclusive by design).

    Final aggregation over all resolutions R:
        L_MR =
            mean_r L_r   if reduction == 'mean'
            sum_r  L_r   if reduction == 'sum'
            [L_r]_r      if reduction == 'none'

    Args:
        fft_sizes (Sequence[int]): STFT FFT sizes per resolution.
        hop_sizes (Optional[Sequence[int]]): Hop sizes; defaults to n_fft // 4.
        win_lengths (Optional[Sequence[int]]): Window lengths; defaults to fft_sizes.
        window_fn (Callable): Window function constructor (e.g. torch.hann_window).
        factor_sc (float): Weight w_sc for spectral convergence.
        factor_mag (float): Weight w_complex for complex L1 term (|Ŝ - S|).
        linear_mag (bool): Enable linear magnitude L1 term (| |Ŝ| - |S| |).
        log_mag (bool): Enable log-magnitude L1 term (| log(|Ŝ|) - log(|S|) |).
                        Mutually exclusive with linear_mag.
        factor_linear_mag (float): Weight w_lin applied if linear_mag=True.
        factor_log_mag (float): Weight w_log applied if log_mag=True.
        eps (float): Numerical stability for spectral convergence denominator.
        eps_mag (float): Numerical stability for magnitude + logarithm.
        reduction (str): {'mean','sum','none'} aggregation over resolutions.
        return_details (bool): If True returns (total_loss, per_resolution_losses).

    Forward Args:
        wav_hat (Tensor): Predicted waveform (B, C, T) or (B, T).
        wav_gt (Tensor): Reference waveform (same shape as wav_hat).

    Returns:
        Tensor if return_details=False else (total_loss, per_resolution_losses).

    Notes:
        - Phase is not directly penalized except via SC and complex L1.
        - Set exactly one of linear_mag or log_mag to True to add a pure magnitude term.
        - If both linear_mag and log_mag are False, the loss reduces to SC + complex L1.
    """
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
        eps: float = 1e-8,
        eps_mag: float = 1e-8,
        reduction: str = "mean",
        return_details: bool = False,
        apply_pre_transform: bool = False,
        pre_transform: Optional[Any] = None,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes if hop_sizes is not None else [n // 4 for n in fft_sizes]
        self.win_lengths = win_lengths if win_lengths is not None else fft_sizes

        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")

        if linear_mag and log_mag:
            raise ValueError("linear_mag and log_mag are mutually exclusive. Choose only one.")
        if factor_sc < 0 or factor_mag < 0 or factor_linear_mag < 0 or factor_log_mag < 0:
            raise ValueError("All factor weights must be non-negative.")

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

            # Spectral convergence
            loss_sc = self.sc_loss(S_hat, S_gt)

            # Complex L1 (difference in the complex plane)
            loss_complex = (S_hat - S_gt).abs().mean()

            # Optional magnitude-only terms
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

        if self.return_details:
            return total_loss, losses_per_res
        return total_loss



class MRMelLoss(nn.Module):
    """
    Computes the Multi-Resolution Mel-Spectrogram Loss (MR-Mel).

    This loss calculates the L1 distance between the Mel-scaled spectrograms of the predicted
    and ground-truth waveforms across multiple resolutions. It is designed to capture perceptual
    differences by projecting the magnitude spectrogram onto the Mel frequency scale, which
    approximates human hearing sensitivity.

    The total loss L_Mel is defined as the aggregation (mean or sum) over all resolutions r:

    L_Mel = Reduce_r( || log(M^{(r)}(\hat{y}) + \epsilon) - log(M^{(r)}(y) + \epsilon) ||_1 )

    Where:
      - M^{(r)} is the Mel-spectrogram projection at resolution r.
      - \epsilon is a small constant for numerical stability in the logarithmic domain.

    Args:
        sample_rate (int): The sampling rate of the input audio.
        fft_sizes (Sequence[int]): A sequence of FFT sizes for the multi-resolution analysis.
            Default is (512, 1024, 2048).
        hop_sizes (Optional[Sequence[int]]): A sequence of hop sizes corresponding to each FFT size.
            If None, defaults to `fft_size // 4`.
        win_lengths (Optional[Sequence[int]]): A sequence of window lengths corresponding to each FFT size.
            If None, defaults to `fft_sizes`.
        n_mels (Sequence[int]): Number of Mel bands for each resolution. Must match the length of fft_sizes.
            Default is (80, 80, 80).
        window_fn (Callable): The window function to apply (e.g., `torch.hann_window`).
            Default is `torch.hann_window`.
        f_min (float): Minimum frequency for the Mel filterbank. Default is 0.0.
        f_max (Optional[float]): Maximum frequency for the Mel filterbank. If None, uses sample_rate // 2.
        power (float): Exponent for the magnitude spectrogram before Mel projection. Default is 1.0 (Magnitude).
        log_mel (bool): If True, applies a logarithmic transformation to the Mel spectrograms. Default is True.
        mel_scale (str): Scale to use: 'htk' or 'slaney'. Default is 'slaney'.
        norm (Optional[str]): Normalization for the Mel filterbank. 'slaney' or None. Default is 'slaney'.
        eps_mag (float): Epsilon added to magnitude before power. Default is 1e-8.
        eps_log (float): Epsilon added before log. Default is 1e-5.
        reduction (str): Specifies the reduction to apply to the output over resolutions:
            'mean' | 'sum' | 'none'. Default is 'mean'.
        return_details (bool): If True, the forward pass returns a tuple containing the aggregated
            total loss and a tensor of losses per resolution. If False, returns only the total loss.
            Default is False.
    """
    def __init__(
        self,
        sample_rate: int,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Optional[Sequence[int]] = None,
        win_lengths: Optional[Sequence[int]] = None,
        n_mels: Sequence[int] = (80, 80, 80),
        window_fn = torch.hann_window,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 1.0,
        log_mel: bool = True,
        mel_scale: str = "slaney",
        norm: Optional[str] = "slaney",
        eps_mag: float = 1e-8,
        eps_log: float = 1e-5,
        reduction: str = "mean",
        return_details: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes if hop_sizes is not None else [n // 4 for n in fft_sizes]
        self.win_lengths = win_lengths if win_lengths is not None else fft_sizes
        self.n_mels = n_mels
        
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths) == len(self.n_mels)):
            raise ValueError("fft_sizes, hop_sizes, win_lengths, and n_mels must have the same length.")

        self.window_fn = window_fn
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.log_mel = log_mel
        self.eps_mag = eps_mag
        self.eps_log = eps_log
        self.reduction = reduction
        self.return_details = return_details

        # Pre-compute Mel Filterbanks and register as buffers
        for i, (n_fft, n_mel) in enumerate(zip(self.fft_sizes, self.n_mels)):
            n_freqs = n_fft // 2 + 1
            
            # Handle torchaudio version differences or specific functional calls
            try:
                fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_freqs,
                    f_min=self.f_min,
                    f_max=self.f_max if self.f_max is not None else float(self.sample_rate // 2),
                    n_mels=n_mel,
                    sample_rate=self.sample_rate,
                    norm=norm,
                    mel_scale=mel_scale,
                )
            except AttributeError:
                 # Fallback for older torchaudio versions if necessary, or use create_fb_matrix
                 fb = torchaudio.functional.create_fb_matrix(
                    n_freqs=n_freqs,
                    f_min=self.f_min,
                    f_max=self.f_max if self.f_max is not None else float(self.sample_rate // 2),
                    n_mels=n_mel,
                    sample_rate=self.sample_rate,
                    norm=norm,
                    mel_scale=mel_scale,
                )

            # Ensure shape is (n_mels, n_freqs) for matmul
            if fb.shape[0] != n_mel:
                fb = fb.transpose(0, 1)
                
            self.register_buffer(f"mel_basis_{i}", fb)

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win_len: int, window: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_reshaped = x.reshape(B * C, T)
        Z = torch.stft(
            x_reshaped, n_fft=n_fft, hop_length=hop, win_length=win_len,
            window=window, center=True, return_complex=True,
            pad_mode="reflect"
        )
        # Z: (B*C, F, T_frames)
        _, F, T_frames = Z.shape
        return Z.view(B, C, F, T_frames)

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

        for i, (n_fft, hop, win_len) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            # Construct window on the fly to ensure correct device/dtype
            window = self.window_fn(win_len, device=wav_hat.device, dtype=wav_hat.dtype)
            # Retrieve Mel basis
            mel_basis = getattr(self, f"mel_basis_{i}")

            # STFT
            S_hat = self._stft(wav_hat, n_fft, hop, win_len, window)
            S_gt = self._stft(wav_gt, n_fft, hop, win_len, window)

            # Magnitude & Power
            mag_hat = S_hat.abs().clamp_min(self.eps_mag).pow(self.power)
            mag_gt = S_gt.abs().clamp_min(self.eps_mag).pow(self.power)

            # Mel Projection: (B, C, F, T) -> (B, C, M, T)
            # We need to handle the channel dimension for matmul
            B, C, F, T = mag_hat.shape
            mag_hat_flat = mag_hat.view(B * C, F, T)
            mag_gt_flat = mag_gt.view(B * C, F, T)

            mel_hat = torch.matmul(mel_basis, mag_hat_flat).view(B, C, -1, T)
            mel_gt = torch.matmul(mel_basis, mag_gt_flat).view(B, C, -1, T)

            # Log (Optional)
            if self.log_mel:
                mel_hat = torch.log(mel_hat + self.eps_log)
                mel_gt = torch.log(mel_gt + self.eps_log)

            # L1 Loss
            loss = F.l1_loss(mel_hat, mel_gt, reduction="mean")
            losses_per_res.append(loss)

        losses_per_res = torch.stack(losses_per_res)

        if self.reduction == "mean":
            total_loss = losses_per_res.mean()
        elif self.reduction == "sum":
            total_loss = losses_per_res.sum()
        elif self.reduction == "none":
            total_loss = losses_per_res
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        if self.return_details:
            return total_loss, losses_per_res
        return total_loss
