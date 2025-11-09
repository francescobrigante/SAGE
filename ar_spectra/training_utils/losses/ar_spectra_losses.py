import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple
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
    ):
        super().__init__()
        if len(fft_sizes) != len(hop_sizes):
            raise ValueError("fft_sizes and hop_sizes must have the same length.")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.eps = eps
        self.window = window
        self.win_lengths = win_lengths if win_lengths is not None else fft_sizes

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

            sc = ComplexSpectralConvergence()(S_hat, S_gt)  # scalar per-batch (mean)
            sc_vals.append(sc)

        sc_vals = torch.stack(sc_vals, dim=0)  # (R,)
        if reduction == "mean":
            return sc_vals.mean()
        elif reduction == "sum":
            return sc_vals.sum()
        elif reduction == "none":
            return sc_vals

