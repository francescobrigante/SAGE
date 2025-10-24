import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

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
        Xv = X.view(B, C, 2, F, T).contiguous()
        real = Xv[:, :, 0, :, :]
        imag = Xv[:, :, 1, :, :]
        return torch.complex(real, imag)
    elif X.ndim == 3:
        C2, F, T = X.shape
        if C2 % 2 != 0:
            raise ValueError(f"Channel dimension must be even for complex-as-channels. Got {C2}.")
        C = C2 // 2
        Xv = X.view(C, 2, F, T).contiguous()
        real = Xv[:, 0, :, :]
        imag = Xv[:, 1, :, :]
        return torch.complex(real, imag)

    raise ValueError("Unsupported spectrogram shape. Expected (..., C, F, T), (..., C, F, T, 2) or (B, 2C, F, T)/(2C, F, T).")

class PerceptuallyWeightedComplexMSE(nn.Module):
    r"""
    Computes a perceptually-motivated Mean Squared Error on complex-valued spectrograms.

    This loss function calculates a p-norm of the relative error between a predicted
    spectrogram `S_hat` and a target spectrogram `S`. It includes an optional
    energy-based weighting mechanism to focus the model's attention on perceptually
    significant time-frequency bins, mimicking psychoacoustic masking principles.

    The per-bin loss `L` for a given time-frequency bin is defined as:
    $$
    L = w \cdot \left( \frac{|\hat{S} - S|}{d + \epsilon} \right)^p
    $$
    where:
    - $|\hat{S} - S|$ is the magnitude of the complex error vector.
    - `d` is a normalization factor derived from the magnitudes of `S` and `S_hat`.
    - `p` is the exponent of the norm (e.g., 2.0 for MSE, 1.0 for MAE).
    - `w` is an optional perceptual weight based on the bin's energy.

    Args:
        normalize (str, optional): The method for normalizing the error. Must be one
            of {'avg', 'ref'}.
            - 'avg': Normalizes by the arithmetic mean of the magnitudes, `d = 0.5 * (|\hat{S}| + |S|)`.
            - 'ref': Normalizes by the magnitude of the reference signal, `d = |S|`.
            Defaults to "avg".
        p (float, optional): The exponent for the p-norm. `p=2.0` corresponds to a
            squared error, while `p=1.0` corresponds to an absolute error. Defaults to 2.0.
        eps (float, optional): A small constant added to the denominator for numerical
            stability. Defaults to 1e-7.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
        dim (Optional[Sequence[int]], optional): The dimensions over which to reduce the
            loss. If None, reduces over all dimensions. Defaults to None.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not.
            Defaults to False.
        bin_weighted (bool, optional): If True, enables the perceptual energy-based
            weighting mechanism. Defaults to False.
        bin_ref (str, optional): The reference for calculating the energy weight `w`.
            Must be one of {'avg', 'ref'}.
            - 'avg': Uses the average magnitude of `S` and `S_hat`.
            - 'ref': Uses the magnitude of the reference `S`.
            Defaults to "avg".
        bin_power (float, optional): The exponent applied to the normalized energy to
            create the weight `w`. Values > 1.0 focus more on high-energy bins, while
            values < 1.0 create a softer weighting. Defaults to 1.0.
    """
    def __init__(
        self,
        *,
        normalize: str = "avg",
        p: float = 2.0,
        eps: float = 1e-7,
        reduction: str = "mean",
        dim: Optional[Sequence[int]] = None,
        keepdim: bool = False,
        bin_weighted: bool = False,
        bin_ref: str = "avg",
        bin_power: float = 1.0,
    ):
        super().__init__()
        if normalize not in {"avg", "ref"}:
            raise ValueError(f"normalize must be 'avg' or 'ref', but got {normalize}.")
        if bin_ref not in {"avg", "ref"}:
            raise ValueError(f"bin_ref must be 'avg' or 'ref', but got {bin_ref}.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', but got {reduction}.")

        self.normalize = normalize
        self.p = p
        self.eps = eps
        self.reduction = reduction
        self.dim = dim
        self.keepdim = keepdim
        self.bin_weighted = bin_weighted
        self.bin_ref = bin_ref
        self.bin_power = bin_power

    def forward(
        self,
        S_hat: torch.Tensor,
        S: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Convert to complex if inputs are RI or CAC from dataset
        S_hat = to_complex_spectrogram(S_hat)
        S = to_complex_spectrogram(S)
        # --- Input Validation ---
        if S_hat.shape != S.shape:
            raise ValueError(f"Input shapes must match after conversion, but got {S_hat.shape} and {S.shape}.")

        # --- Core Loss Calculation ---
        error_mag = (S_hat - S).abs()
        abs_hat = S_hat.abs()
        abs_ref = S.abs()

        # Normalization denominator
        if self.normalize == "avg":
            d = 0.5 * (abs_hat + abs_ref)
        else:  # "ref"
            d = abs_ref
        d = d + self.eps

        loss_tensor = (error_mag / d).pow(self.p)

        # --- Perceptual Bin Weighting ---
        if self.bin_weighted:
            # Reference energy for weighting
            if self.bin_ref == "avg":
                E = 0.5 * (abs_hat + abs_ref)
            else:  # "ref"
                E = abs_ref

            # Normalize energy across specified dimensions to [0, 1]
            reduce_dims = tuple(range(loss_tensor.ndim)) if self.dim is None else tuple(self.dim)
            E_max = E.amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
            E_norm = E / E_max

            # Compute and apply weights
            w = E_norm.pow(self.bin_power)
            loss_tensor = loss_tensor * w

        # --- Optional External Weighting ---
        if weight is not None:
            loss_tensor = loss_tensor * weight

        # --- Final Reduction ---
        if self.reduction == "none":
            return loss_tensor

        reduce_dims = tuple(range(loss_tensor.ndim)) if self.dim is None else tuple(self.dim)
        if self.reduction == "mean":
            return loss_tensor.mean(dim=reduce_dims, keepdim=self.keepdim)
        else:  # "sum"
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