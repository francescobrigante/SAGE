# =============================================================================
# L_STFT of the paper (eq. 2): squared error between the power-compressed complex
# STFTs. The other spectral losses tried during development live in
# sage.nn.losses.experimental.spectral.
# =============================================================================
from typing import Optional, Sequence

import torch
import torch.nn as nn

from sage.utils.spectral import to_complex_spectrogram


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
