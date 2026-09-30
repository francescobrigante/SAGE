# =============================================================================
# Miscellaneous complex activations (Tanh, GLU, Mish, Softplus).
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTanh(nn.Module):
    """Complex Tanh: applies tanh to real and imaginary parts independently."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return torch.tanh(x)
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


class CGLU(nn.Module):
    """Gated Linear Unit applied separately to real and imaginary parts."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("CGLU expects complex input.")
        return torch.complex(F.glu(z.real, dim=self.dim), F.glu(z.imag, dim=self.dim))


class CMish(nn.Module):
    """Complex Mish: applies Mish to real and imaginary parts independently."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.mish(x)
        return torch.complex(F.mish(x.real), F.mish(x.imag))


class CSoftplus(nn.Module):
    """Complex Softplus: applies Softplus to real and imaginary parts."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.softplus(x)
        return torch.complex(F.softplus(x.real), F.softplus(x.imag))


class CCeLU(nn.Module):
    """Complex CELU: applies CELU to real and imaginary parts."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.celu(x, self.alpha)
        return torch.complex(F.celu(x.real, self.alpha), F.celu(x.imag, self.alpha))

