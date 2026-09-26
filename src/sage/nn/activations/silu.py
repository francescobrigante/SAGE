# =============================================================================
# SiLU / Swish based activations adapted for complex signals.
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSiLU(nn.Module):
    """Complex SiLU (Swish): ``y = x * σ(x)`` applied to real and imag."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.silu(x)
        return torch.complex(F.silu(x.real), F.silu(x.imag))


class Abs_SiLU(nn.Module):
    """Magnitude-gated SiLU for complex inputs.

    ``y = (|z| * σ(|z|) / (|z| + ε)) * z``
    Preserves phase while gating by magnitude via SiLU.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("Abs_SiLU expects complex input.")
        mag = z.abs()
        gate = mag / (1.0 + torch.exp(-mag))
        return gate * z / (mag + 1e-8)


class CSwish(nn.Module):
    """Complex Swish (same as CSiLU, kept for naming compatibility)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.silu(x)
        return torch.complex(F.silu(x.real), F.silu(x.imag))
