# =============================================================================
# ReLU-family activations for real and complex tensors (e.g., ModReLU, CReLU, CELU).
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Real ReLU wrappers (thin subclasses / aliases) ────────────────────────────

class CELU(nn.Module):
    """Complex-safe CELU: applies ``torch.nn.CELU`` to real and imag separately."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return torch.complex(F.celu(x.real, self.alpha), F.celu(x.imag, self.alpha))
        return F.celu(x, self.alpha)


# ── Complex modReLU family ────────────────────────────────────────────────────

class ModReLUScalar(nn.Module):
    """modReLU with a single learnable scalar bias.

    Definition::
        y = ReLU(|x| + b) * x / (|x| + eps), with b ∈ ℝ

    Properties:
    - Phase equivariant: only the magnitude is gated; phase is preserved.
    - Works for complex inputs of arbitrary shape via broadcasting.

    Args:
        init_bias: Initial value of the scalar bias b.
        eps: Denominator stabilizer for |x|.
        enforce_negative: If True, constrains b ≤ 0 to avoid large positive shifts.
    """

    def __init__(self, init_bias: float = -0.05, eps: float = 1e-8, enforce_negative: bool = True):
        super().__init__()
        self.eps = eps
        self.enforce_negative = enforce_negative
        self.b_free = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ModReLUScalar expects a complex-valued tensor.")
        b = (-abs(self.b_free) if self.enforce_negative else self.b_free).to(x.real.dtype)
        mag = x.abs()
        gate = F.relu(mag + b) / (mag + self.eps)
        return x * gate


class ModReLU2d(nn.Module):
    """Channel-wise modReLU for 2-D feature maps (B, C, H, W).

    Each channel c has its own learnable bias b_c.
    """

    def __init__(self, channels: int, init_bias: float = -0.05, eps: float = 1e-8, enforce_negative: bool = True):
        super().__init__()
        self.eps = eps
        self.enforce_negative = enforce_negative
        self.b_free = nn.Parameter(torch.full((1, channels, 1, 1), init_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ModReLU2d expects a complex tensor of shape (B, C, H, W).")
        b = (-abs(self.b_free) if self.enforce_negative else self.b_free).to(x.real.dtype)
        mag = x.abs()
        gate = F.relu(mag + b) / (mag + self.eps)
        return gate * x


class ModReLU1d(nn.Module):
    """Channel-wise modReLU for 1-D sequences (B, C, L)."""

    def __init__(self, channels: int, init_bias: float = -0.05, eps: float = 1e-8, enforce_negative: bool = True):
        super().__init__()
        self.eps = eps
        self.enforce_negative = enforce_negative
        self.b_free = nn.Parameter(torch.full((1, channels, 1), init_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ModReLU1d expects a complex tensor of shape (B, C, L).")
        b = (-abs(self.b_free) if self.enforce_negative else self.b_free).to(x.real.dtype)
        mag = x.abs()
        gate = F.relu(mag + b) / (mag + self.eps)
        return gate * x


class ModReLU2dPerFreq(nn.Module):
    """modReLU with per-frequency bias for spectrograms (B, C, F, T)."""

    def __init__(self, channels: int, n_freq: int, init_bias: float = -0.2,
                 eps: float = 1e-8, enforce_negative: bool = True):
        super().__init__()
        self.eps = eps
        self.enforce_negative = enforce_negative
        self.b_free = nn.Parameter(torch.full((1, channels, n_freq, 1), init_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ModReLU2dPerFreq expects a complex tensor of shape (B, C, F, T).")
        b = (-F.softplus(self.b_free) if self.enforce_negative else self.b_free).to(x.real.dtype)
        mag = x.abs()
        gate = F.relu(mag + b) / (mag + self.eps)
        return gate * x


# ── Other phase-equivariant ReLU variants ─────────────────────────────────────

class CReLU(nn.Module):
    """Cardioid ReLU: ``y = 0.5 * (1 + cos(phase(x))) * x``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.relu(x)
        phase = torch.angle(x)
        return 0.5 * (1 + torch.cos(phase)) * x


class zReLU(nn.Module):
    """zReLU: passes x only when both real and imaginary parts are non-negative."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.relu(x)
        mask = (x.real >= 0) & (x.imag >= 0)
        return x * mask


class CardioidActivation(nn.Module):
    """Alias for CReLU (cardioid shape in the complex plane)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.relu(x)
        phase = torch.angle(x)
        return 0.5 * (1 + torch.cos(phase)) * x


class CPReLU(nn.Module):
    """Complex PReLU: learnable per-channel slope for the negative-magnitude region."""

    def __init__(self, channels: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.abs()
        slope = self.weight.view(1, -1, *([1] * (x.dim() - 2))).to(x.real.dtype)
        gate = torch.where(mag >= 0, torch.ones_like(mag), slope)
        return x * gate


class SplitReLU(nn.Module):
    """Applies ReLU independently to real and imaginary parts."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.relu(x)
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class magReLU(nn.Module):
    """ReLU applied to the magnitude; phase is passed through unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.relu(x)
        mag = F.relu(x.abs())
        # avoid division by zero
        phase = x / (x.abs() + 1e-8)
        return mag * phase
