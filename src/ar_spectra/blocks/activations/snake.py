# =============================================================================
# Periodic activations (Snake) adapted to maintain regularity in continuous and complex signals.
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn


# ── Real snake ────────────────────────────────────────────────────────────────

def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake functional form: ``x + (1/α) * sin²(α x)``."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    """Learnable Snake activation for real 1-D tensors (B, C, L)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


# ── Complex snake ─────────────────────────────────────────────────────────────

def snake_complex(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Complex Snake: applies the Snake formula in the complex domain."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    a_c = (alpha + 1e-9).to(x.dtype)
    alpha_c = alpha.to(x.dtype)
    x = x + a_c.reciprocal() * torch.sin(alpha_c * x).pow(2)
    return x.reshape(shape)


class Snake1dComplex(nn.Module):
    """Learnable complex Snake activation for (B, C, L) tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake_complex(x, self.alpha)
