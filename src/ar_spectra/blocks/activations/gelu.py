# =============================================================================
# GELU-family activations for real and complex tensors (e.g., ComplexGELU, CGeLU).
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexGELU2d(nn.Module):
    """Phase-equivariant GELU for 2-D complex feature maps (B, C, H, W).

    ``y = x * g(|x|)`` where ``g(r) = 0.5(1 + erf((r - μ) / (√2 σ)))``
    with per-channel learnable μ and log σ.
    """

    def __init__(self, channels: int, init_mu: float = 0.0,
                 init_log_sigma: float = 0.0, eps: float = 1e-8,
                 use_tanh_approx: bool = False):
        super().__init__()
        self.eps = eps
        self.use_tanh_approx = use_tanh_approx
        self.mu = nn.Parameter(torch.full((1, channels, 1, 1), init_mu, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((1, channels, 1, 1), init_log_sigma, dtype=torch.float32))

    def _gate(self, r: torch.Tensor) -> torch.Tensor:
        mu = self.mu.to(r.dtype)
        sigma = F.softplus(self.log_sigma).to(r.dtype) + self.eps
        u = (r - mu) / sigma
        if self.use_tanh_approx:
            c = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=r.dtype, device=r.device))
            return 0.5 * (1.0 + torch.tanh(c * (u + 0.044715 * u ** 3)))
        return 0.5 * (1.0 + torch.erf(u / torch.sqrt(torch.tensor(2.0, dtype=r.dtype, device=r.device))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexGELU2d expects a complex tensor of shape (B, C, H, W).")
        return x * self._gate(x.abs())


class ComplexGELU1d(nn.Module):
    """Phase-equivariant GELU for 1-D sequences (B, C, L)."""

    def __init__(self, channels: int, init_mu: float = 0.0,
                 init_log_sigma: float = 0.0, eps: float = 1e-8,
                 use_tanh_approx: bool = False):
        super().__init__()
        self.eps = eps
        self.use_tanh_approx = use_tanh_approx
        self.mu = nn.Parameter(torch.full((1, channels, 1), init_mu, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((1, channels, 1), init_log_sigma, dtype=torch.float32))

    def _gate(self, r: torch.Tensor) -> torch.Tensor:
        mu = self.mu.to(r.dtype)
        sigma = F.softplus(self.log_sigma).to(r.dtype) + self.eps
        u = (r - mu) / sigma
        if self.use_tanh_approx:
            c = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=r.dtype, device=r.device))
            return 0.5 * (1.0 + torch.tanh(c * (u + 0.044715 * u ** 3)))
        return 0.5 * (1.0 + torch.erf(u / torch.sqrt(torch.tensor(2.0, dtype=r.dtype, device=r.device))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexGELU1d expects a complex tensor of shape (B, C, L).")
        return x * self._gate(x.abs())


class CGeLU(nn.Module):
    """Complex GELU: applies GELU to real and imaginary parts independently."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            return F.gelu(x)
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))
