"""Complex-valued activation functions with a unified import namespace.

This module exposes complex activations under a pseudo-namespace "eulero.nn"
so they can be accessed uniformly via:

    from importlib import import_module
    cnn = import_module("eulero.nn")
    act = getattr(cnn, "ModReLU2d")(channels=64)

The alias is created dynamically by registering entries in `sys.modules`.
"""

from __future__ import annotations

import types
import sys
import torch
from torch import nn
import torch.nn.functional as F


class ModReLUScalar(nn.Module):
    """modReLU with a single learnable scalar bias.

    Definition:
        y = ReLU(|x| + b) * x / (|x| + eps), with b ∈ ℝ (shape: ())

    Properties:
    - Phase equivariant: only the magnitude is gated; phase is preserved.
    - Works for complex inputs of arbitrary shape via broadcasting.
    - If `enforce_negative=True`, the bias is constrained to be non-positive
      (stabilizes training by avoiding large positive shifts).

    Args:
        init_bias: Initial value of the scalar bias b.
        eps: Denominator stabilizer for |x|.
        enforce_negative: If True, parameterization ensures b ≤ 0.
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
        return gate * x


class ModReLU1d(nn.Module):
    """Channel-wise modReLU for 1D inputs of shape (B, C, L).

    Definition:
        y = ReLU(|x| + b_c) * x / (|x| + eps), with b_c ∈ ℝ^(1×C×1)

    Args:
        channels: Number of channels C.
        init_bias: Initial value for each channel bias.
        eps: Denominator stabilizer for |x|.
        enforce_negative: If True, constrains b_c ≤ 0.
    """

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


class ModReLU2d(nn.Module):
    """Channel-wise modReLU for 2D inputs of shape (B, C, H, W).

    Definition:
        y = ReLU(|x| + b_c) * x / (|x| + eps), with b_c ∈ ℝ^(1×C×1×1)

    Args:
        channels: Number of channels C.
        init_bias: Initial value for each channel bias.
        eps: Denominator stabilizer for |x|.
        enforce_negative: If True, constrains b_c ≤ 0.
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


class ModReLU2dPerFreq(nn.Module):
    """modReLU with per-frequency bias for spectrogram-like inputs (B, C, F, T).

    Definition:
        y = ReLU(|x| + b_{c,f}) * x / (|x| + eps),
        with b_{c,f} ∈ ℝ^(1×C×F×1), enabling frequency-dependent gating.

    Args:
        channels: Number of channels C.
        n_freq: Number of frequency bins F.
        init_bias: Initial value for each (channel, frequency) bias.
        eps: Denominator stabilizer for |x|.
        enforce_negative: If True, constrains b_{c,f} ≤ 0 via softplus proxy.
    """

    def __init__(self, channels: int, n_freq: int, init_bias: float = -0.2, eps: float = 1e-8, enforce_negative: bool = True):
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


class ComplexGELU2d(nn.Module):
    """Phase-equivariant GELU for complex tensors (B, C, H, W).

    Form:
        y = x * g(|x|), where g is a smooth gating function of the magnitude.

    Parameterization:
        g(r) = 0.5 * (1 + erf((r - μ) / (sqrt(2) * σ)))
        with per-channel parameters μ and log σ.

    Args:
        channels: Number of channels C.
        init_mu: Initial μ.
        init_log_sigma: Initial log σ (σ is softplus(log σ)).
        eps: Stabilizer for σ.
        use_tanh_approx: If True, uses the tanh-based GELU approximation.
    """

    def __init__(
        self,
        channels: int,
        init_mu: float = 0.0,
        init_log_sigma: float = 0.0,
        eps: float = 1e-8,
        use_tanh_approx: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.use_tanh_approx = use_tanh_approx
        self.mu = nn.Parameter(torch.full((1, channels, 1, 1), init_mu, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((1, channels, 1, 1), init_log_sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexGELU2d expects a complex tensor of shape (B, C, H, W).")
        r = x.abs()
        mu = self.mu.to(r.dtype)
        sigma = F.softplus(self.log_sigma).to(r.dtype) + self.eps
        u = (r - mu) / sigma
        if self.use_tanh_approx:
            c = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=r.dtype, device=r.device))
            gate = 0.5 * (1.0 + torch.tanh(c * (u + 0.044715 * (u ** 3))))
        else:
            gate = 0.5 * (1.0 + torch.erf(u / torch.sqrt(torch.tensor(2.0, dtype=r.dtype, device=r.device))))
        return x * gate


class ComplexGELU1d(nn.Module):
    """1D variant of phase-equivariant GELU for inputs (B, C, L).

    Form:
        y = x * g(|x|), with per-channel μ and σ as in ComplexGELU2d.

    Args:
        channels: Number of channels C.
        init_mu: Initial μ.
        init_log_sigma: Initial log σ.
        eps: Stabilizer for σ.
        use_tanh_approx: If True, uses the tanh-based GELU approximation.
    """

    def __init__(
        self,
        channels: int,
        init_mu: float = 0.0,
        init_log_sigma: float = 0.0,
        eps: float = 1e-8,
        use_tanh_approx: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.use_tanh_approx = use_tanh_approx
        self.mu = nn.Parameter(torch.full((1, channels, 1), init_mu, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((1, channels, 1), init_log_sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexGELU1d expects a complex tensor of shape (B, C, L).")
        r = x.abs()
        mu = self.mu.to(r.dtype)
        sigma = F.softplus(self.log_sigma).to(r.dtype) + self.eps
        u = (r - mu) / sigma
        if self.use_tanh_approx:
            c = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=r.dtype, device=r.device))
            gate = 0.5 * (1.0 + torch.tanh(c * (u + 0.044715 * (u ** 3))))
        else:
            gate = 0.5 * (1.0 + torch.erf(u / torch.sqrt(torch.tensor(2.0, dtype=r.dtype, device=r.device))))
        return x * gate

class CReLU(nn.Module):
    """Elementwise ReLU on real and imaginary parts.

    For z = a + i b:
        CReLU(z) = ReLU(a) + i ReLU(b).
    Useful as a simple complex extension of real-valued ReLU.
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("CReLU expects complex input.")
        real = self.relu(z.real)
        imag = self.relu(z.imag)
        return torch.complex(real, imag)


    
class CELU(nn.Module):
    """Elementwise ELU on real and imaginary parts.

    For z = a + i b:
        CELU(z) = ELU(a) + i ELU(b).
    Useful as a simple complex extension of real-valued ELU.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.elu = nn.ELU(**kwargs)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("Cplx_ELU expects complex input.")
        real = self.elu(z.real)
        imag = self.elu(z.imag)
        return torch.complex(real, imag)


class CGLU(nn.Module):
    """GLU applied separately to the real and imaginary parts."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("CGLU expects complex input.")
        real = F.glu(z.real, dim=self.dim)
        imag = F.glu(z.imag, dim=self.dim)
        return torch.complex(real, imag)


class Abs_SiLU(nn.Module):
    """Magnitude-gated SiLU applied to complex inputs.

    Definition:
        Let r = |z|. Then gate = r * sigmoid(r), and
        Abs_SiLU(z) = (gate / (r + eps)) * z.

    This preserves phase while smoothly gating by magnitude.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise TypeError("abs_silu expects complex input.")
        mag = z.abs()
        gate = mag / (1.0 + torch.exp(-mag))
        return gate * z / (mag + 1e-8)


@torch.jit.script
def snake_complex(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    # x: (B, C, L, ...) complesso
    # alpha: (1, C, 1, ...) reale
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)

    # Evita divisioni per zero
    a = alpha + 1e-9

    # Cast di alpha al dtype di x (complesso) per evitare mismatch
    a_c = a.to(x.dtype)
    alpha_c = alpha.to(x.dtype)

    x = x + a_c.reciprocal() * torch.sin(alpha_c * x).pow(2)

    x = x.reshape(shape)
    return x


class Snake1dComplex(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # alpha reale, ma verrà castato a complesso al volo
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x deve essere complesso: dtype=torch.complex64 / complex128
        return snake_complex(x, self.alpha)

# Register alias for "eulero.nn" 

_this = sys.modules[__name__]
_pkg = sys.modules.setdefault("eulero", types.ModuleType("eulero"))
setattr(_pkg, "nn", _this)
sys.modules["eulero.nn"] = _this

__all__ = [
    "ModReLUScalar",
    "ModReLU1d",
    "ModReLU2d",
    "ModReLU2dPerFreq",
    "ComplexGELU1d",
    "ComplexGELU2d",
    "CELU",
    "Abs_SiLU",
    "CReLU",
    "CGLU",
]
