#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch

from ar_spectra.blocks.normalization import LayerNorm
from ar_spectra.blocks.conv import NormLinear
from ar_spectra.blocks.activations import get_activation


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation: str = "relu", is_complex=True):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = NormLinear(idim, hidden_units, is_complex=is_complex)
        self.w_2 = NormLinear(hidden_units, idim, is_complex=is_complex)
        self.dropout = ComplexDropout(dropout_rate) if is_complex else torch.nn.Dropout(dropout_rate)
        self.activation = get_activation(activation, is_complex=is_complex)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, adim=None, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim if adim is None else adim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = LayerNorm(hidden_units)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))

import torch
import torch.nn as nn

class ComplexDropout(nn.Module):
    """
    Dropout phase-preserving per tensori complessi.
    La maschera è reale e condivisa su parte reale/immaginaria.
    `broadcast_dims` specifica le dimensioni da comprimere a 1 per ottenere
    varianti tipo token-drop o channel-drop via broadcasting.
    """
    def __init__(self, p: float = 0.1, broadcast_dims: tuple[int, ...] = ()):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p deve stare in [0,1).")
        self.p = float(p)
        self.broadcast_dims = tuple(broadcast_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = list(x.shape)
        for d in self.broadcast_dims:
            shape[d] = 1
        # maschera reale, stesso keep per R e I
        mask = (torch.rand(shape, device=x.device, dtype=x.real.dtype) < keep)
        mask = mask.to(x.real.dtype) / keep
        return x * mask.to(x.dtype)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import torch


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)])

import torch
import torch.nn as nn

class ComplexLinearHalf(nn.Module):
    def __init__(self, in_f, out_f, bias=False, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.Wr = nn.Parameter(torch.empty(out_f, in_f, device=device, dtype=dtype))
        self.Wi = nn.Parameter(torch.empty(out_f, in_f, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.Wr); nn.init.kaiming_uniform_(self.Wi)
        self.bias = bias
        if bias:
            self.br = nn.Parameter(torch.zeros(out_f, device=device, dtype=torch.float32))
            self.bi = nn.Parameter(torch.zeros(out_f, device=device, dtype=torch.float32))
            self.bias = True

    def forward(self, x):  # x: complex, real/imag in half
        xr, xi = x.real.to(self.Wr.dtype), x.imag.to(self.Wr.dtype)
        # accumulo in fp32 per stabilità
        yr = (self.Wr @ xr.T - self.Wi @ xi.T).to(torch.float32).T
        yi = (self.Wr @ xi.T + self.Wi @ xr.T).to(torch.float32).T
        if self.bias:
            yr = yr + self.br
            yi = yi + self.bi
        return torch.complex(yr, yi)

class CLinear(nn.Module):
    """
    Complex Linear layer to bypass PyTorch autograd bugs on complex dtypes (F.linear).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=torch.complex64)

    @property
    def weight(self):
        return self.linear.weight
        
    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight
        bias = self.linear.bias
        out_r = torch.nn.functional.linear(x.real, weight.real) - torch.nn.functional.linear(x.imag, weight.imag)
        out_i = torch.nn.functional.linear(x.real, weight.imag) + torch.nn.functional.linear(x.imag, weight.real)
        if bias is not None:
            out_r = out_r + bias.real
            out_i = out_i + bias.imag
        return torch.complex(out_r, out_i)

