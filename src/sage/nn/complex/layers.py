# Complex-valued layers used by the complex attention (blocks/attention/complex.py).

import torch
import torch.nn as nn


class ComplexDropout(nn.Module):
    """
    Phase-preserving dropout for complex tensors.
    The mask is real and shared by the real and imaginary parts.
    `broadcast_dims` are the dimensions collapsed to 1, which gives
    token-drop or channel-drop variants through broadcasting.
    """
    def __init__(self, p: float = 0.1, broadcast_dims: tuple[int, ...] = ()):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p must be in [0, 1).")
        self.p = float(p)
        self.broadcast_dims = tuple(broadcast_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = list(x.shape)
        for d in self.broadcast_dims:
            shape[d] = 1
        # real mask, the same keep for R and I
        mask = (torch.rand(shape, device=x.device, dtype=x.real.dtype) < keep)
        mask = mask.to(x.real.dtype) / keep
        return x * mask.to(x.dtype)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""


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
