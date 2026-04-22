# ===============================================================
# Swin Transformer V2 — shared utilities.
# make_norm:          is_complex-aware LayerNorm factory.
# make_drop_path:     is_complex-aware DropPath factory.
# init_swin_weights:  weight init for real and complex Linear/LN.
# ===============================================================

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
from ar_spectra.blocks.normalization.complex import ComplexLayerNorm


def make_norm(dim: int, is_complex: bool) -> nn.Module:
    """Return a LayerNorm appropriate for real or complex tokens.

    Args:
        dim: Normalised feature dimension.
        is_complex: If True returns ComplexLayerNorm (complextorch),
            otherwise standard nn.LayerNorm.
    """
    return ComplexLayerNorm(dim) if is_complex else nn.LayerNorm(dim)


class ComplexSafeDropPath(nn.Module):
    """Stochastic depth (drop-path) compatible with complex64 tensors.

    timm's DropPath calls ``x.new_empty(...).bernoulli_()`` which fails on
    complex dtype. This version generates a float32 Bernoulli mask and
    broadcasts it over the complex tensor — mathematically identical, safe
    for any dtype.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # probability of dropping a sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Real mask — shape (B, 1, 1, ...) to broadcast over all spatial/channel dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=torch.float32, device=x.device).bernoulli_(keep_prob)
        mask.div_(keep_prob)                                           # scale to preserve expectation
        return x * mask                                                # broadcast: works for real and complex


def make_drop_path(drop_prob: float, is_complex: bool) -> nn.Module:
    """Return a DropPath module appropriate for real or complex tensors.

    Args:
        drop_prob: Probability of dropping the sample (stochastic depth).
        is_complex: If True uses ComplexSafeDropPath; otherwise timm DropPath.
    """
    if drop_prob <= 0.0:
        return nn.Identity()
    return ComplexSafeDropPath(drop_prob) if is_complex else DropPath(drop_prob)


def init_swin_weights(m: nn.Module) -> None:
    """Swin V2 weight initialisation for real and complex linear layers.

    - ``nn.Linear`` (real):    truncated normal (std=0.02) on weight, zero bias.
    - ``nn.Linear`` (complex): truncated normal applied separately to real and
      imaginary parts (std=0.02 / sqrt(2) each so total variance ≈ 0.02²).
    - ``nn.LayerNorm``:        weight=1, bias=0.

    Usage::

        model.apply(init_swin_weights)
    """
    if isinstance(m, nn.Linear):
        if m.weight.is_complex():
            # Initialise real and imaginary parts independently
            # std / sqrt(2) keeps total complex weight variance at std²
            trunc_normal_(m.weight.real, std=0.02 / (2 ** 0.5))
            trunc_normal_(m.weight.imag, std=0.02 / (2 ** 0.5))
        else:
            trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
