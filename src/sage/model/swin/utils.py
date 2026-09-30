# ===============================================================
# Swin Transformer V2 — shared utilities.
# make_norm:          is_complex-aware LayerNorm factory.
# make_drop_path:     is_complex-aware DropPath factory.
# init_swin_weights:  weight init for real and complex Linear/LN.
# random_patch_mask:  MIM-style augmentation — zero-out random 2D patches.
# ===============================================================

from typing import Optional, Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath


def make_norm(dim: int, is_complex: bool) -> nn.Module:
    """Return a LayerNorm appropriate for real or complex tokens.

    Args:
        dim: Normalised feature dimension.
        is_complex: If True returns ComplexLayerNorm (complextorch),
            otherwise standard nn.LayerNorm.
    """
    if is_complex:
        from sage.nn.complex.normalization import ComplexLayerNorm   # optional [complex] extra
        return ComplexLayerNorm(dim)
    return nn.LayerNorm(dim)


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
    - ``ComplexLayerNorm``:    self-initializes in __init__ (reset_parameters).

    Usage::

        model.apply(init_swin_weights)
        for stage in model.stages: stage._init_respostnorm()
    """
    if isinstance(m, nn.Linear):
        if m.weight.is_complex():
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
    # ComplexLayerNorm: __init__ already calls reset_parameters() (weight=(1/√2)·I₂, bias=0).
    # norm1/norm2 inside blocks are then zeroed by _init_respostnorm(); other norms keep (1/√2)·I₂.


def random_patch_mask(
    spec: torch.Tensor,
    patch_size: Tuple[int, int],
    mask_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Zero-out random 2-D patches of a (B, C, F, T) spectrogram (MIM-style augmentation).

    Used as a denoising-VAE augmentation: the encoder receives a perturbed input where
    `mask_ratio` fraction of patches are set to zero, while the reconstruction target
    (handled in the engine) remains the unmasked original. This forces the encoder to
    use spatial context to fill in the missing patches → context-aware (more semantic)
    latent representation.

    Differences vs canonical SimMIM:
      - No learnable `mask_token` — we just zero-out (simpler, dtype-agnostic for complex64).
      - No prediction head, no masked-only loss: standard reconstruction loss is computed
        on the FULL output. Structurally a Denoising-VAE with structured (patch-aligned) noise.
      - Mask ratio is much lower (~15%) than canonical SimMIM (~60%) since this is
        augmentation, not pretraining.

    Per-sample mask: each batch element draws its own random patch mask, independently.
    All channels of a sample share the same mask (stereo Re/Im or L/R drop together —
    physically meaningful: a missing time-frequency region is missing in both ears).

    Args:
        spec: Spectrogram tensor of shape (B, C, F, T). Real or complex dtype.
        patch_size: (patch_h, patch_w) — must divide F and T exactly.
        mask_ratio: Fraction of patches to mask in [0, 1]. 0 disables (no-op).
        generator: Optional torch.Generator for deterministic masking (per-step seed).

    Returns:
        Same-shape, same-dtype tensor with random patches zeroed.
    """
    if mask_ratio <= 0.0:
        return spec

    B, C, F, T = spec.shape
    ph, pw = patch_size
    if F % ph != 0 or T % pw != 0:
        raise ValueError(
            f"random_patch_mask: F={F}, T={T} not divisible by patch_size=({ph},{pw})"
        )

    nF, nT = F // ph, T // pw                                              # patch grid resolution
    n_patches = nF * nT
    n_mask = int(round(n_patches * mask_ratio))                            # number of patches to drop
    if n_mask == 0:
        return spec

    # Per-sample independent mask via top-k of uniform noise — vectorised, no Python loop.
    # rand: (B, n_patches) ∈ [0, 1); kthvalue gives the n_mask-th smallest value per row;
    # patches with value <= threshold get masked (exactly n_mask per sample, modulo ties).
    rand = torch.rand(B, n_patches, device=spec.device, generator=generator)
    threshold = rand.kthvalue(n_mask, dim=1, keepdim=True).values          # (B, 1)
    mask_flat = (rand <= threshold)                                         # (B, n_patches) bool — True = drop

    # Reshape to patch grid then upsample to full (F, T) resolution by patch repeat.
    mask = mask_flat.view(B, 1, nF, nT)                                    # (B, 1, nF, nT)
    mask = mask.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)  # (B, 1, F, T)

    # keep = (1 − mask), cast to spec real dtype. complex64 * float32 broadcasts cleanly,
    # preserving complex output. Channel dim (C) is broadcast — all channels share the mask.
    keep = (~mask).to(spec.real.dtype if spec.is_complex() else spec.dtype)
    return spec * keep                                                      # (B, C, F, T)
