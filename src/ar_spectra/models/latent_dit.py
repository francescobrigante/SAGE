# ============================================================================
# latent_dit.py — Small DiT over the VAE latent for flow-matching alignment
# (LATENT_ALIGNMENT_PLAN.md Fase 6 / SAME §3.3.1). Wraps ContinuousTransformer;
# a timestep is mapped through Fourier features into the AdaLN global-cond path.
# ============================================================================
from __future__ import annotations

import math

import torch
from torch import nn

from ..blocks.transformer_sat import ContinuousTransformer


class FourierFeatures(nn.Module):
    """Random Fourier embedding of a scalar timestep ``t∈[0,1]`` → ``(B, out_features)``."""

    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0, "out_features must be even (cos/sin halves)."
        self.in_features = in_features        # scalar timestep → in_features (=1) columns
        # Fixed random projection (non-trainable) — a buffer so it moves with the module.
        self.register_buffer("weight", torch.randn(out_features // 2, in_features) * std)

    def forward(self, t: torch.Tensor) -> torch.Tensor:        # t: (B, in_features)
        f = 2 * math.pi * t @ self.weight.T                    # (B, out_features/2)
        return torch.cat([f.cos(), f.sin()], dim=-1)           # (B, out_features)


class LatentDiT(nn.Module):
    """Predicts the flow-matching velocity ``v_θ(z_t, t)`` on the flat latent ``(B, D, T)``.

    A thin wrapper over ``ContinuousTransformer`` (the SAME-style backbone already in
    ``blocks/transformer_sat.py``, pre-built for exactly this): a Fourier timestep
    embedding feeds the transformer's AdaLN global-conditioning path, and the in/out
    projections map the latent channel dim ``D`` to the transformer width and back. No
    transformer is written from scratch here.

    Args:
        latent_dim: channel dim ``D`` of the featurized latent (``standardize_bottleneck``).
        hidden: transformer width.
        depth: number of transformer blocks (SAME uses 4).
        fourier_dim: dimensionality of the timestep Fourier embedding.
    """

    def __init__(self, latent_dim: int, hidden: int = 768, depth: int = 4,
                 fourier_dim: int = 256):
        super().__init__()
        self.t_embed = FourierFeatures(1, fourier_dim)         # scalar t → cond vector
        self.transformer = ContinuousTransformer(
            dim=hidden, depth=depth, dim_in=latent_dim, dim_out=latent_dim,
            global_cond_dim=fourier_dim, rotary_pos_emb=True,
            zero_init_branch_outputs=True,                     # stable joint-train init
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z_t`` ``(B, D, T)``, ``t`` ``(B,)`` → predicted velocity ``(B, D, T)``."""
        cond = self.t_embed(t.unsqueeze(-1))                   # (B, fourier_dim)
        x = z_t.transpose(1, 2)                                # (B, D, T) → (B, T, D)
        v = self.transformer(x, global_cond=cond)              # (B, T, D)
        return v.transpose(1, 2)                               # (B, T, D) → (B, D, T)
