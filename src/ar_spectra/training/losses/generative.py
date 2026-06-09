# ============================================================================
# generative.py — Generative-family latent-alignment losses
# (LATENT_ALIGNMENT_PLAN.md Fase 6). Unlike the semantic losses, these do NOT
# align the latent to an external teacher: they make it more *generable* for the
# downstream latent-diffusion model. First member: LatentFlowMatchingLoss, a
# jointly-trained flow-matching DiT whose gradient flows back into the encoder.
# ============================================================================
from __future__ import annotations

import torch
from torch.nn import functional as F

from .base import LossModule
from .semantic import standardize_bottleneck       # shared Fase-0 latent featurizer
from ...models.latent_dit import LatentDiT


class LatentFlowMatchingLoss(LossModule):
    """Rectilinear flow-matching loss on the VAE latent (SAME §3.3.1).

    A small DiT learns to generate the latent from Gaussian noise via flow-matching;
    the velocity-prediction gradient back-propagates into the encoder, pushing it
    toward a latent that is easier to generate. Self-contained on the latent — there
    is NO MERT teacher and NO temporal reconciliation, so §0.2 does not apply here.

    The DiT is owned by ``LossManager.flow_dit`` (so its params are collected into the
    auxiliary optimizer ``opt_aux``); this module holds a reference to the same object.

    Args:
        flow_dit: the velocity network ``v_θ`` (also registered as ``LossManager.flow_dit``).
        name: loss name for logging.
        weight: ``λ_diff`` — final contribution weight.
        detach_warmup_steps: while ``global_step`` is below this, the latent is detached
            so the DiT trains without the gradient escaping into the encoder (anti
            prior-escape, SAME/SALAD schedule).
        scale_invariant: if True, divide the latent by its own per-sample std before
            building the flow target (degree-0 homogeneous, NOT detached). Makes the
            flow loss blind to latent scale: the encoder gradient is orthogonal to
            scale, removing the variance-collapse incentive (shrinking the latent
            toward the noise prior no longer lowers L_diff). The real latent
            (encoder→decoder, downstream) is untouched — this is loss-local only.
    """

    def __init__(self, flow_dit: LatentDiT, name: str = "latent_flow_loss",
                 weight: float = 1.0, detach_warmup_steps: int = 10000,
                 scale_invariant: bool = True):
        super().__init__(name=name, weight=weight)
        self.flow_dit = flow_dit                          # shared with LossManager.flow_dit
        self.detach_warmup_steps = int(detach_warmup_steps)  # encoder-grad gate (steps)
        self.scale_invariant = bool(scale_invariant)         # divide z by its detached std

    def forward(self, info: dict) -> torch.Tensor:
        z = info["latents"]                                            # (B, C, T) latent
        if info.get("global_step", 0) < self.detach_warmup_steps:
            z = z.detach()                                             # train DiT, gate encoder grad
        z = standardize_bottleneck(z)                                  # (B, D, T) flat featurized
        if self.scale_invariant:
            # Scale-invariant flow loss. NOT detached on purpose: z/std(z) is degree-0
            # homogeneous, so the encoder gradient is orthogonal to latent scale — it
            # reshapes structure but exerts zero pull on variance → no collapse incentive.
            # clamp_min (not +eps): divides by *exactly* std in the normal regime, so the
            # homogeneity is exact (zero residual scale-gradient); the floor only guards
            # the degenerate std→0 case, untouched at any realistic latent scale.
            std = z.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)      # (B, 1, 1)
            z = z / std                                               # (B, D, T) unit-std
        t = torch.sigmoid(torch.randn(z.size(0), device=z.device))     # logit-normal t∈(0,1)
        t_b = t.view(-1, 1, 1)                                         # (B, 1, 1) broadcast over D,T
        eps = torch.randn_like(z)                                     # (B, D, T) Gaussian noise
        z_t = (1.0 - t_b) * z + t_b * eps                              # (B, D, T) interpolant
        target = eps - z                                              # (B, D, T) rectilinear velocity
        v = self.flow_dit(z_t, t)                                     # (B, D, T) predicted velocity
        self.decay_weight()
        return self.weight * F.mse_loss(v, target)
