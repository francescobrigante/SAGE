"""Shared types and GAN loss helpers for discriminator modules."""

# =============================================================================
# Shared types, constants, and GAN loss functions used across all discriminators.
# Includes hinge (EnCodec/Oobleck), relativistic (RpGAN), and sigmoid-relativistic losses.
# =============================================================================

from typing import NamedTuple, List, Dict
import torch
import torch.nn.functional as F

TensorDict = Dict[str, torch.Tensor]

# Default frequency-band boundaries used by multi-band discriminators (Hz)
BANDS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


class IndividualDiscriminatorOut(NamedTuple):
    """Output of a single discriminator branch."""
    logits: torch.Tensor
    feature_maps: List[torch.Tensor]


def get_hinge_losses(
    scores_real: torch.Tensor,
    scores_fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard hinge GAN losses (EnCodec / Oobleck style).

    Returns:
        dis_loss: discriminator hinge loss
        gen_loss: generator adversarial loss
    """
    dis_loss = torch.relu(1.0 - scores_real).mean() + torch.relu(1.0 + scores_fake).mean()
    gen_loss = -scores_fake.mean()
    return dis_loss, gen_loss


def get_relativistic_losses(
    scores_real: torch.Tensor,
    scores_fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relativistic GAN losses (RpGAN, Jolicoeur-Martineau 2019).

    Discriminator sees the *relative* realism difference rather than
    absolute scores, making training more stable at high quality regimes.

    Returns:
        dis_loss: softplus(-real + fake)
        gen_loss: softplus(real - fake)
    """
    diff = scores_real - scores_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss


def get_sigmoid_relgan_losses(
    scores_real: torch.Tensor,
    scores_fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid relativistic GAN losses (sigmoid-RpGAN variant).

    A softer version of relativistic GAN: uses sigmoid instead of softplus.
    Bounded losses → more stable gradient magnitudes throughout training.

    Returns:
        dis_loss: 2 * sigmoid(0.5 * (fake - real))
        gen_loss: 2 * sigmoid(0.5 * (real - fake))
    """
    diff = 0.5 * (scores_fake - scores_real)
    dis_loss = 2.0 * torch.sigmoid(diff).mean()
    gen_loss = 2.0 * torch.sigmoid(-diff).mean()
    return dis_loss, gen_loss

