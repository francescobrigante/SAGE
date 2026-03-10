"""Shared types for discriminator modules."""

# =============================================================================
# Definition of Shared Types and Constants (Type Dict, Alias) used globally by various discriminators.
# =============================================================================

from typing import NamedTuple, List, Dict
import torch

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
):
    """Standard hinge GAN losses.

    Returns
    -------
    dis_loss : discriminator hinge loss
    gen_loss : generator adversarial loss
    """
    dis_loss = torch.relu(1.0 - scores_real).mean() + torch.relu(1.0 + scores_fake).mean()
    gen_loss = -scores_fake.mean()
    return dis_loss, gen_loss

