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

