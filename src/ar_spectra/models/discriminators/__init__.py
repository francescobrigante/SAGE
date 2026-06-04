
# =============================================================================
# Global registry to import discriminator functions conforming to eulero.nn / GAN models.
# =============================================================================

from .encodec import EncodecDiscriminator, SharedDiscriminatorConvNet
from .multi import MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiDiscriminator
from .oobleck import OobleckDiscriminator, MPD, MSD, MRD
from .subband import MultiScaleSubbandCQTDiscriminator
from .dac import DACDiscriminator, DACGANLoss
from .bigvgan import BigVGANDiscriminator
from .transformer import (
    MultiTransformerDiscriminator,
    TransformerMultiSTFTDiscriminator,
    TransformerMultiPatchedDiscriminator,
    TransformerMultiWaveletDiscriminator,
    TransformerMultiChromaDiscriminator,
)
from .hil import HILDiscriminator, MultiFilterBankDiscriminator

__all__ = [
    "EncodecDiscriminator", "SharedDiscriminatorConvNet",
    "MultiScaleDiscriminator", "MultiPeriodDiscriminator", "MultiDiscriminator",
    "OobleckDiscriminator", "MPD", "MSD", "MRD",
    "MultiScaleSubbandCQTDiscriminator",
    "DACDiscriminator", "DACGANLoss",
    "BigVGANDiscriminator",
    "MultiTransformerDiscriminator",
    "TransformerMultiSTFTDiscriminator",
    "TransformerMultiPatchedDiscriminator",
    "TransformerMultiWaveletDiscriminator",
    "TransformerMultiChromaDiscriminator",
    "HILDiscriminator", "MultiFilterBankDiscriminator",
]
