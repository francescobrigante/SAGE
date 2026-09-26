
# =============================================================================
# Registry of the GAN discriminators (the paper uses WavTokenizer; the others are alternatives).
# =============================================================================

from sage.nn.discriminators.encodec import EncodecDiscriminator, SharedDiscriminatorConvNet
from sage.nn.discriminators.multi import MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiDiscriminator
from sage.nn.discriminators.oobleck import OobleckDiscriminator, MPD, MSD, MRD
from sage.nn.discriminators.subband import MultiScaleSubbandCQTDiscriminator
from sage.nn.discriminators.dac import DACDiscriminator, DACGANLoss
from sage.nn.discriminators.bigvgan import BigVGANDiscriminator
from sage.nn.discriminators.transformer import (
    MultiTransformerDiscriminator,
    TransformerMultiSTFTDiscriminator,
    TransformerMultiPatchedDiscriminator,
    TransformerMultiWaveletDiscriminator,
    TransformerMultiChromaDiscriminator,
)
from sage.nn.discriminators.hil import HILDiscriminator, MultiFilterBankDiscriminator
from sage.nn.discriminators.wavtokenizer import WavTokenizerDiscriminator, WavTokenizerGANLoss

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
    "WavTokenizerDiscriminator", "WavTokenizerGANLoss",
]
