
# =============================================================================
# Global registry to import discriminator functions conforming to eulero.nn / GAN models.
# =============================================================================

from .encodec import EncodecDiscriminator, SharedDiscriminatorConvNet
from .multi import MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiDiscriminator
from .oobleck import OobleckDiscriminator, MPD, MSD, MRD
from .subband import MultiScaleSubbandCQTDiscriminator
from .dac import DACDiscriminator, DACGANLoss
from .bigvgan import BigVGANDiscriminator
