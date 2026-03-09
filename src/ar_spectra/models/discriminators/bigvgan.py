
# =============================================================================
# High-quality discriminator for neural vocoders derived from BigVGAN.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as tp
from typing import List, Tuple
from functools import reduce
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class BigVGANDiscriminator(nn.Module):
    def __init__(self, sample_rate: int,
        channels: int = 1,
        use_hinge: bool = False,
        periods: List[int] = [2, 3, 5, 7, 11],
        **cqt_kwargs,
    ):
        super().__init__()

        # Use MPD discriminator from DAC GAN, disable others.
        self.mpd = DACGANLoss(use_hinge=use_hinge, sample_rate=sample_rate,
            periods=periods, rates=[], fft_sizes=[], channels = channels)

        self.cqt = MultiScaleSubbandCQTDiscriminator({
            "cqtd_in_channels": channels,
            "sampling_rate": sample_rate, **cqt_kwargs,
        })

    def loss(self, reals, fakes):
        cqt_dis_loss, cqt_gen_loss, cqt_feature_distance = self.cqt.loss(reals, fakes)
        mpd_dis_loss, mpd_gen_loss, mpd_feature_distance = self.mpd.loss(reals, fakes)
        return (
            mpd_dis_loss + cqt_dis_loss,
            mpd_gen_loss + cqt_gen_loss,
            mpd_feature_distance + cqt_feature_distance)
