
# =============================================================================
# Sub-band discrimination for analyzing metrics over specific ranges (e.g., CQT).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        # Using get with defaults
        self.cfg["cqtd_filters"] = self.cfg.get("cqtd_filters", 32)
        self.cfg["cqtd_max_filters"] = self.cfg.get("cqtd_max_filters", 1024)
        self.cfg["cqtd_filters_scale"] = self.cfg.get("cqtd_filters_scale", 1)
        self.cfg["cqtd_dilations"] = self.cfg.get("cqtd_dilations", [1, 2, 4])
        self.cfg["cqtd_in_channels"] = self.cfg.get("cqtd_in_channels", 1)
        self.cfg["cqtd_out_channels"] = self.cfg.get("cqtd_out_channels", 1)
        # Multi-scale params to loop over
        self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
        self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
        self.cfg["cqtd_bins_per_octaves"] = self.cfg.get(
            "cqtd_bins_per_octaves", [24, 36, 48])
        self.cfg["cqtd_fmin"] = self.cfg.get("fmin", 32.7)

        n_discriminators = len(self.cfg["cqtd_hop_lengths"])
        self.discriminators = nn.ModuleList([CQTDiscriminator(    # type: ignore
            self.cfg,
            hop_length=self.cfg["cqtd_hop_lengths"][i],
            n_octaves=self.cfg["cqtd_n_octaves"][i],
            bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
        ) for i in range(n_discriminators)])

    def forward(self, reals: torch.Tensor, gens: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = checkpoint(disc,reals)
            y_d_g, fmap_g = checkpoint(disc,gens)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def discriminator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake.clone().detach())

        loss_d = 0
        for x_fake, x_real in zip(y_fake, y_real):
            loss_d += torch.mean(x_fake ** 2)
            loss_d += torch.mean((1 - x_real) ** 2)
        loss_d /= len(y_fake)
        return loss_d

    def generator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake)

        loss_g = 0
        for x_fake in y_fake:
            loss_g += torch.mean((1 - x_fake) ** 2)

        counter = 0
        loss_feature = 0
        for i in range(len(fmap_fake)):
            for j in range(len(fmap_fake[i])):
                denominator = fmap_real[i][j].abs().mean().detach()
                loss_feature += F.l1_loss(fmap_fake[i][j], fmap_real[i][j].detach()) / denominator
                counter += 1
        loss_feature /= counter
        loss_g /= len(y_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance
