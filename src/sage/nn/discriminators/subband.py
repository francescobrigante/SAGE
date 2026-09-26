
# =============================================================================
# Sub-band discrimination for analyzing metrics over specific ranges (e.g., CQT).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint
from torchaudio.transforms import Resample
from einops import rearrange

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

class CQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict, hop_length: int, n_octaves:int, bins_per_octave: int):
        super().__init__()
        self.cfg = cfg

        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = cfg["sampling_rate"]
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.fmin = cfg["cqtd_fmin"]

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            fmin=cfg["cqtd_fmin"],
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(nn.Conv2d(
                self.in_channels * 2,
                self.in_channels * 2,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            ))

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(
            self.in_channels * 2,
            self.filters,
            kernel_size=self.kernel_size,
            padding=self.get_2d_padding(self.kernel_size),
        ))

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(self.max_filters,
                (self.filters_scale ** (i + 1)) * self.filters)
            self.convs.append(weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=(dilation, 1),
                padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
            )))
            in_chs = out_chs

        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(weight_norm(nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        )))

        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!")

    def get_2d_padding(self,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        input_channels = x.shape[1]

        if input_channels >= 2:
            x = rearrange(x, 'b c ... -> (b c) ...')

        z = self.cqt_transform(x)
      
        if input_channels >= 2:
            z = rearrange(z, '(b c) ... -> b c ...', c = input_channels)

        z_amplitude = z[..., 0]
        z_phase = z[..., 1]

        if len(z_amplitude.shape) == 3:
            z_amplitude = z_amplitude.unsqueeze(1)
            z_phase = z_phase.unsqueeze(1)
        
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            s = i * self.bins_per_octave
            e = (i + 1) * self.bins_per_octave
            latent_z.append(self.conv_pres[i](z[..., s:e]))
        latent_z = torch.cat(latent_z, dim=-1)

        fmap = []
        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)
        return latent_z, fmap
class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        self.use_checkpoint = cfg.get("use_checkpoint", False)
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
            if self.use_checkpoint:
                y_d_r, fmap_r = checkpoint(disc, reals, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(disc, gens, use_reentrant=False)
            else:
                y_d_r, fmap_r = disc(reals)
                y_d_g, fmap_g = disc(gens)
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
