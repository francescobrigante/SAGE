
# =============================================================================
# EnCodec family discriminators (e.g., multi-resolution).
# =============================================================================

from .types import IndividualDiscriminatorOut
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as tp
from typing import List, Tuple
from functools import reduce
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class EncodecDiscriminator(nn.Module):
    def __init__(self, normalize_losses=False, loss_type: tp.Literal["hinge", "rpgan"]="hinge", *args, **kwargs):
        super().__init__()
        from .encodec import MultiScaleSTFTDiscriminator
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)
        self.normalize_losses = normalize_losses
        self.fm_reduction = (lambda x, y: abs(x - y).mean()/(abs(x).mean() + 1e-3)) if normalize_losses else (lambda x, y: abs(x - y).mean())
        self.loss_type = loss_type

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        dis_loss = torch.tensor(0., device=reals.device)
        adv_loss = torch.tensor(0., device=reals.device)

        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)

        # Compute per-scale losses
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            if self.loss_type == "hinge":
                _dis, _adv = get_hinge_losses(logits_true[i], logits_fake[i])
            else:  # rpgan
                print("NameError, implementation yet to be imported from stable_audio in ar-spectra")
                #_dis, _adv = get_relativistic_losses(logits_true[i], logits_fake[i])

            dis_loss = dis_loss + _dis 
            adv_loss = adv_loss + _adv

        num_scales = len(logits_true)

        return dis_loss / num_scales, adv_loss / num_scales, feature_matching_distance / num_scales

class SharedDiscriminatorConvNet(nn.Module):

    def __init__(
        self,
        in_size: int,
        convolution: tp.Union[nn.Conv1d, nn.Conv2d],
        out_size: int = 1,
        capacity: int = 32,
        n_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 4,
        activation: tp.Callable[[], nn.Module] = lambda: nn.SiLU(),
        normalization: tp.Callable[[nn.Module], nn.Module] = torch.nn.utils.parametrizations.weight_norm,
    ) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
                s = stride[i]
            else:
                pad = kernel_size[0] // 2
                s = (stride[i], 1)

            net.append(
                normalization(
                    convolution(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    )))
            net.append(activation())

        net.append(convolution(channels[-1], out_size, 1))

        self.net = nn.ModuleList(net)

    def forward(self, x) -> IndividualDiscriminatorOut:
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        score = x.reshape(x.shape[0], -1).mean(-1)
        return score, features
