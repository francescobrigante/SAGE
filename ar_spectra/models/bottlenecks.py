import numpy as np
import random 
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError
    
def _complex_to_channel_view(t: torch.Tensor) -> torch.Tensor:
    return torch.cat((t.real, t.imag), dim=1)


def _channel_view_to_complex(t: torch.Tensor) -> torch.Tensor:
    real, imag = t.chunk(2, dim=1)
    return torch.complex(real, imag)


def vae_sample(mean, scale):
    was_complex = torch.is_complex(mean)

    if was_complex:
        mean = _complex_to_channel_view(mean)
        scale = _complex_to_channel_view(scale)

    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    if was_complex:
        latents = _channel_view_to_complex(latents)

    return latents, kl
    
    
class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        assert x.shape[1] % 2 == 0, "VAEBottleneck expects even channels [mu|scale] along dim=1"
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl
        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

# Skip/passthrough bottleneck (JSON-controlled)
class SkipBottleneck(Bottleneck):
    """
    Passthrough bottleneck. 
    """
    def __init__(self, target_channels: int | None = None):
        super().__init__(is_discrete=False)
        self.target_channels = target_channels

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        if self.target_channels is not None:
            cx = x.shape[1]

            if self.target_channels % cx != 0:
                raise AssertionError(
                    f"SkipBottleneck: encoder channels={cx} must be uquals or multiple of "
                    f"decoder target={self.target_channels} "
                    f"to bypass VAE."
                )
        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x

