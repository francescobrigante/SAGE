# =============================================================================
# Structural bottlenecks for Autoencoders, supporting continuous (VAE) and passthrough (Skip) configurations.
# =============================================================================
import torch
from torch import nn

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
    def __init__(self, parameters_to_predict: int = 2):
        super().__init__(is_discrete=False)
        self.parameters_to_predict = parameters_to_predict  # num encoder slots per latent channel

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

            if cx % self.target_channels != 0:
                raise AssertionError(
                    f"SkipBottleneck: encoder channels={cx} must be equal or multiple of "
                    f"decoder target={self.target_channels} "
                    f"to bypass VAE."
                )
        if return_info:
            return x, info
        return x

    def decode(self, x):
        return x

class SoftNormBottleneck(Bottleneck):
    def __init__(self, dim=32, noise_augment_dim=0, noise_regularize=False, auto_scale=False, freeze=False, **kwargs):
        super().__init__(is_discrete=False)

        self.noise_augment_dim = noise_augment_dim
        self.scaling_factor = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1, noise_augment_dim, 1))
        self.noise_regularize = noise_regularize
        self.freeze = freeze
        if self.freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            running_std = torch.ones(1)
            self.register_parameter("running_std", nn.Parameter(running_std, requires_grad=False))

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        x = x * self.scaling_factor + self.bias

        if self.training and hasattr(self, "running_std") and not self.freeze:
            # Update running std
            self.running_std.data = (self.running_std.data * 0.999 + x.std().detach() * 0.001).clamp(min=1e-4)
        
        if hasattr(self, "running_std"):
            x = x / self.running_std

        if self.training and return_info:
            var = (x.std(dim=-1) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-1)
            loss = (mean * mean + var - logvar - 1).mean()
            var = (x.std(dim=-2) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-2)
            loss = loss + 0.4 * (mean * mean + var - logvar - 1).mean()
            info["softnorm_loss"] = loss 
        
        if return_info:
            return x, info
        
        return x

    def decode(self, x, **kwargs):
        if hasattr(self, "running_std"):
            x = x * self.running_std

        if self.noise_regularize:
            if hasattr(self, "running_std"):
                scaling = self.running_std
            else:
                scaling = x.std(dim=-1).unsqueeze(-1)
            if self.training:
                scale = 5e-2
            else:
                scale = 1e-3
            noise = torch.randn_like(x) * scaling * scale
            x = x + noise

        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x