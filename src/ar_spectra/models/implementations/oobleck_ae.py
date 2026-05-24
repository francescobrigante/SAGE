# =============================================================================
# Standalone Oobleck encoder/decoder for Stable Audio Open checkpoint loading.
# Extracted from stable_audio_baseline/stable_audio_tools/models/autoencoders.py
# without the k_diffusion / torchdiffeq dependency chain.
# =============================================================================

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from alias_free_torch import Activation1d


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


def _get_activation(activation: str, antialias: bool = False, channels: int = None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    if antialias:
        act = Activation1d(act)
    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels, out_channels, kernel_size=7, dilation=dilation, padding=padding),
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=9, use_snake=use_snake),
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False,
                 antialias_activation=False, use_nearest_upsample=False):
        super().__init__()
        if use_nearest_upsample:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=1, bias=False, padding="same"),
            )
        else:
            upsample = WNConvTranspose1d(in_channels, out_channels,
                                         kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        self.layers = nn.Sequential(
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample,
            ResidualUnit(out_channels, out_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=128, latent_dim=32,
                 c_mults=None, strides=None, use_snake=False, antialias_activation=False):
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        c_mults = [1] + c_mults
        layers = [WNConv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels,
                                       stride=strides[i], use_snake=use_snake))
        layers += [
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=128, latent_dim=32,
                 c_mults=None, strides=None, use_snake=False, antialias_activation=False,
                 use_nearest_upsample=False, final_tanh=True):
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        c_mults = [1] + c_mults
        layers = [WNConv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels,
                                       stride=strides[i - 1], use_snake=use_snake,
                                       antialias_activation=antialias_activation,
                                       use_nearest_upsample=use_nearest_upsample))
        layers += [
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SAOAutoencoder(nn.Module):
    """Minimal wrapper around OobleckEncoder + OobleckDecoder for SAO checkpoint inference."""

    def __init__(self, encoder: OobleckEncoder, decoder: OobleckDecoder, downsampling_ratio: int = 2048):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.downsampling_ratio = downsampling_ratio  # used for input padding in _encode_sao


def build_sao_autoencoder(model_config: dict) -> SAOAutoencoder:
    """Construct SAOAutoencoder from a SAO-format model_config dict."""
    m = model_config["model"]
    enc_cfg = m["encoder"]["config"]
    dec_cfg = m["decoder"]["config"]

    encoder = OobleckEncoder(
        in_channels=enc_cfg.get("in_channels", 2),
        channels=enc_cfg.get("channels", 128),
        latent_dim=enc_cfg.get("latent_dim", 128),
        c_mults=enc_cfg.get("c_mults"),
        strides=enc_cfg.get("strides"),
        use_snake=enc_cfg.get("use_snake", False),
        antialias_activation=enc_cfg.get("antialias_activation", False),
    )
    decoder = OobleckDecoder(
        out_channels=dec_cfg.get("out_channels", 2),
        channels=dec_cfg.get("channels", 128),
        latent_dim=dec_cfg.get("latent_dim", 64),
        c_mults=dec_cfg.get("c_mults"),
        strides=dec_cfg.get("strides"),
        use_snake=dec_cfg.get("use_snake", False),
        antialias_activation=dec_cfg.get("antialias_activation", False),
        use_nearest_upsample=dec_cfg.get("use_nearest_upsample", False),
        final_tanh=dec_cfg.get("final_tanh", True),
    )
    downsampling_ratio = m.get("downsampling_ratio", 2048)
    return SAOAutoencoder(encoder, decoder, downsampling_ratio=downsampling_ratio)
