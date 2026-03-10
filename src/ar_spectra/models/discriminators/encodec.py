
# =============================================================================
# EnCodec family discriminators (e.g., multi-resolution).
# =============================================================================

from .types import IndividualDiscriminatorOut, get_hinge_losses
import torch
import torch.nn as nn
import numpy as np
import typing as tp
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import torchaudio

from ar_spectra.blocks.conv.normed import NormConv2d, get_2d_padding

class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator."""
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 1), normalized: bool = True,
                 activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2},
                 spec_scale_pow=0.0, **kwargs):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window_fn=torch.hann_window, normalized=self.normalized,
            center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size,
                       padding=get_2d_padding(kernel_size), norm='weight_norm')
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1),
                                         padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm='weight_norm'))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs,
                                     kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm='weight_norm'))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm='weight_norm')
        self.spec_scale_pow = spec_scale_pow

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)
        if self.spec_scale_pow != 0.0:
            z = z * torch.pow(z.abs() + 1e-6, self.spec_scale_pow)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = checkpoint(layer, z)
            z = self.activation(z)
            fmap.append(z)
        z = checkpoint(self.conv_post, z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator."""
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512],
                 hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class EncodecDiscriminator(nn.Module):
    def __init__(self, normalize_losses=False, loss_type: tp.Literal["hinge", "rpgan"]="hinge", *args, **kwargs):
        super().__init__()
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
