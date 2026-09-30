# =============================================================================
# Vocos *legacy* discriminators — the pre-DAC version pinned by WavTokenizer
# (jishengpeng/WavTokenizer decoder/discriminators.py). Ported verbatim:
#   - MultiPeriodDiscriminator / DiscriminatorP  (HiFi-GAN, waveform period view)
#   - MultiResolutionDiscriminator / DiscriminatorR  (UnivNet, **magnitude** STFT, no bands)
# The MRD here is phase-BLIND (`torch.stft(...).abs()`); this is exactly what makes the
# WavTokenizer-full stack distinct from DAC (whose MRD is complex+banded). Used only by the
# `wavtokenizer` composite discriminator for the as-paper ablation. Inputs are mono (B, T);
# the composite folds channels into the batch before calling these.
# =============================================================================

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (HiFi-GAN), optionally conditional via an embedding table."""

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11), num_embeddings: Optional[int] = None):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=p, num_embeddings=num_embeddings) for p in periods]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0))),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=1024)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.unsqueeze(1)                                   # (B, T) -> (B, 1, T)
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:                             # pad to a multiple of the period
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)      # (B, 1, T//p, p)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)                                       # fmap[-1] = logit map
        x += h
        x = torch.flatten(x, 1, -1)                          # (B, N) logits
        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    """Multi-Resolution Discriminator (UnivNet) over **magnitude** spectrograms (phase-blind)."""

    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int], ...] = ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=r, num_embeddings=num_embeddings) for r in resolutions]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int],
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: Optional[int] = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution                         # (n_fft, hop_length, win_length)
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x)                              # (B, F, T_frames) magnitude
        x = x.unsqueeze(1)                                   # (B, 1, F, T_frames)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)                                       # fmap[-1] = logit map
        x += h
        x = torch.flatten(x, 1, -1)                          # (B, N) logits
        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,                                     # rectangular window (as in the original)
            center=True,
            return_complex=True,
        ).abs()                                              # magnitude only → phase-blind
        return magnitude_spectrogram
