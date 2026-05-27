
# =============================================================================
# Discriminator pretransforms: ComplexSTFTPretransform (simplified, encode-only),
# PatchedPretransform (fold), WaveletPretransform (biorthogonal DWT via pywt),
# ChromaPretransform (chroma spectrogram per octave centre).
# Ported from stable-audio-tools — stripped of EMA, compander, Griffin-Lim.
# =============================================================================

import math
from typing import Literal, Optional

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Shared channel helpers (also defined in blocks/transformer_sat.py)
# ---------------------------------------------------------------------------

def fold_channels_into_batch(x: torch.Tensor) -> torch.Tensor:
    """[B, C, ...] → [B*C, ...]."""
    return rearrange(x, 'b c ... -> (b c) ...')


def unfold_channels_from_batch(x: torch.Tensor, channels: int) -> torch.Tensor:
    """[B*C, ...] → [B, C, ...]."""
    if channels == 1:
        return x.unsqueeze(1)
    return rearrange(x, '(b c) ... -> b c ...', c=channels)


# ---------------------------------------------------------------------------
# Complex-STFT packing helpers
# ---------------------------------------------------------------------------

def _pack_complex_to_channels(U: torch.Tensor) -> torch.Tensor:
    """Pack complex U[B, C, F, M] → real Z[B, 2·C·F, M] — Re/Im interleaved."""
    B, C, F, M = U.shape
    Z = torch.empty((B, 2 * C * F, M), dtype=U.real.dtype, device=U.device)
    Z[:, 0::2, :] = U.real.reshape(B, C * F, M)
    Z[:, 1::2, :] = U.imag.reshape(B, C * F, M)
    return Z


def _demod_sign(F_bins: int, M_frames: int, device, dtype, expand_bc: bool = True) -> torch.Tensor:
    """Parity demodulation sign for hop = n_fft // 2."""
    k_odd = (torch.arange(F_bins, device=device, dtype=torch.int8) & 1).view(F_bins, 1)
    m_odd = (torch.arange(M_frames, device=device, dtype=torch.int8) & 1).view(1, M_frames)
    parity = (k_odd & m_odd).to(torch.float32)
    sign = (1.0 - 2.0 * parity).to(dtype)
    return sign.view(1, 1, F_bins, M_frames) if expand_bc else sign


def _sine_window(n: int, device, dtype) -> torch.Tensor:
    """Tight (WOLA) sine window: w[k] = sin(π(k + 0.5)/N)."""
    k = torch.arange(n, device=device, dtype=dtype)
    return torch.sin(math.pi * (k + 0.5) / n)


# ---------------------------------------------------------------------------
# ComplexSTFTPretransform (encode-only, simplified for discriminators)
# ---------------------------------------------------------------------------

class ComplexSTFTPretransform(nn.Module):
    """
    Waveform [B, C, T] → packed complex STFT channels [B, 2·C·F, M].

    Attributes:
        encoded_channels: number of output channels (= C * 2 * (n_fft//2 + 1))
        downsampling_ratio: hop_length = n_fft // 2
    """

    def __init__(
        self,
        channels: int,
        n_fft: int = 1024,
        demodulate: bool = True,
        center: bool = False,
        ema_flatten: bool = False,  # accepted but ignored in this simplified version
        use_compander: bool = False,  # accepted but ignored
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.win_length = n_fft
        self.hop_length = n_fft // 2
        self.demodulate = demodulate
        self.center = center
        self.F = n_fft // 2 + 1
        self.downsampling_ratio = self.hop_length
        self.encoded_channels = channels * 2 * self.F

        win = _sine_window(n_fft, torch.device('cpu'), torch.float32)
        self.register_buffer('window', win, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] waveform
        Returns:
            [B, 2·C·F, M] packed real/imag channels
        """
        B, C, T = x.shape
        if x.dtype == torch.bfloat16:
            x = x.float()
        x_flat = fold_channels_into_batch(x)                                # (B*C, T)
        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x_flat.device, x_flat.dtype),
            center=self.center,
            normalized=True,
            return_complex=True,
        )                                                                    # (B*C, F, M)
        X = unfold_channels_from_batch(X, C)                               # (B, C, F, M)
        if self.demodulate:
            sign = _demod_sign(X.shape[2], X.shape[3], X.device, X.real.dtype)
            X = X * sign
        return _pack_complex_to_channels(X)                                 # (B, 2*C*F, M)


# ---------------------------------------------------------------------------
# PatchedPretransform (fold patches into channels)
# ---------------------------------------------------------------------------

class PatchedPretransform(nn.Module):
    """
    Fold length-`patch_size` segments into channels.
    [B, C, T] → [B, C·patch_size, T//patch_size].

    Attributes:
        encoded_channels: C * patch_size
        downsampling_ratio: patch_size
    """

    def __init__(self, channels: int, patch_size: int, **kwargs):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.downsampling_ratio = patch_size
        self.encoded_channels = channels * patch_size

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Right-pad T to a multiple of patch_size."""
        pad_len = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C·h, T//h] where h = patch_size."""
        x = self._pad(x)
        return rearrange(x, 'b c (l h) -> b (c h) l', h=self.patch_size)   # (B, C*h, T//h)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C·h, L] → [B, C, L·h]."""
        return rearrange(x, 'b (c h) l -> b c (l h)', h=self.patch_size)   # (B, C, L*h)


# ---------------------------------------------------------------------------
# Wavelet DWT 1D (analysis + synthesis)
# ---------------------------------------------------------------------------

def _get_filter_bank(wavelet: str) -> torch.Tensor:
    """Load biorthogonal filter bank from pywt; trim leading zero column."""
    filt = torch.tensor(pywt.Wavelet(wavelet).filter_bank)
    if wavelet.startswith('bior') and torch.all(filt[:, 0] == 0):
        filt = filt[:, 1:]
    return filt


class WaveletEncode1d(nn.Module):
    """Multi-level 1D DWT analysis (low+high subbands packed into channels)."""

    def __init__(
        self,
        channels: int,
        levels: int,
        wavelet: Literal['bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior4.4', 'bior6.8'] = 'bior4.4',
    ):
        super().__init__()
        self.channels = channels
        self.levels = levels
        filt = _get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1, 'filter length must be odd (after trim)'
        kernel = filt[:2, None]
        kernel = torch.flip(kernel, dims=(-1,))  # correlation → convolution
        idx_i = torch.repeat_interleave(torch.arange(2), channels)
        idx_j = torch.tile(torch.arange(channels), (2,))
        kf = torch.zeros(channels * 2, channels, filt.shape[-1])
        kf[idx_i * channels + idx_j, idx_j] = kernel[idx_i, 0]
        self.register_buffer('kernel', kf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C·2^levels, T//2^levels]."""
        for _ in range(self.levels):
            low = x[:, :self.channels]                                       # (B, C, T)
            rest = x[:, self.channels:]                                      # (B, C*(k-1), T)
            pad = self.kernel.shape[-1] // 2
            low = F.pad(low, (pad, pad), 'reflect')
            low = F.conv1d(low, self.kernel, stride=2)                      # (B, 2C, T//2)
            rest = rearrange(rest, 'n (c c2) (l l2) -> n (c l2 c2) l', l2=2, c2=self.channels)
            x = torch.cat([low, rest], dim=1)
        return x


class WaveletDecode1d(nn.Module):
    """Multi-level 1D DWT synthesis."""

    def __init__(
        self,
        channels: int,
        levels: int,
        wavelet: Literal['bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior4.4', 'bior6.8'] = 'bior4.4',
    ):
        super().__init__()
        self.channels = channels
        self.levels = levels
        filt = _get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1
        kernel = filt[2:, None]
        idx_i = torch.repeat_interleave(torch.arange(2), channels)
        idx_j = torch.tile(torch.arange(channels), (2,))
        kf = torch.zeros(channels * 2, channels, filt.shape[-1])
        kf[idx_i * channels + idx_j, idx_j] = kernel[idx_i, 0]
        self.register_buffer('kernel', kf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C·2^levels, T//2^levels] → [B, C, T]."""
        for _ in range(self.levels):
            low = x[:, :self.channels * 2]
            rest = x[:, self.channels * 2:]
            pad = self.kernel.shape[-1] // 2 + 2
            low = rearrange(low, 'n (l2 c) l -> n c (l l2)', l2=2)
            low = F.pad(low, (pad, pad), 'reflect')
            low = rearrange(low, 'n c (l l2) -> n (l2 c) l', l2=2)
            low = F.conv_transpose1d(low, self.kernel, stride=2,
                                     padding=self.kernel.shape[-1] // 2)
            low = low[..., pad - 1:-pad]
            rest = rearrange(rest, 'n (c l2 c2) l -> n (c c2) (l l2)', l2=2, c2=self.channels)
            x = torch.cat([low, rest], dim=1)
        return x


# ---------------------------------------------------------------------------
# WaveletPretransform
# ---------------------------------------------------------------------------

class WaveletPretransform(nn.Module):
    """
    Biorthogonal DWT pretransform.
    [B, C, T] → [B, C·2^levels, T//2^levels].

    Attributes:
        encoded_channels: C * 2^levels
        downsampling_ratio: 2^levels
    """

    def __init__(
        self,
        channels: int,
        levels: int,
        wavelet: str = 'bior4.4',
        **kwargs,
    ):
        super().__init__()
        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)
        self.downsampling_ratio = 2 ** levels
        self.encoded_channels = channels * self.downsampling_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ---------------------------------------------------------------------------
# ChromaPretransform (chroma spectrogram, one octave centre per instance)
# ---------------------------------------------------------------------------

class ChromaPretransform(nn.Module):
    """
    Waveform [B, C, T] → chroma spectrogram [B, C*n_chroma, M].

    One pretransform per octave centre (ctroct/octwidth parameters), so that
    TransformerDiscriminator can receive a different chroma view at each scale.

    Attributes:
        encoded_channels: n_chroma * in_channels
        downsampling_ratio: hop_length (= n_fft // 2 by default)
    """

    def __init__(
        self,
        channels: int,
        n_chroma: int = 64,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        ctroct: float = 5.0,        # centre octave (like librosa chroma_cqt)
        octwidth: float = 1.5,      # octave width for the Gaussian weighting
        normalized: bool = True,
        norm: int = 1,
        **kwargs,
    ):
        super().__init__()
        from torchaudio.prototype.transforms import ChromaSpectrogram
        self.channels = channels                    # number of audio channels (1=mono, 2=stereo)
        self.n_chroma = n_chroma                    # chroma bins (pitch classes per octave)
        self.n_fft = n_fft
        self.hop_length = n_fft // 2               # matches SAT default
        self.chroma = ChromaSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_chroma=n_chroma,
            ctroct=ctroct,
            octwidth=octwidth,
            normalized=normalized,
            norm=norm,
        )
        self.downsampling_ratio = self.hop_length   # frames per sample
        self.encoded_channels = n_chroma * channels # output channels for d_model heuristic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.reshape(B * C, T)            # (B*C, T)   — process each channel independently
        x = self.chroma(x)                 # (B*C, n_chroma, M)
        return x.reshape(B, C * self.n_chroma, -1)  # (B, C*n_chroma, M)
