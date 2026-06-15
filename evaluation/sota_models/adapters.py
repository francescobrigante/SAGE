# =============================================================================
# evaluation/sota_models/adapters.py
# Uniform codec adapters for SOTA baselines (music2latent, codicodec, SAME).
# Each adapter exposes the same interface so evaluate_sota.py can score any model
# with the identical metric pipeline (losses / CLAP / FAD) used for Swin & SAO.
# Model packages are imported lazily inside each adapter, so importing this
# module needs no model package installed (only the one being evaluated).
# =============================================================================
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


def _to_channel_time(rec: "np.ndarray | torch.Tensor", audio_channels: int) -> torch.Tensor:
    """Normalise a decoded waveform to a contiguous CPU float [C, T].

    Robust to the channel-axis convention (codec docstrings are unreliable —
    codicodec/music2latent both claim [T, C] but return [C, T]) and to numpy
    vs tensor outputs. Detects the channel axis as the one equal to
    audio_channels and orients it first.
    """
    if isinstance(rec, np.ndarray):
        rec = torch.from_numpy(rec)
    rec = rec.detach().cpu().float()
    if rec.dim() > 2:
        rec = rec.squeeze()
    if rec.dim() == 1:
        rec = rec.unsqueeze(0)                                              # [T] -> [1, T]
    if rec.shape[0] != audio_channels and rec.shape[-1] == audio_channels:
        rec = rec.transpose(0, 1)                                          # [T, C] -> [C, T]
    return rec.contiguous()


class CodecAdapter(ABC):
    """Uniform encode/reconstruct interface over a pretrained audio codec.

    Attributes set by every subclass:
        sample_rate:    native sample rate the model expects/returns (Hz).
        audio_channels: native channel count (1=mono, 2=stereo).
        device:         torch device the model runs on.
    """

    sample_rate: int
    audio_channels: int
    device: torch.device

    @abstractmethod
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode then decode a single clip.

        Args:
            wav: float waveform [C, T] at self.sample_rate, C == audio_channels.
        Returns:
            Reconstructed waveform [C, T'] on CPU, trimmed to <= input length.
        """
        raise NotImplementedError

    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the deterministic latent [D, T_lat] for MAEB probing.

        Optional — only needed for the semantic-latent track (M5). Default
        raises so reconstruction-only adapters (e.g. SAME) can skip it.
        """
        raise NotImplementedError(f"{type(self).__name__} has no encode_latent.")


# ===========================================================================
# Identity — pipeline self-test (M0). reconstruct() returns the input unchanged
# so the scorer must report ~perfect metrics (SI-SDR huge, STFT~0, CLAP~1, FAD~0).
# ===========================================================================
class IdentityAdapter(CodecAdapter):
    def __init__(self, device: str = "cpu", sample_rate: int = 44100,
                 audio_channels: int = 2, **_: Any):
        self.device = torch.device(device)          # unused, kept for interface parity
        self.sample_rate = sample_rate              # GT sample rate to load at
        self.audio_channels = audio_channels        # GT channel count to load at

    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        return wav.detach().cpu().float()                                   # [C, T]


# ===========================================================================
# CoDiCodec (Sony CSL) — consistency codec, stereo 44.1 kHz, ~x128, native 64-d
# continuous latent. API verified from codicodec/inference.py (the docstring's
# "[waveform_samples, audio_channels]" is wrong for the batched path — the code
# returns channels-first [C, T] via to_waveform(...).squeeze(0)):
#   encode([C,T], discrete=False, desired_channels=64) -> [C, 64, L]
#   decode(lat, mode='parallel')                        -> [C, T']
# ===========================================================================
class CodicodecAdapter(CodecAdapter):
    def __init__(self, device: str = "cuda", decode_mode: str = "parallel",
                 desired_channels: int = 64, **_: Any):
        from codicodec import EncoderDecoder

        self.device = torch.device(device)          # model + I/O device
        self.sample_rate = 44100                    # codicodec operates at 44.1 kHz
        self.audio_channels = 2                     # stereo model
        self.decode_mode = decode_mode              # 'parallel' (fast) | 'autoregressive'
        self.desired_channels = desired_channels    # continuous-latent channel fold (64 for MAEB)
        self.model = EncoderDecoder(device=self.device)

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        _, T = wav.shape                                                    # [C, T]
        lat = self.model.encode(                                            # [C, 64, L]
            wav.to(self.device), discrete=False,
            desired_channels=self.desired_channels,
        )
        rec = self.model.decode(lat, mode=self.decode_mode)                 # [C, T']
        rec = _to_channel_time(rec, self.audio_channels)                    # [C, T']
        n = min(T, rec.shape[-1])
        return rec[..., :n]                                                 # [C, T'<=T]

    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        # encode returns [C, L, 64] (channels, latent_time, FEATURES-LAST — the
        # docstring's [C, dim, length] is wrong, verified empirically).
        lat = self.model.encode(
            wav.to(self.device), discrete=False,
            desired_channels=self.desired_channels,
        )                                                                   # [C, L, 64]
        return lat.mean(0).transpose(0, 1).contiguous().cpu().float()       # [64, L]


# ===========================================================================
# music2latent (Sony CSL) — consistency codec, 44.1 kHz, x64, native 64-d latent.
# CHANNEL-INDEPENDENT: each audio channel is encoded as a separate batch element
# (latent [audio_channels, 64, L]); it reconstructs stereo but with no joint
# stereo modelling. Run stereo (audio_channels=2) — x64 holds per channel.
# API verified from music2latent/inference.py:
#   encode([C,T] tensor)  -> [C, 64, L]   (uses audio.shape[0] as channels)
#   decode(lat, denoising_steps=1) -> [C, T']  (consistency: 1 step)
# ===========================================================================
class Music2LatentAdapter(CodecAdapter):
    def __init__(self, device: str = "cuda", denoising_steps: int = 1, **_: Any):
        from music2latent import EncoderDecoder

        self.device = torch.device(device)          # model + I/O device
        self.sample_rate = 44100                    # music2latent operates at 44.1 kHz
        self.audio_channels = 2                     # stereo (channel-independent)
        self.denoising_steps = denoising_steps      # consistency decode steps (1 = default)
        self.model = EncoderDecoder(device=self.device)

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        _, T = wav.shape                                                    # [C, T]
        lat = self.model.encode(wav)                                        # [C, 64, L]
        rec = self.model.decode(lat, denoising_steps=self.denoising_steps)  # [C, T']
        rec = _to_channel_time(rec, self.audio_channels)                    # [C, T']
        n = min(T, rec.shape[-1])
        return rec[..., :n]                                                 # [C, T'<=T]

    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        lat = self.model.encode(wav)                                        # [C, 64, L]
        if isinstance(lat, np.ndarray):
            lat = torch.from_numpy(lat)
        return lat.float().mean(0).cpu()                                    # [64, L] (avg channels)


# ===========================================================================
# SAME (Stable Audio 3 autoencoder, Stability AI) — Soft-Norm bottleneck, stereo
# 44.1 kHz, ~x4096 temporal, 256-d latent. Reconstruction-only here (256-d latent
# is excluded from the 64-d MAEB probing). AE needs no flash_attn / transformers 5.
# API verified from stable_audio_3/model.py:
#   encode([C,T], sr) -> [B, 256, T_lat]   (auto resample/channel/pad)
#   decode(lat)       -> [B, C, samples]
# ===========================================================================
class SAMEAdapter(CodecAdapter):
    def __init__(self, device: str = "cuda", model_name: str = "same-l",
                 chunked: bool = False, **_: Any):
        from stable_audio_3 import AutoencoderModel

        self.device = torch.device(device)          # model + I/O device
        self.chunked = chunked                      # overlapping chunked enc/dec (memory saver)
        self.model = AutoencoderModel.from_pretrained(model_name, device=str(self.device))
        self.sample_rate = int(self.model.sample_rate)   # native SR from the model config
        self.audio_channels = 2                     # stereo model

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        _, T = wav.shape                                                    # [C, T]
        lat = self.model.encode(wav, self.sample_rate, chunked=self.chunked)  # [1, 256, T_lat]
        rec = self.model.decode(lat, chunked=self.chunked)                  # [1, C, T']
        rec = _to_channel_time(rec, self.audio_channels)                    # [C, T']
        n = min(T, rec.shape[-1])
        return rec[..., :n]                                                 # [C, T'<=T]


# ===========================================================================
# Stable Audio Open VAE (Stability AI) — the AutoencoderOobleck VAE used by the
# diffusers StableAudioPipeline. Per the SAO paper this autoencoder is "a variant
# of Stable Audio 2.0 trained on CC data" → it IS the SA2-family VAE (CC-retrained
# open weights), NOT the closed commercial SA2 checkpoint. Stereo 44.1 kHz,
# 64-d continuous latent (~2048:1, latent rate 21.5 Hz) → eligible for MAEB-64.
# Only the VAE is loaded (no T5 / no DiT). API verified from diffusers:
#   encode([B,C,T]) -> .latent_dist (OobleckDiagonalGaussian); .mode()/.sample() -> [B, 64, L]
#   decode(lat)     -> .sample -> [B, C, T']
# ===========================================================================
class StableAudioVAEAdapter(CodecAdapter):
    def __init__(self, device: str = "cuda", model_dir: str | None = None,
                 subfolder: str = "vae", **_: Any):
        from diffusers import AutoencoderOobleck

        # Prefer the local snapshot (downloaded via --local-dir to $FAST, works
        # offline on compute nodes); fall back to the gated HF repo id online.
        if model_dir is None:
            local = os.path.join(os.environ.get("FAST", ""),
                                 "models", "stable-audio-open-1.0")
            model_dir = local if os.path.isdir(os.path.join(local, subfolder)) \
                else "stabilityai/stable-audio-open-1.0"
        self.device = torch.device(device)          # model + I/O device
        self.model = (AutoencoderOobleck
                      .from_pretrained(model_dir, subfolder=subfolder)
                      .to(self.device).eval())
        # native config from the VAE itself (44100 Hz, stereo)
        self.sample_rate = int(getattr(self.model, "sampling_rate",
                                       self.model.config.sampling_rate))
        self.audio_channels = 2                     # stereo model

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        _, T = wav.shape                                                    # [C, T]
        x = wav.to(self.device).unsqueeze(0)                                # [1, C, T]
        lat = self.model.encode(x).latent_dist.mode()                       # [1, 64, L]
        rec = self.model.decode(lat).sample                                 # [1, C, T']
        rec = _to_channel_time(rec, self.audio_channels)                    # [C, T']
        n = min(T, rec.shape[-1])
        return rec[..., :n]                                                 # [C, T'<=T]

    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.to(self.device).unsqueeze(0)                                # [1, C, T]
        lat = self.model.encode(x).latent_dist.mode()                       # [1, 64, L]
        return lat[0].cpu().float()                                         # [64, L]


# ===========================================================================
# Factory
# ===========================================================================
_ADAPTERS = {
    "identity":     IdentityAdapter,
    "codicodec":    CodicodecAdapter,
    "music2latent": Music2LatentAdapter,
    "same":         SAMEAdapter,
    "sao-vae":      StableAudioVAEAdapter,
}


def build_adapter(name: str, device: str = "cuda", **kwargs: Any) -> CodecAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](device=device, **kwargs)
