#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/audio_prep.py
# Convert a raw MAEB audio item into a model-ready waveform tensor.
# Fixes the SAO channel bug: forces the model's channel count (mono<->stereo).
# =============================================================================
from __future__ import annotations

import io
import math
from typing import Any

import torch
import torchaudio


class AudioDecodeError(Exception):
    """Raised when an audio item cannot be decoded (corrupt / unreadable file)."""


def _decode_item(audio_item: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """Return (waveform, sample_rate) from a MAEB audio item.

    Supports both HuggingFace ``Audio`` decode modes:
      * decoded   → item has "array" (+ "sampling_rate");
      * undecoded → item has "path"/"bytes" (decode=False); decoded here with
        torchaudio so a corrupt file raises a catchable error instead of
        crashing the DataLoader worker.

    Raises:
        AudioDecodeError: if the item carries no usable audio or decoding fails.
    """
    array = audio_item.get("array")
    if array is not None:
        return torch.as_tensor(array, dtype=torch.float32), int(audio_item["sampling_rate"])

    # decode=False path: decode from raw bytes or file path (robust).
    src = audio_item.get("bytes") or audio_item.get("path")
    if src is None:
        raise AudioDecodeError("audio item has neither 'array' nor 'path'/'bytes'")
    try:
        handle = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
        wav, sr = torchaudio.load(handle)
    except Exception as e:  # corrupt / unreadable / transient I/O
        raise AudioDecodeError(f"failed to decode {audio_item.get('path')!r}: {e}") from e
    return wav.to(torch.float32), int(sr)


def prepare_audio(
    audio_item: dict[str, Any],
    target_sr: int,
    channels: int,
    max_seconds: float | None = None,
) -> torch.Tensor:
    """Convert one MAEB audio item to a (channels, samples) float32 tensor.

    Handles: array -> tensor, (samples, channels) -> (channels, samples) transpose
    heuristic, mono unsqueeze, resample to ``target_sr``, channel normalization to
    exactly ``channels`` (the bug the monolith missed), and optional length clip.

    Args:
        audio_item:  MAEB dict — decoded ("array"+"sampling_rate") or undecoded
                     ("path"/"bytes", HF Audio decode=False).
        target_sr:   Model sample rate (Hz).
        channels:    Channels the model expects (e.g. 2 for stereo SAO-ACE).
        max_seconds: If set, clip to this many seconds.

    Returns:
        Contiguous float32 tensor of shape (channels, samples).

    Raises:
        AudioDecodeError: if the clip cannot be decoded (caller should skip it).
    """
    array, sr = _decode_item(audio_item)

    # (samples, channels) -> (channels, samples) for transposed stereo arrays
    if array.dim() == 2 and array.shape[0] > array.shape[1] and array.shape[1] <= 8:
        array = array.transpose(0, 1)
    if array.dim() == 1:
        array = array.unsqueeze(0)
    elif array.dim() != 2:
        raise ValueError(f"Unsupported audio shape {tuple(array.shape)}")

    if sr != target_sr:
        array = torchaudio.functional.resample(array, sr, target_sr)

    # Force exactly `channels` channels: upmix mono->stereo by repeat, downmix by slice.
    cur = array.shape[0]
    if cur < channels:
        array = array.repeat((channels + cur - 1) // cur, 1)[:channels]
    elif cur > channels:
        array = array[:channels]

    if max_seconds is not None:
        max_samples = int(max_seconds * target_sr)
        if array.shape[-1] > max_samples:
            array = array[..., :max_samples]

    return array.contiguous()


def valid_latent_frames(num_samples: int, downsample: int) -> int:
    """Number of non-padding latent frames for a clip of ``num_samples`` samples.

    A latent frame spans ``downsample`` waveform samples; clamped to at least 1.
    """
    return max(1, math.ceil(num_samples / downsample))
