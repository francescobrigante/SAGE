"""Initialization utilities for training and inference.

This module provides helper functions for:
- Custom collate function for STFT datasets
- Inference dataloader construction
- Chunk size resolution for streaming inference

All dataset/model instantiation now uses Hydra's `instantiate` API directly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader


def collate_stft(batch):
    """Default collate function for STFT datasets returning (S, wav[, meta]).

    Supports optional metadata (e.g., source paths) by forwarding them as a list
    without altering legacy behaviour when metadata is absent.
    """

    if not batch:
        raise ValueError("collate_stft received an empty batch")

    first = batch[0]
    if not isinstance(first, tuple):
        raise TypeError(f"collate_stft expects tuples, got {type(first).__name__}")

    if len(first) == 3:
        Ss, wavs, metas = zip(*batch)
    elif len(first) == 2:
        Ss, wavs = zip(*batch)
        metas = None
    else:
        raise ValueError(f"collate_stft expects 2 or 3 items per sample, got {len(first)}")

    first_spec = Ss[0]
    if first_spec is None:
        assert all(x is None for x in Ss), "Mixed spectrogram/None batches are not supported."
        w0 = wavs[0].shape
        assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
        stacked_specs = None
    else:
        s0 = first_spec.shape
        w0 = wavs[0].shape
        assert all(x.shape == s0 for x in Ss), f"STFT shapes differ: {[x.shape for x in Ss]}"
        assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
        stacked_specs = torch.stack(Ss, 0)

    stacked_wavs = torch.stack(wavs, 0)

    if metas is not None:
        return stacked_specs, stacked_wavs, list(metas)
    return stacked_specs, stacked_wavs


def build_inference_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    pin_memory_device: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader configured for inference workloads."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer for inference")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": collate_stft,
    }

    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor) if prefetch_factor is not None else 2
    else:
        loader_kwargs["persistent_workers"] = False

    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = str(pin_memory_device)

    return DataLoader(dataset, **loader_kwargs)


def resolve_chunk_sizes(
    *,
    sr: int,
    hop_length: int,
    samples_per_latent: int,
    frames_per_latent: int,
    segment_seconds: Optional[Any],
    segment_frames: Optional[Any],
    chunk_size_latent_cfg: Optional[Any],
    overlap_latent: int,
) -> Tuple[int, int, int]:
    """Resolve chunk sizing options returning (chunk_samples, overlap_samples, chunk_latent)."""

    options_selected = [segment_seconds is not None, segment_frames is not None, chunk_size_latent_cfg is not None]
    if sum(options_selected) > 1:
        raise ValueError("Specify only one among segment_seconds, segment_frames, chunk_size.")

    if segment_seconds is not None:
        segment_samples = int(round(float(segment_seconds) * sr))
        chunk_size_latent = max(1, int(round(segment_samples / max(1, samples_per_latent))))
    elif segment_frames is not None:
        segment_frames = int(segment_frames)
        chunk_size_latent = max(1, int(round(segment_frames / max(1, frames_per_latent))))
    elif chunk_size_latent_cfg is not None:
        chunk_size_latent = max(1, int(chunk_size_latent_cfg))
    else:
        chunk_size_latent = 128

    chunk_size_samples = chunk_size_latent * samples_per_latent
    overlap_samples = max(0, int(overlap_latent)) * samples_per_latent
    return chunk_size_samples, overlap_samples, chunk_size_latent

