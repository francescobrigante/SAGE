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

    # Drop None samples silently (e.g., if dataset signals unusable item)
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("collate_stft received only invalid/None samples")

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