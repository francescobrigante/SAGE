# =============================================================================
# tests/test_maeb_audio_prep.py
# Unit tests for evaluation/maeb/audio_prep.py — channel normalization (the bug
# the SAO monolith missed), shape heuristics, resampling, and frame math.
# =============================================================================
import sys
from pathlib import Path

import pytest
import torch

from maeb.audio_prep import prepare_audio, valid_latent_frames  # noqa: E402

_SR = 44100


def _item(array, sr=_SR):
    return {"array": array, "sampling_rate": sr}


def test_mono_upmixed_to_stereo():
    # 1-channel input must become exactly 2 channels (the channel bug fix).
    out = prepare_audio(_item(torch.zeros(1, 1000)), _SR, channels=2)
    assert out.shape == (2, 1000)


def test_flat_mono_array_upmixed_to_stereo():
    # 1-D array is treated as mono and upmixed to the requested channel count.
    out = prepare_audio(_item(torch.zeros(1000)), _SR, channels=2)
    assert out.shape == (2, 1000)


def test_stereo_downmixed_to_mono():
    out = prepare_audio(_item(torch.zeros(2, 1000)), _SR, channels=1)
    assert out.shape == (1, 1000)


def test_transposed_stereo_is_corrected():
    # (samples, channels) with samples >> channels must be transposed to (C, T).
    out = prepare_audio(_item(torch.zeros(1000, 2)), _SR, channels=2)
    assert out.shape == (2, 1000)


def test_resample_changes_length():
    out = prepare_audio(_item(torch.zeros(1, 22050), sr=22050), _SR, channels=2)
    assert out.shape[0] == 2
    assert out.shape[-1] == pytest.approx(44100, rel=0.01)


def test_max_seconds_clips_length():
    out = prepare_audio(_item(torch.zeros(2, _SR * 40)), _SR, channels=2, max_seconds=30.0)
    assert out.shape[-1] == _SR * 30


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        prepare_audio(_item(torch.zeros(2, 3, 4)), _SR, channels=2)


def test_output_is_float32_and_contiguous():
    out = prepare_audio(_item(torch.zeros(1, 500)), _SR, channels=2)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


@pytest.mark.parametrize("n,ds,expected", [
    (1, 2048, 1),        # below one frame -> at least 1
    (2048, 2048, 1),     # exactly one frame
    (2049, 2048, 2),     # one sample over -> 2 frames (ceil)
    (4096, 2048, 2),
    (0, 2048, 1),        # empty -> clamped to 1
])
def test_valid_latent_frames(n, ds, expected):
    assert valid_latent_frames(n, ds) == expected
