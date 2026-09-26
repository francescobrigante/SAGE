# ===============================================================
# test_partial_read.py
#
#   Correctness of the windowed partial-read path in
#   OnTheFlySTFTDataset: decoding only a ~segment-long slice of a
#   long file yields a valid, contiguous, non-silent crop of the
#   right shape; it stays deterministic per (epoch, index), varies
#   across epochs, resamples correctly, and falls back to full-load
#   on short files. (Throughput is measured separately on real
#   long files — see scripts/bench_partial_read.)
# ===============================================================
import sys
from pathlib import Path

import pytest
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import dataloader as dl

SR = 44100
SEGMENT = 127 * 512                 # segment_samples for target_frames=128, hop=512
N_FILES = 6


def _save_ramp(path: Path, length: int, sr: int = SR):
    """Write a stereo ramp wav (float32). Sample t has value ~t/length, so a clean
    contiguous read is strictly increasing with near-constant step."""
    ramp = torch.arange(length, dtype=torch.float32).unsqueeze(0).repeat(2, 1) / length
    torchaudio.save(str(path), ramp, sr, encoding="PCM_F", bits_per_sample=32)


def _make_dataset(audio_dir: Path, partial_read: bool, seed: int = 123):
    return dl.OnTheFlySTFTDataset(
        audio_dir=str(audio_dir),
        sample_rate=SR,
        n_fft=2048, hop_length=512, win_length=2048,
        target_frames=128,
        extensions=[".wav"],
        stereo=True, cac=True,
        skip_failed_samples=False,
        partial_read=partial_read,
        seed=seed,
    )


def _is_exact_ramp_slice(seg: torch.Tensor, length: int) -> bool:
    """Exact correctness: in a ramp, the sample value IS its position. Reconstruct the
    expected contiguous slice from the first sample and assert the whole crop matches —
    proves the seek landed on real contiguous audio (no garbage, no discontinuity).
    Only valid when SR matches (no resampling)."""
    start = round(seg[0, 0].item() * length)
    expected = torch.arange(start, start + seg.shape[-1], dtype=torch.float32) / length
    return torch.allclose(seg[0], expected, atol=1e-5) and torch.equal(seg[0], seg[1])


@pytest.fixture
def long_corpus(tmp_path):
    d = tmp_path / "long"; d.mkdir()
    for i in range(N_FILES):
        _save_ramp(d / f"{i:03d}.wav", length=SR * 8)   # 8 s ≫ window
    return d


# ── T1: partial read returns a valid, non-silent crop of correct shape ────────
def test_partial_shape_and_nonsilent(long_corpus):
    ds = _make_dataset(long_corpus, partial_read=True)
    S, seg = ds[0]
    assert seg.shape == (2, SEGMENT)
    assert seg.abs().max() > 0
    assert S.shape[0] == 4  # cac stereo → 4 channels


# ── T2: the decoded slice is genuine contiguous audio (not seek garbage) ──────
def test_partial_is_clean_contiguous_audio(long_corpus):
    ds = _make_dataset(long_corpus, partial_read=True)
    for i in range(N_FILES):
        _, seg = ds[i]
        assert _is_exact_ramp_slice(seg, SR * 8), f"file {i}: partial crop is not a contiguous slice"


# ── T3: deterministic per (epoch, index); varies across epochs ────────────────
def test_partial_determinism_and_epoch_variation(long_corpus):
    ds = _make_dataset(long_corpus, partial_read=True)
    a = ds[0][1].clone()
    b = ds[0][1].clone()
    assert torch.equal(a, b)                # same epoch, same index → identical
    ds.set_epoch(1)
    c = ds[0][1].clone()
    assert not torch.equal(a, c)            # advancing epoch moves the window


# ── T4: short file (< window) falls back to full load and still works ─────────
def test_partial_fallback_on_short_file(tmp_path):
    d = tmp_path / "short"; d.mkdir()
    # length > segment but < window (segment + margin) → forces the fallback branch
    _save_ramp(d / "000.wav", length=SEGMENT + 1000)
    ds = _make_dataset(d, partial_read=True)
    S, seg = ds[0]
    assert seg.shape == (2, SEGMENT)
    assert seg.abs().max() > 0


# ── T5: partial read resamples a non-target SR file to the right length ───────
def test_partial_resample(tmp_path):
    d = tmp_path / "sr"; d.mkdir()
    _save_ramp(d / "000.wav", length=22050 * 8, sr=22050)   # 8 s @ 22.05 kHz
    ds = _make_dataset(d, partial_read=True)
    S, seg = ds[0]
    assert seg.shape == (2, SEGMENT)
    assert seg.abs().max() > 0


# ── T6: partial and full both yield exact contiguous slices of the same files ─
def test_partial_matches_full(long_corpus):
    ds_p = _make_dataset(long_corpus, partial_read=True, seed=7)
    ds_f = _make_dataset(long_corpus, partial_read=False, seed=7)
    for i in range(N_FILES):
        _, sp = ds_p[i]
        _, sf = ds_f[i]
        assert _is_exact_ramp_slice(sp, SR * 8)   # partial read → valid contiguous slice
        assert _is_exact_ramp_slice(sf, SR * 8)   # full read    → valid contiguous slice
