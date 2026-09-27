# ===============================================================
# test_compute_spectral.py — Tests of the per-file reconstruction metrics of
# the paper evaluator (evaluation/metrics/signal.py: SDR/SI-SDR, multi-resolution
# STFT and mel L1) and of the helpers around them (evaluation/common.py:
# write_csv, atomic_save_npy). Synthetic signals, no checkpoints.
# ===============================================================
import csv

import numpy as np
import pytest
import torch

from evaluation.common import atomic_save_npy, write_csv
from evaluation.metrics.signal import compute_sdr_and_sisdr, si_sdr, spectral_losses, stft_loss

SR = 44100


# ── helpers ──────────────────────────────────────────────────────────────────

def _sine_wav(duration: float = 1.0, freq: float = 440.0, sr: int = SR) -> torch.Tensor:
    """Generate a clean mono sine wave → [1, T]."""
    t = torch.linspace(0, duration, int(sr * duration))
    return (torch.sin(2 * torch.pi * freq * t) * 0.5).unsqueeze(0)


def _compute_pair(target: torch.Tensor, pred: torch.Tensor):
    """Min-trim → metrics on a single [C, T] pair, as the evaluator does. Returns (stft, sisdr)."""
    T = min(target.shape[-1], pred.shape[-1])
    stft = spectral_losses(target[..., :T], pred[..., :T], sample_rate=SR)["stft_loss"]
    _, sisdr = compute_sdr_and_sisdr(target[..., :T], pred[..., :T])
    return stft, sisdr


# ── T1: identical signals ─────────────────────────────────────────────────────

def test_identical_signals_high_sisdr():
    """SI-SDR should be very high (> 30 dB) when pred == target."""
    wav = _sine_wav()
    _, sisdr = _compute_pair(wav, wav.clone())
    assert sisdr > 30.0, f"Identical signals should have SI-SDR > 30 dB, got {sisdr:.2f}"


def test_identical_signals_near_zero_stft():
    """STFT loss should be near zero when pred == target (up to fp precision)."""
    wav = _sine_wav()
    stft, _ = _compute_pair(wav, wav.clone())
    assert stft < 0.05, f"Identical signals should have STFT loss < 0.05, got {stft:.4f}"


# ── T2: corrupted signal degrades metrics ─────────────────────────────────────

def test_corrupted_signal_lower_sisdr():
    wav = _sine_wav()
    _, sisdr_clean = _compute_pair(wav, wav.clone())
    torch.manual_seed(42)
    _, sisdr_noisy = _compute_pair(wav, wav + torch.randn_like(wav) * 0.3)
    assert sisdr_noisy < sisdr_clean


def test_corrupted_signal_higher_stft():
    wav = _sine_wav()
    stft_clean, _ = _compute_pair(wav, wav.clone())
    torch.manual_seed(42)
    stft_noisy, _ = _compute_pair(wav, wav + torch.randn_like(wav) * 0.3)
    assert stft_noisy > stft_clean


# ── T3: the fused metrics agree with the single-metric functions ──────────────

def test_fused_metrics_match_reference_functions():
    torch.manual_seed(0)
    target = 0.3 * torch.randn(2, SR)
    pred = target + 0.1 * torch.randn(2, SR)
    _, sisdr = compute_sdr_and_sisdr(target, pred)
    assert sisdr == pytest.approx(si_sdr(target, pred), abs=1e-3)
    assert spectral_losses(target, pred)["stft_loss"] == pytest.approx(stft_loss(target, pred), rel=1e-6)


def test_sisdr_is_scale_invariant_sdr_is_not():
    torch.manual_seed(1)
    target = 0.3 * torch.randn(2, SR)
    pred = target + 0.05 * torch.randn(2, SR)
    sdr1, sisdr1 = compute_sdr_and_sisdr(target, pred)
    sdr2, sisdr2 = compute_sdr_and_sisdr(target, 0.5 * pred)
    assert sisdr2 == pytest.approx(sisdr1, abs=1e-3)
    assert sdr2 < sdr1 - 1.0


# ── T4: end-to-end per-file CSV (as written by the evaluators) ────────────────

def test_per_file_csv_correct_columns_and_ordering(tmp_path):
    torch.manual_seed(0)
    pairs = {}
    for stem in ("clean_001", "clean_002"):
        w = _sine_wav(freq=440.0)
        pairs[stem] = (w, w.clone())
    for stem in ("noisy_001", "noisy_002"):
        w = _sine_wav(freq=660.0)
        pairs[stem] = (w, w + torch.randn_like(w) * 0.5)

    rows = []
    for stem, (t, p) in pairs.items():
        stft, sisdr = _compute_pair(t, p)
        rows.append({"file": stem, "stft_loss": stft, "si_sdr": sisdr})
    path = tmp_path / "metrics" / "spectral.csv"
    write_csv(path, ["file", "stft_loss", "si_sdr"], rows)

    back = {r["file"]: r for r in csv.DictReader(open(path))}
    assert len(back) == 4 and set(next(iter(back.values()))) == {"file", "stft_loss", "si_sdr"}
    for stem in ("clean_001", "clean_002"):
        assert float(back[stem]["si_sdr"]) > 30.0 and float(back[stem]["stft_loss"]) < 0.05
    for stem in ("noisy_001", "noisy_002"):
        assert float(back[stem]["si_sdr"]) < 20.0


# ── T6: min-trim handles length mismatch ──────────────────────────────────────

def test_min_trim_one_sample_mismatch():
    """Pred 1 sample shorter (MP3 off-by-one): min-trim should still yield near-perfect metrics."""
    wav = _sine_wav(duration=1.5)
    stft, sisdr = _compute_pair(wav, wav[..., :-1])
    assert sisdr > 30.0 and stft < 0.05


def test_min_trim_large_mismatch():
    """Pred ~1024 samples shorter (codec hop rounding): metrics still indicate high quality."""
    wav = _sine_wav(duration=2.0)
    stft, sisdr = _compute_pair(wav, wav[..., :-1024])
    assert sisdr > 20.0 and stft < 0.5


# ── T7: atomic_save_npy data integrity ───────────────────────────────────────

def test_atomic_save_npy_roundtrip(tmp_path):
    data = np.random.default_rng(42).random((512,)).astype(np.float32)
    save_path = tmp_path / "emb.npy"
    atomic_save_npy(save_path, data)
    np.testing.assert_array_equal(np.load(save_path), data)


def test_atomic_save_npy_no_leftover_tmp(tmp_path):
    save_path = tmp_path / "sub" / "emb.npy"
    atomic_save_npy(save_path, np.zeros((64,), dtype=np.float32))
    assert [p.name for p in save_path.parent.iterdir()] == ["emb.npy"]


def test_atomic_save_npy_overwrites_existing(tmp_path):
    save_path = tmp_path / "emb.npy"
    atomic_save_npy(save_path, np.ones((32,), dtype=np.float32))
    atomic_save_npy(save_path, np.zeros((32,), dtype=np.float32))
    np.testing.assert_array_equal(np.load(save_path), np.zeros((32,), dtype=np.float32))
