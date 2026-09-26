# ===============================================================
# test_batch_align.py — Unit tests for batch_align cross-correlation.
# Verifies lag detection, waveform alignment, and edge cases with
# synthetic signals — no audio files or model weights required.
# ===============================================================
import sys
import pytest
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from utils import batch_align          # evaluation/utils.py

SR = 44100


def _make_shifted(T: int = 22050, lag: int = 0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target [1,1,T], pred [1,1,T]) where pred = target shifted by +lag samples."""
    torch.manual_seed(seed)
    t = torch.randn(1, 1, T)
    if lag == 0:
        return t, t.clone()
    p = torch.zeros(1, 1, T)
    p[0, 0, lag:] = t[0, 0, :T - lag]
    return t, p


# ── T1: zero lag — signals are already aligned ───────────────────────────────

def test_zero_lag_detected():
    t, p = _make_shifted(lag=0)
    _, _, lags = batch_align(t, p, sr=SR)
    assert lags[0].item() == 0, f"Expected lag=0 for identical signals, got {lags[0].item()}"


def test_zero_lag_output_matches_input():
    t, p = _make_shifted(lag=0)
    t_al, p_al, _ = batch_align(t, p, sr=SR)
    assert torch.allclose(t_al, t), "Aligned target should equal original for zero lag"
    assert torch.allclose(p_al, p), "Aligned pred should equal original for zero lag"


# ── T2: known positive lag (pred is delayed) ─────────────────────────────────

@pytest.mark.parametrize("lag", [50, 200, 1000])
def test_positive_lag_detected(lag):
    t, p = _make_shifted(lag=lag)
    _, _, lags = batch_align(t, p, sr=SR)
    assert lags[0].item() == lag, f"Expected lag={lag}, got {lags[0].item()}"


def test_positive_lag_alignment_correct():
    lag = 200
    T = 22050
    t, p = _make_shifted(T=T, lag=lag)
    t_al, p_al, _ = batch_align(t, p, sr=SR)
    cur_len = T - lag
    max_diff = (t_al[0, 0, :cur_len] - p_al[0, 0, :cur_len]).abs().max().item()
    assert max_diff < 1e-5, f"Aligned signals should match in valid region, max diff={max_diff:.2e}"


def test_positive_lag_tail_is_zero():
    lag = 200
    T = 22050
    t, p = _make_shifted(T=T, lag=lag)
    t_al, p_al, _ = batch_align(t, p, sr=SR)
    cur_len = T - lag
    assert t_al[0, 0, cur_len:].abs().max().item() == 0.0, "Tail of aligned target should be zero"
    assert p_al[0, 0, cur_len:].abs().max().item() == 0.0, "Tail of aligned pred should be zero"


# ── T3: shared zero tail does not corrupt SI-SDR ─────────────────────────────

def test_aligned_sisdr_matches_unaligned_identical():
    """After alignment, SI-SDR of (nearly) identical signals should be very high."""
    from torchmetrics.audio.sdr import SignalDistortionRatio as SISDRMetric
    lag = 100
    T = 22050
    t, p = _make_shifted(T=T, lag=lag)
    t_al, p_al, _ = batch_align(t, p, sr=SR)
    sisdr = SISDRMetric()(p_al.squeeze(1), t_al.squeeze(1)).item()
    assert sisdr > 30.0, f"After alignment, identical signals should have SI-SDR>30 dB, got {sisdr:.1f}"


# ── T4: batch dimension — multiple signals in one call ────────────────────────

def test_batch_of_two_signals():
    """batch_align must handle B=2 independently."""
    T = 22050
    lags_expected = [50, 300]
    targets, preds = [], []
    for lag in lags_expected:
        t, p = _make_shifted(T=T, lag=lag, seed=lag)
        targets.append(t[0])   # [1, T]
        preds.append(p[0])     # [1, T]

    t_batch = torch.stack(targets)   # [2, 1, T]
    p_batch = torch.stack(preds)     # [2, 1, T]

    _, _, lags = batch_align(t_batch, p_batch, sr=SR)
    for i, expected_lag in enumerate(lags_expected):
        assert lags[i].item() == expected_lag, (
            f"Batch item {i}: expected lag={expected_lag}, got {lags[i].item()}"
        )
