# ===============
# Verification suite for stereo_imaging.py. Run BEFORE producing any numbers:
#   uv run pytest tests/test_stereo_imaging.py -q
# Covers: identity/gain invariance, symmetric penalty (collapse AND widening),
# pan sensitivity, delay estimation, misalignment degradation, band selectivity.
# ===============
import math

import pytest
import torch

from evaluation.metrics.stereo import align, band_weights, estimate_delay, stereo_imaging_distance

SR = 44100
T = SR * 3


def make_stereo(seed: int = 0, side_gain: float = 0.4) -> torch.Tensor:
    """Wide-ish stereo test signal: correlated mid + decorrelated side."""
    g = torch.Generator().manual_seed(seed)
    mid = torch.randn(T, generator=g)
    side = side_gain * torch.randn(T, generator=g)
    return torch.stack([mid + side, mid - side]) / math.sqrt(2)


def test_identity_is_zero():
    x = make_stereo()
    m = stereo_imaging_distance(x, x)
    assert m["d_width"] < 1e-6 and m["d_pan"] < 1e-6 and m["score"] < 1e-6


def test_gain_invariance():
    x = make_stereo()
    m = stereo_imaging_distance(x, 0.25 * x)
    assert m["d_width"] < 1e-6 and m["d_pan"] < 1e-6


def test_mono_collapse_penalized_with_negative_bias():
    x = make_stereo()
    mono = x.mean(0, keepdim=True).expand(2, -1).contiguous()
    m = stereo_imaging_distance(x, mono)
    assert m["d_width"] > 0.05, m
    assert m["width_bias"] < -0.05, "collapse must show as NEGATIVE width bias"
    assert m["sm_rec_db"] < m["sm_ref_db"] - 20


def test_over_widening_penalized_with_positive_bias():
    x = make_stereo()
    mid = x.mean(0)
    side = (x[0] - x[1]) / 2
    wide = torch.stack([mid + 3 * side, mid - 3 * side])
    m = stereo_imaging_distance(x, wide)
    assert m["d_width"] > 0.05, "over-widening must be penalized too"
    assert m["width_bias"] > 0.05, "widening must show as POSITIVE width bias"


def test_pan_error_detected_independently_of_width():
    x = make_stereo()
    swapped = x.flip(0)                       # L/R swap: width identical, pan flipped
    m = stereo_imaging_distance(x, swapped)
    assert m["d_width"] < 1e-6
    assert m["d_pan"] > 0.1


def test_estimate_delay_recovers_known_shift():
    x = make_stereo()
    for d in (0, 7, 501, -313):
        y = torch.roll(x, shifts=d, dims=-1)
        assert estimate_delay(x, y) == d


def test_misalignment_degrades_the_metric_and_align_fixes_it():
    x = make_stereo()
    shifted = torch.roll(x, shifts=997, dims=-1)
    degraded = stereo_imaging_distance(x, shifted)
    assert degraded["score"] > 0.02, "misalignment must visibly degrade the metric"
    ref_a, rec_a = align(x, shifted)
    fixed = stereo_imaging_distance(ref_a, rec_a)
    assert fixed["score"] < 1e-3
    assert fixed["score"] < degraded["score"] / 10


def test_align_handles_length_mismatch():
    x = make_stereo()
    rec = x[:, 250:-750]                      # shorter + delayed copy
    ref_a, rec_a = align(x, rec)
    assert ref_a.shape == rec_a.shape
    assert stereo_imaging_distance(ref_a, rec_a)["score"] < 1e-3


def test_align_robust_to_polarity_flip():
    x = make_stereo()
    rec = -torch.roll(x, shifts=421, dims=-1)     # inverted polarity + delay
    assert estimate_delay(x, rec) == 421
    ref_a, rec_a = align(x, rec)
    assert stereo_imaging_distance(ref_a, rec_a)["score"] < 1e-3  # ratios ignore polarity


def test_freq_weights_band_selectivity():
    """Collapse ONLY >4kHz; the low band must stay clean, the high band must flag it."""
    g = torch.Generator().manual_seed(1)
    n_fft, hop = 2048, 512
    lo_side = torch.randn(T, generator=g)
    hi_side = torch.randn(T, generator=g)
    # band-limit sides via FFT masking
    def bandpass(sig, lo, hi):
        X = torch.fft.rfft(sig)
        f = torch.linspace(0, SR / 2, X.numel())
        X[(f < lo) | (f >= hi)] = 0
        return torch.fft.irfft(X, T)
    side = 0.5 * (bandpass(lo_side, 100, 2000) + bandpass(hi_side, 4000, 12000))
    mid = torch.randn(T, generator=g)
    ref = torch.stack([mid + side, mid - side]) / math.sqrt(2)
    side_lo_only = 0.5 * bandpass(lo_side, 100, 2000)
    rec = torch.stack([mid + side_lo_only, mid - side_lo_only]) / math.sqrt(2)

    w_lo, w_hi = band_weights([(100, 2000), (4000, 12000)], SR, n_fft)
    m_lo = stereo_imaging_distance(ref, rec, freq_weights=w_lo, n_fft=n_fft, hop=hop)
    m_hi = stereo_imaging_distance(ref, rec, freq_weights=w_hi, n_fft=n_fft, hop=hop)
    assert m_hi["d_width"] > 5 * m_lo["d_width"], (m_lo["d_width"], m_hi["d_width"])
    assert m_hi["width_bias"] < -0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
