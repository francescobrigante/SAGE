# ===============
# Stereo metrics added for the stereo-discriminator study: CCPC (earVAE), IPD/ILD errors below
# 1.5 kHz (earVAE2 App. J.1) and side_db. Golden values come from the reference implementation
# validated on the controlled-degradation battery (local/stereo_metric_probe/paper_stereo_metrics.py).
# ===============
import math

import pytest
import torch

from evaluation.metrics.stereo import MS_COLUMNS, ccpc, ipd_ild_low, ms_metrics_row

SR = 44100


def _signal() -> torch.Tensor:
    """Deterministic (2, T) stereo noise with a correlated Mid and a partly independent Side."""
    g = torch.Generator().manual_seed(0)
    a, b, c = (torch.randn(2 * SR, generator=g) for _ in range(3))
    return torch.stack([a + 0.4 * b, a - 0.4 * b + 0.1 * c])


def _delay_r(x: torch.Tensor, n: int) -> torch.Tensor:
    return torch.stack([x[0], torch.cat([torch.zeros(n), x[1, :-n]])])


@pytest.fixture(scope="module")
def ref():
    return _signal()


def test_identity_is_perfect(ref):
    assert ccpc(ref, ref) == pytest.approx(1.0, abs=1e-6)
    ipd, ild = ipd_ild_low(ref, ref, SR)
    assert ipd == pytest.approx(0.0, abs=1e-6) and ild == pytest.approx(0.0, abs=1e-6)


def test_golden_values_of_the_reference_implementation(ref):
    rec = torch.stack([0.9 * ref[0], _delay_r(ref, 11)[1]])
    assert ccpc(ref, rec) == pytest.approx(0.0738530196249485, rel=1e-4)
    ipd, ild = ipd_ild_low(ref, rec, SR)
    assert ipd == pytest.approx(1.16395103931427, rel=1e-4)
    assert ild == pytest.approx(0.9140424728393555, rel=1e-4)


def test_polarity_inversion_is_seen_by_ipd_but_not_by_ccpc(ref):
    """Documented blind spot: a constant inter-channel phase offset keeps CCPC at 1."""
    rec = torch.stack([ref[0], -ref[1]])
    assert ccpc(ref, rec) == pytest.approx(1.0, abs=1e-5)
    ipd, ild = ipd_ild_low(ref, rec, SR)
    assert ipd == pytest.approx(math.pi, abs=1e-3) and ild == pytest.approx(0.0, abs=1e-4)


def test_inter_channel_delay_lowers_ccpc(ref):
    assert ccpc(ref, _delay_r(ref, 44)) < 0.9


def test_global_gain_changes_nothing(ref):
    assert ccpc(ref, 0.5 * ref) == pytest.approx(1.0, abs=1e-5)
    ipd, ild = ipd_ild_low(ref, 0.5 * ref, SR)
    assert ipd == pytest.approx(0.0, abs=1e-5) and ild == pytest.approx(0.0, abs=1e-4)


def test_balance_of_3_db_is_an_ild_error_of_3_db(ref):
    g = 10 ** (3 / 40)
    _, ild = ipd_ild_low(ref, torch.stack([ref[0] * g, ref[1] / g]), SR)
    assert ild == pytest.approx(3.0, abs=1e-3)


def test_errors_above_the_low_band_are_ignored(ref):
    """A phase rotation of R above 5 kHz must not move IPD/ILD, which only read [0, 1.5 kHz)."""
    spec = torch.fft.rfft(ref[1])
    spec[torch.fft.rfftfreq(ref.shape[-1], 1 / SR) > 5000] *= 1j
    rec = torch.stack([ref[0], torch.fft.irfft(spec, n=ref.shape[-1])])
    ipd, ild = ipd_ild_low(ref, rec, SR)
    assert ipd < 1e-2 and ild < 1e-2


def test_row_has_the_new_columns_after_the_legacy_ones(ref):
    legacy = ["file", "width_bias", "d_width", "sisdr_s", "sisdr_m", "d_pan", "pan_bias", "score",
              "sm_ref_db", "sm_rec_db", "ref_mono"]
    assert MS_COLUMNS[:len(legacy)] == legacy
    assert MS_COLUMNS[len(legacy):] == ["side_db", "ccpc", "ipd_lo", "ild_lo"]
    row = ms_metrics_row(ref, 0.8 * ref + 0.01 * _signal().flip(0), SR)
    assert set(MS_COLUMNS[1:]) <= set(row)
    assert row["side_db"] == pytest.approx(row["sm_rec_db"] - row["sm_ref_db"])


def test_row_on_a_mono_reference_is_finite(ref):
    mono = torch.stack([ref[0], ref[0]])
    row = ms_metrics_row(mono, ref, SR)
    assert row["ref_mono"] == 1
    assert all(math.isfinite(row[k]) for k in ("ccpc", "ipd_lo", "ild_lo"))
