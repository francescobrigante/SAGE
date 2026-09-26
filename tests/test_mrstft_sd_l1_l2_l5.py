# =============================================================================
# tests/test_mrstft_sd_l1_l2_l5.py
# Unit tests for the three stereo-loss interventions of STEREO_COLLAPSE_DIAGNOSIS
# §11.4: L1 (separable Mid/Side weights), L2 (near-mono skip gate) and L5
# (guarded spectral-convergence denominator).
#
# The load-bearing test is L1's: a Side-only configuration must put EXACTLY zero
# gradient on the Mid. That is what makes the intervention safe to graft onto a
# half-trained checkpoint — FAD/CLAP/CDPAM measure the Mid alone, so a term that
# cannot touch the Mid cannot cost them anything.
# =============================================================================
from __future__ import annotations

import math

import pytest
import torch

from ar_spectra.training.losses.signal import (
    SpectralConvergenceLoss,
    SumAndDifferenceSTFTLoss,
)

SR = 44100
T = 16384                                   # ~0.37 s: enough for the 2048 window
FFT = dict(fft_sizes=[2048, 512], hop_sizes=[512, 128], win_lengths=[2048, 512])


def _loss(**kw) -> SumAndDifferenceSTFTLoss:
    """A/B-comparable instance: A-weighting off so tests stay fast and exact."""
    return SumAndDifferenceSTFTLoss(sample_rate=SR, perceptual_weighting=False,
                                    **FFT, **kw)


def _stereo(batch: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mid = torch.randn(batch, 1, T, generator=g) * 0.3                  # (B, 1, T)
    side = torch.randn(batch, 1, T, generator=g) * 0.1                 # (B, 1, T)
    return torch.cat([mid + side, mid - side], dim=1)                  # (B, 2, T)


def _mid_side_grad(loss_val: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    """Split ∂loss/∂pred into its Mid and Side projections."""
    g, = torch.autograd.grad(loss_val, pred, retain_graph=True)        # (B, 2, T)
    g_mid = (g[:, 0] + g[:, 1]) / math.sqrt(2)                         # (B, T)
    g_side = (g[:, 0] - g[:, 1]) / math.sqrt(2)                        # (B, T)
    return float(g_mid.norm()), float(g_side.norm())


# ── L1 — separable Mid/Side weights ─────────────────────────────────────────

def test_l1_side_only_has_exactly_zero_mid_gradient():
    """The whole point of L1: a Side-only term cannot move the Mid.

    The gradient on L and R is antisymmetric, so it cancels bit-for-bit in the
    M = L + R projection. Not "small" — zero.
    """
    target = _stereo()
    pred = (target * 0.9).detach().requires_grad_(True)
    side_only = _loss(w_lr=0.0, w_mid=0.0, w_side=1.0)

    g_mid, g_side = _mid_side_grad(side_only(pred, target), pred)

    assert g_mid == 0.0, f"Mid gradient must be exactly zero, got {g_mid:e}"
    assert g_side > 0.0, "Side gradient must be non-zero or the term does nothing"


def test_l1_defaults_reproduce_sao_exactly():
    """Leaving w_mid/w_side unset must not change a single bit vs the old code."""
    target, pred = _stereo(), _stereo(seed=1)
    legacy = _loss(w_ms=1.0, w_lr=1.0)                       # w_mid/w_side = None
    explicit = _loss(w_ms=1.0, w_lr=1.0, w_mid=0.5, w_side=0.5)
    assert torch.allclose(legacy(pred, target), explicit(pred, target), atol=0, rtol=0)


def test_l1_mid_only_has_zero_side_gradient():
    """The dual of the first test — confirms the split is clean in both directions."""
    target = _stereo()
    pred = (target * 0.9).detach().requires_grad_(True)
    mid_only = _loss(w_lr=0.0, w_mid=1.0, w_side=0.0)

    g_mid, g_side = _mid_side_grad(mid_only(pred, target), pred)

    assert g_side == 0.0, f"Side gradient must be exactly zero, got {g_side:e}"
    assert g_mid > 0.0


def test_l1_zero_weights_give_zero_loss_and_no_nan():
    target, pred = _stereo(), _stereo(seed=2)
    out = _loss(w_lr=0.0, w_mid=0.0, w_side=0.0)(pred, target)
    assert out.item() == 0.0 and torch.isfinite(out)


# ── L2 — near-mono skip gate ────────────────────────────────────────────────

def _mono_batch(batch: int = 4) -> torch.Tensor:
    """Mono duplicated onto two channels — `utils/audio.py:110` `wav.repeat(2,1)`."""
    g = torch.Generator().manual_seed(3)
    mid = torch.randn(batch, 1, T, generator=g) * 0.3                  # (B, 1, T)
    return torch.cat([mid, mid], dim=1)                                # (B, 2, T), S ≡ 0


def test_l2_gate_zeroes_the_side_term_on_exactly_mono_targets():
    target = _mono_batch()                                  # S ≡ 0 → S/M = −inf dB
    pred = _stereo(seed=4)
    gated = _loss(w_lr=0.0, w_mid=0.0, w_side=1.0, side_gate_db=-40.0)
    assert gated(pred, target).item() == 0.0


def test_l2_no_gate_by_default_and_mono_target_is_finite():
    """Default (gate off) must stay finite on a mono target — that is L5's job."""
    target, pred = _mono_batch(), _stereo(seed=5)
    out = _loss(w_lr=0.0, w_mid=0.0, w_side=1.0)(pred, target)
    assert torch.isfinite(out), "un-gated mono target produced a non-finite loss"


def test_l2_gate_fires_exactly_at_the_threshold():
    """Two items straddling −40 dB: the quiet one is dropped, the loud one kept."""
    g = torch.Generator().manual_seed(6)
    mid = torch.randn(2, 1, T, generator=g) * 0.3                      # (2, 1, T)
    # Side at −30 dB (kept) and −50 dB (dropped) of the Mid, in POWER.
    gains = torch.tensor([10 ** (-30 / 20), 10 ** (-50 / 20)]).view(2, 1, 1)
    side = torch.randn(2, 1, T, generator=g)
    side = side / side.norm(dim=-1, keepdim=True) * mid.norm(dim=-1, keepdim=True) * gains
    target = torch.cat([mid + side, mid - side], dim=1)                # (2, 2, T)
    pred = _stereo(batch=2, seed=7)

    gated = _loss(w_lr=0.0, w_mid=0.0, w_side=1.0, side_gate_db=-40.0)
    kept_only = gated(pred[:1], target[:1])                # the −30 dB item alone
    both = gated(pred, target)

    assert torch.isclose(both, kept_only, rtol=1e-5), (
        f"gate should leave only the −30 dB item: {both.item()} vs {kept_only.item()}")


def test_l2_gate_keeps_gradients_flowing_for_kept_items():
    target = _mono_batch(batch=2)
    target = torch.cat([target, _stereo(batch=2, seed=8)], dim=0)      # 2 mono + 2 stereo
    pred = _stereo(batch=4, seed=9).detach().requires_grad_(True)
    out = _loss(w_lr=0.0, w_mid=0.0, w_side=1.0, side_gate_db=-40.0)(pred, target)
    g, = torch.autograd.grad(out, pred)
    assert torch.isfinite(g).all() and g.norm() > 0


# ── L5 — guarded spectral-convergence denominator ───────────────────────────

def test_l5_finite_when_the_prediction_vanishes():
    """SAO's reversed order puts the PREDICTION in the denominator."""
    x = torch.randn(2, 64, 32).abs() + 0.1                             # target mag
    y = torch.zeros_like(x)                                            # collapsed pred
    out = SpectralConvergenceLoss()(x, y)
    assert torch.isfinite(out).all(), "guard failed: collapsed prediction gave inf/NaN"


def test_l5_unchanged_on_non_degenerate_inputs():
    """The guard must be inert in normal training — same value as the raw ratio."""
    x = torch.randn(2, 64, 32).abs() + 0.1
    y = torch.randn(2, 64, 32).abs() + 0.1
    guarded = SpectralConvergenceLoss()(x, y)
    raw = (torch.norm(y - x, p="fro", dim=[-1, -2])
           / torch.norm(y, p="fro", dim=[-1, -2])).unsqueeze(-1).unsqueeze(-1)
    assert torch.allclose(guarded, raw, rtol=1e-6)


def test_l5_gradient_is_finite_through_a_collapsing_prediction():
    x = torch.randn(2, 64, 32).abs() + 0.1
    y = (torch.zeros_like(x) + 1e-30).requires_grad_(True)
    g, = torch.autograd.grad(SpectralConvergenceLoss()(x, y).sum(), y)
    assert torch.isfinite(g).all(), "non-finite gradient at the collapse limit"


def test_l5_still_punishes_shrink_harder_than_growth():
    """The asymmetry is the anti-collapse property — the guard must not flatten it."""
    x = torch.randn(2, 64, 32).abs() + 0.1
    sc = SpectralConvergenceLoss()
    shrink = sc(x, 0.25 * x).mean()          # prediction 4× too quiet
    grow = sc(x, 4.00 * x).mean()            # prediction 4× too loud
    assert shrink > grow, f"shrink {shrink:.3f} should cost more than growth {grow:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
