# =============================================================================
# tests/test_stereo_coherence_l6.py
# Unit tests for L6 (StereoCoherenceLoss) — STEREO_COLLAPSE_DIAGNOSIS §11.4.
#
# The two tests that justify the term's existence:
#   * it MOVES when only the Side phase changes, where mrstft_sd is provably
#     blind (4.8e-09) — that blindness is why every M/S arm restored the Side's
#     level and left d_pan pinned at the mono null;
#   * its gradient on the Mid is EXACTLY zero, so it cannot cost FAD/CLAP/CDPAM,
#     which measure the Mid alone.
# =============================================================================
from __future__ import annotations

import math

import pytest
import torch

from sage.nn.losses.experimental import StereoCoherenceLoss
from sage.nn.losses.signal import SumAndDifferenceSTFTLoss

SR, T = 44100, 16384
RES = dict(fft_sizes=[1024], hop_sizes=[256], win_lengths=[1024])


def _ms_to_lr(mid: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """(B, T) mid/side → (B, 2, T) left/right."""
    return torch.stack([(mid + side) / 2 ** 0.5, (mid - side) / 2 ** 0.5], dim=1)


def _signals(batch: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mid = torch.randn(batch, T, generator=g) * 0.3                     # (B, T)
    side = torch.randn(batch, T, generator=g) * 0.1                    # (B, T)
    return mid, side


def _phase_rotated_side(side: torch.Tensor, deg: float) -> torch.Tensor:
    """Same magnitude spectrum, phase rotated by `deg` — the blind spot probe."""
    n_fft, hop = 1024, 256
    win = torch.hann_window(n_fft)
    S = torch.stft(side, n_fft, hop, window=win, return_complex=True)
    S = S * torch.exp(torch.tensor(1j * math.pi * deg / 180.0))
    return torch.istft(S, n_fft, hop, window=win, length=side.shape[-1])


# ── the reason the term exists ──────────────────────────────────────────────

def test_l6_sees_a_side_phase_error_that_mrstft_sd_cannot():
    """Rotate only the Side's phase: L6 must react, the magnitude loss must not.

    Rotation is applied in the STFT domain and compared there, so the ISTFT
    inconsistency that would also perturb magnitudes never enters the comparison.
    """
    mid, side = _signals()
    target = _ms_to_lr(mid, side)
    pred = _ms_to_lr(mid, _phase_rotated_side(side, 90.0))

    l6 = StereoCoherenceLoss(**RES)
    mag_only = SumAndDifferenceSTFTLoss(sample_rate=SR, perceptual_weighting=False,
                                        w_lr=0.0, w_mid=0.0, w_side=1.0, **RES)

    l6_gap = l6(pred, target) - l6(target, target)
    mag_gap = mag_only(pred, target) - mag_only(target, target)

    assert l6_gap > 0.05, f"L6 barely reacted to a 90° Side rotation: {l6_gap:.2e}"
    assert l6_gap > mag_gap, (
        f"L6 ({l6_gap:.4f}) must be more phase-sensitive than magnitude ({mag_gap:.4f})")


def test_l6_mid_gradient_is_exactly_zero():
    """The safety property: FAD/CLAP/CDPAM see the Mid alone and cannot be moved."""
    mid, side = _signals()
    target = _ms_to_lr(mid, side)
    pred = (target * 0.9).detach().requires_grad_(True)

    g, = torch.autograd.grad(StereoCoherenceLoss(**RES)(pred, target), pred)
    g_mid = ((g[:, 0] + g[:, 1]) / math.sqrt(2)).norm().item()
    g_side = ((g[:, 0] - g[:, 1]) / math.sqrt(2)).norm().item()

    assert g_mid == 0.0, f"Mid gradient must be exactly zero, got {g_mid:e}"
    assert g_side > 0.0, "Side gradient must be non-zero or the term does nothing"


def test_l6_without_detach_does_touch_the_mid():
    """Guards the guard: without the detach the property is genuinely lost."""
    mid, side = _signals()
    target = _ms_to_lr(mid, side)
    pred = (target * 0.9).detach().requires_grad_(True)

    g, = torch.autograd.grad(StereoCoherenceLoss(**RES, detach_mid=False)(pred, target), pred)
    g_mid = ((g[:, 0] + g[:, 1]) / math.sqrt(2)).norm().item()
    assert g_mid > 0.0, "detach_mid=False should NOT be Mid-orthogonal"


# ── correctness of the quantity ─────────────────────────────────────────────

def test_l6_is_zero_on_a_perfect_reconstruction():
    mid, side = _signals()
    target = _ms_to_lr(mid, side)
    assert StereoCoherenceLoss(**RES)(target, target).item() < 1e-5


def test_l6_penalises_hallucinated_side_on_a_mono_reference():
    """The ms_replace failure mode: a mono reference with invented Side energy."""
    mid, _ = _signals()
    mono = _ms_to_lr(mid, torch.zeros_like(mid))                       # S ≡ 0
    _, fake_side = _signals(seed=7)
    hallucinated = _ms_to_lr(mid, fake_side)

    l6 = StereoCoherenceLoss(**RES)
    assert l6(hallucinated, mono) > l6(mono, mono) + 0.05


def test_l6_is_bounded_and_finite_on_extremes():
    """|γ| ≤ 1 by construction, so the term cannot blow up — silence included."""
    l6 = StereoCoherenceLoss(**RES)
    mid, side = _signals()
    for pred, target in [
        (torch.zeros(2, 2, T), _ms_to_lr(mid, side)),                  # silent prediction
        (_ms_to_lr(mid, side), torch.zeros(2, 2, T)),                  # silent reference
        (torch.zeros(2, 2, T), torch.zeros(2, 2, T)),                  # both silent
        (_ms_to_lr(mid, side) * 1e6, _ms_to_lr(mid, side)),            # huge gain
    ]:
        out = l6(pred, target)
        assert torch.isfinite(out), f"non-finite loss on {target.abs().max():.1e}"
        assert 0.0 <= out.item() <= 2.0 + 1e-6, f"|γ|≤1 violated: {out.item()}"


def test_l6_gate_skips_near_mono_items():
    mid, _ = _signals()
    mono = _ms_to_lr(mid, torch.zeros_like(mid))                       # S ≡ 0 → gated
    _, fake = _signals(seed=8)
    assert StereoCoherenceLoss(**RES, side_gate_db=-40.0)(_ms_to_lr(mid, fake), mono).item() == 0.0


def test_l6_gradient_is_finite_through_a_collapsed_side():
    """The regime the term must survive: prediction with essentially no Side."""
    mid, side = _signals()
    target = _ms_to_lr(mid, side)
    pred = _ms_to_lr(mid, side * 1e-8).detach().requires_grad_(True)
    g, = torch.autograd.grad(StereoCoherenceLoss(**RES)(pred, target), pred)
    assert torch.isfinite(g).all(), "non-finite gradient at the collapse limit"


def test_l6_tracks_d_pan_direction():
    """Sanity against the metric it is a surrogate for: worse d_pan ⇒ larger L6."""
    from stereo_diagnosis.ms_eval import ms_metrics_row

    mid, side = _signals(batch=1)
    target = _ms_to_lr(mid, side)
    l6 = StereoCoherenceLoss(**RES)

    prev_l6 = prev_pan = -1.0
    for deg in (10.0, 45.0, 120.0):
        pred = _ms_to_lr(mid, _phase_rotated_side(side, deg))
        cur_l6 = l6(pred, target).item()
        cur_pan = ms_metrics_row(target[0], pred[0], SR)["d_pan"]
        assert cur_l6 > prev_l6 and cur_pan > prev_pan, (
            f"non-monotone at {deg}°: L6 {prev_l6:.4f}→{cur_l6:.4f}, "
            f"d_pan {prev_pan:.4f}→{cur_pan:.4f}")
        prev_l6, prev_pan = cur_l6, cur_pan


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
