# ===============
# SAO-faithful stereo M/S reconstruction-loss tests (STEREO_COLLAPSE fix).
# Validates SumAndDifferenceSTFTLoss: numerical parity vs Stable Audio Open's
# vendored auraloss, the crucial "not a no-op" side-channel sensitivity that a
# complex-STFT MSE lacks, mono/near-silent safety, and loss_manager wiring.
# ===============
import importlib.util
import os

import pytest
import torch

from sage.nn.losses.signal import SumAndDifferenceSTFTLoss

# 6 SAO resolutions used by the mrstft_sd config (n_fft=32 omitted).
_FFT = [2048, 1024, 512, 256, 128, 64]
_HOP = [512, 256, 128, 64, 32, 16]
_WIN = [2048, 1024, 512, 256, 128, 64]

# Signals long enough for the largest window; stereo (B, 2, T).
_B, _T = 2, 8192


def _sd_loss(w_ms=1.0, w_lr=1.0, perceptual_weighting=False):
    return SumAndDifferenceSTFTLoss(
        fft_sizes=_FFT, hop_sizes=_HOP, win_lengths=_WIN, sample_rate=44100,
        perceptual_weighting=perceptual_weighting, w_ms=w_ms, w_lr=w_lr,
    )


# Stable Audio Open's vendored auraloss.py (stable-audio-tools, training/losses/auraloss.py).
# Not a dependency: point SAO_AURALOSS_PY at a checkout to run the parity test.
_SAO_AURALOSS = os.environ.get("SAO_AURALOSS_PY", "")


def _load_sao_auraloss():
    """Import Stable Audio Open's vendored auraloss.py standalone (no package init)."""
    path = _SAO_AURALOSS
    spec = importlib.util.spec_from_file_location("sao_auraloss", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_ms_branch_vs_sao_auraloss():
    """Our mid/side branch (w_ms=1, w_lr=0) must equal SAO's SumAndDifferenceSTFTLoss."""
    if not os.path.isfile(_SAO_AURALOSS):
        pytest.skip("set SAO_AURALOSS_PY to stable-audio-tools' auraloss.py for the parity check")
    sao = _load_sao_auraloss()

    torch.manual_seed(0)
    pred = torch.randn(_B, 2, _T)
    target = torch.randn(_B, 2, _T)

    # perceptual_weighting=False → pure SC + log-mag: bit-comparable across the two ports.
    ours = _sd_loss(w_ms=1.0, w_lr=0.0, perceptual_weighting=False)
    theirs = sao.SumAndDifferenceSTFTLoss(
        fft_sizes=_FFT, hop_sizes=_HOP, win_lengths=_WIN, perceptual_weighting=False
    )
    out_ours = ours(pred, target)
    out_theirs = theirs(target, pred)   # SAO AuralossLoss reversed-order chain
    assert torch.allclose(out_ours, out_theirs, atol=1e-5), (out_ours, out_theirs)


def test_not_a_no_op_side_collapse_penalized():
    """The property complex-MSE lacks: collapsing the side channel must cost MORE
    than preserving it. Build a target with real stereo width; compare a
    side-preserving prediction against a side-collapsed (mono) one."""
    torch.manual_seed(1)
    mid = torch.randn(_B, 1, _T)
    side = 0.3 * torch.randn(_B, 1, _T)                       # genuine stereo width
    left = mid + side
    right = mid - side
    target = torch.cat([left, right], dim=1)                 # (B, 2, T)

    # Prediction A: same mid, side halved (partial collapse toward mono).
    side_shrunk = 0.5 * side
    pred_keep = torch.cat([mid + side_shrunk, mid - side_shrunk], dim=1)
    # Prediction B: side fully collapsed → mono (both channels = mid).
    pred_mono = torch.cat([mid, mid], dim=1)

    loss = _sd_loss(w_ms=1.0, w_lr=1.0, perceptual_weighting=False)
    l_keep = loss(pred_keep, target)
    l_mono = loss(pred_mono, target)
    assert l_mono > l_keep > 0.0, (l_mono, l_keep)


def test_zero_on_identical():
    loss = _sd_loss(perceptual_weighting=False)
    x = torch.randn(_B, 2, _T)
    out = loss(x, x.clone())
    assert out.ndim == 0 and torch.isfinite(out)
    assert torch.allclose(out, torch.zeros(()), atol=1e-5)


def test_mono_input_is_finite():
    """L==R → side S≡0 (silent difference): loss must stay finite (no NaN/Inf)."""
    loss = _sd_loss(perceptual_weighting=False)
    mono = torch.randn(_B, 1, _T).repeat(1, 2, 1)            # (B, 2, T), L == R
    other = torch.randn(_B, 1, _T).repeat(1, 2, 1)
    out = loss(mono, other)
    assert torch.isfinite(out)


def test_backprop():
    loss = _sd_loss(perceptual_weighting=False)
    pred = torch.randn(_B, 2, _T, requires_grad=True)
    target = torch.randn(_B, 2, _T)
    loss(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_perceptual_weighting_runs_finite():
    """A-weighting path (SAO default) instantiates and produces a finite loss."""
    loss = _sd_loss(perceptual_weighting=True)
    out = loss(torch.randn(_B, 2, _T), torch.randn(_B, 2, _T))
    assert torch.isfinite(out) and out > 0.0


def test_pred_normalized_sc_no_spike_and_anticollapse():
    """SAO order: quasi-mono targets don't spike; a collapsed pred is
    penalized hard but stays finite (grad clipping handles the rest)."""
    torch.manual_seed(2)
    loss = _sd_loss(w_ms=1.0, w_lr=0.0, perceptual_weighting=False)
    mid = torch.randn(_B, 1, _T)
    # 1) quasi-mono TARGET (S≈0), healthy pred → loss O(1), no spike
    tgt_mono = torch.cat([mid, mid + 1e-6 * torch.randn(_B, 1, _T)], dim=1)
    pred = torch.randn(_B, 2, _T)
    l_mono_tgt = loss(pred, tgt_mono)
    assert torch.isfinite(l_mono_tgt) and l_mono_tgt < 50.0, l_mono_tgt
    # 2) collapsed PRED side (‖Ŝ‖ ≈ 0.15‖S‖) vs wide target → large but finite
    side = 0.3 * torch.randn(_B, 1, _T)
    tgt = torch.cat([mid + side, mid - side], dim=1)
    pred_collapsed = torch.cat([mid + 0.15 * side, mid - 0.15 * side], dim=1)
    pred_ok = torch.cat([mid + side, mid - side], dim=1)
    l_collapsed, l_ok = loss(pred_collapsed, tgt), loss(pred_ok, tgt)
    assert torch.isfinite(l_collapsed) and l_collapsed > l_ok, (l_collapsed, l_ok)


def test_registers_in_loss_manager():
    """The `mrstft_sd` block builds the loss and appends a named LossWithTarget."""
    from sage.training.loss_manager import LossManager

    class _DummyAE(torch.nn.Module):
        bottleneck = None
        has_pre_transform = False
        pre_transform = None

    lm = LossManager(
        autoencoder=_DummyAE(),
        sample_rate=44100,
        loss_config={"mrstft_sd": {"weights": {"mrstft_sd": 1.0},
                                   "config": {"fft_sizes": [256, 128],
                                              "hop_sizes": [64, 32],
                                              "win_lengths": [256, 128],
                                              "perceptual_weighting": False}}},
    )
    assert isinstance(lm.mrstft_sd, SumAndDifferenceSTFTLoss)
    names = [getattr(m, "name", "") for m in lm.losses_gen.losses]
    assert "mrstft_sd_loss" in names
