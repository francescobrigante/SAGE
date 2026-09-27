# ===============================================================
# test_compute_cdpam.py — Unit tests for CDPAM evaluation fixes.
# Tests amplitude scaling, sample-rate conversion and metrics.signal.cdpam_score
# WITHOUT loading the real CDPAM model (mocked).
# ===============================================================
import pytest
import numpy as np
import torch
import torchaudio
from unittest.mock import MagicMock

# Constants mirroring evaluation/metrics/signal.py::cdpam_score
INPUT_SR    = 44100
CDPAM_SR    = 22050
CDPAM_SCALE = 32768.0


# ── T1: resampling ────────────────────────────────────────────────────────────

def test_resample_to_cdpam_sr():
    """Resampling from 44100 Hz produces output at 22050 Hz (half the samples)."""
    resampler = torchaudio.transforms.Resample(INPUT_SR, CDPAM_SR)
    wav = torch.randn(1, INPUT_SR)           # exactly 1 s at 44100 Hz
    out = resampler(wav)
    assert abs(out.shape[-1] - CDPAM_SR) <= 2, (
        f"Expected ~{CDPAM_SR} samples after resampling, got {out.shape[-1]}"
    )


# ── T2: amplitude scaling ─────────────────────────────────────────────────────

def test_amplitude_scaling_puts_audio_in_int16_range():
    """Multiplying by CDPAM_SCALE moves float32 [-1,1] audio into int16 range."""
    wav = torch.randn(1, CDPAM_SR) * 0.3    # typical peak ≈ 0.9 → after scale ≈ 9830 (well under 32768)
    scaled = wav * CDPAM_SCALE
    assert scaled.abs().max().item() > 100, (
        "Scaled amplitude should be >> 1 (int16 range), got near-zero"
    )
    assert scaled.abs().max().item() < CDPAM_SCALE * 2, (
        "Scaled amplitude should not exceed 2× int16 max"
    )


# ── T3: BatchNorm collapse proof ──────────────────────────────────────────────

def test_batchnorm_collapses_without_int16_scaling():
    """
    Demonstrate the root-cause bug: BN trained on int16-scale data (running_var ≈ 9830²)
    collapses float32 [-1,1] inputs to ~0, but passes int16-scaled inputs normally.
    This is exactly what caused CDPAM ≈ 0 for all models.
    """
    bn = torch.nn.BatchNorm1d(64, momentum=None)
    bn.eval()
    # Simulate CDPAM's pretrained BN statistics: trained on int16-scale audio
    bn.running_var.fill_(9830.0 ** 2)
    bn.running_mean.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.bias.data.fill_(0.0)

    x_float  = torch.randn(1, 64, 100) * 0.3   # float32 [-1,1] range
    x_scaled = x_float * CDPAM_SCALE            # int16 range

    out_float  = bn(x_float)
    out_scaled = bn(x_scaled)

    # Float input collapses: BN divides by sqrt(9830²) ≈ 9830, so 0.3 / 9830 ≈ 3e-5
    assert out_float.abs().mean().item() < 1e-3, (
        f"Float input should collapse near zero, got mean={out_float.abs().mean():.6f}"
    )
    # Scaled input produces normal activations
    assert out_scaled.abs().mean().item() > 0.1, (
        f"Scaled input should produce normal activations, got mean={out_scaled.abs().mean():.6f}"
    )


# ── T4: cdpam_score end to end (mocked CDPAM) ─────────────────────────────────

def test_cdpam_score_feeds_mono_22k_int16_scale(monkeypatch):
    """cdpam_score resamples to CDPAM_SR, downmixes to mid, scales by CDPAM_SCALE, trims to the
    shorter signal and returns the model's scalar. Uses a mocked CDPAM: no weights are loaded."""
    from evaluation.metrics import signal as losses
    mock = MagicMock()
    mock.forward.return_value = torch.tensor(0.35)
    monkeypatch.setattr(losses, "_cdpam_model", mock)

    torch.manual_seed(0)
    target = 0.3 * torch.randn(2, INPUT_SR)                     # 1 s stereo
    pred = 0.3 * torch.randn(2, INPUT_SR + 441)                 # 10 ms longer
    score = losses.cdpam_score(target, pred, INPUT_SR, device="cpu")

    assert score == pytest.approx(0.35)
    t_passed, p_passed = mock.forward.call_args[0]
    assert t_passed.shape == p_passed.shape                     # trimmed to the shorter signal
    assert t_passed.shape[0] == 1 and abs(t_passed.shape[-1] - CDPAM_SR) <= 2
    assert t_passed.abs().max().item() > 100                    # int16 scale, not [-1, 1]
    expected = torchaudio.transforms.Resample(INPUT_SR, CDPAM_SR)(target).mean(0) * CDPAM_SCALE
    assert torch.allclose(t_passed[0], expected[: t_passed.shape[-1]], atol=1e-2)
