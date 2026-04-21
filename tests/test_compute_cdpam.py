# ===============================================================
# test_compute_cdpam.py — Unit tests for CDPAM evaluation fixes.
# Tests amplitude scaling, sample-rate conversion, and file-by-file
# processing logic WITHOUT loading the real CDPAM model (mocked).
# ===============================================================
import sys
import csv
import pytest
import numpy as np
import torch
import torchaudio
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from ar_spectra.utils.audio import load_waveform

# Constants mirroring compute_cdpam.py
INPUT_SR    = 44100
CDPAM_SR    = 22050
CDPAM_SCALE = 32768.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_wav(path: Path, sr: int = INPUT_SR, duration: float = 1.0, channels: int = 1):
    """Write a deterministic random WAV file."""
    torch.manual_seed(hash(path.name) % 2**31)
    wav = torch.randn(channels, int(sr * duration)) * 0.3
    torchaudio.save(str(path), wav, sr)


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


# ── T4: end-to-end file loop (mocked CDPAM) ──────────────────────────────────

def test_file_by_file_loop_correct_scale_and_sr(tmp_path):
    """
    End-to-end: the processing loop resamples to CDPAM_SR and scales by CDPAM_SCALE
    before calling loss_fn.forward, and writes a valid CSV.
    Uses a mocked CDPAM so no model weights are loaded.
    """
    target_dir = tmp_path / "targets"
    pred_dir   = tmp_path / "preds"
    target_dir.mkdir()
    pred_dir.mkdir()

    stems = ["track_a", "track_b", "track_c"]
    for stem in stems:
        _write_wav(target_dir / f"{stem}.wav")
        _write_wav(pred_dir   / f"{stem}.wav")

    # Mock CDPAM: returns a realistic scalar score
    mock_loss_fn = MagicMock()
    mock_loss_fn.forward.return_value = torch.tensor(0.35)

    from eval_dataloader import PairedEvalDataset

    dataset = PairedEvalDataset(
        target_dir=str(target_dir),
        preds_dir=str(pred_dir),
        extensions=[".wav"],
        max_files=0,
        fma_csv_path=None,
    )

    resampler = torchaudio.transforms.Resample(INPUT_SR, CDPAM_SR)
    per_file_results = []

    for target_path, pred_path in dataset.pairs:
        t_wav, _, _, _ = load_waveform(target_path, target_sample_rate=INPUT_SR, expected_channels=1)
        p_wav, _, _, _ = load_waveform(pred_path,   target_sample_rate=INPUT_SR, expected_channels=1)

        t = resampler(t_wav) * CDPAM_SCALE   # [1, T_22k]
        p = resampler(p_wav) * CDPAM_SCALE   # [1, T_22k]
        min_len = min(t.shape[-1], p.shape[-1])
        t, p = t[..., :min_len], p[..., :min_len]

        score = mock_loss_fn.forward(t, p).item()
        per_file_results.append({"track": target_path.stem, "cdpam": score})

    # Correct number of results
    assert len(per_file_results) == len(stems), (
        f"Expected {len(stems)} results, got {len(per_file_results)}"
    )

    # All mock scores preserved
    assert all(r["cdpam"] == pytest.approx(0.35) for r in per_file_results)

    # Verify the audio passed to forward was at CDPAM_SR and int16-scale
    first_call = mock_loss_fn.forward.call_args_list[0]
    t_passed = first_call[0][0]
    assert abs(t_passed.shape[-1] - CDPAM_SR) <= 2, (
        f"forward() should receive audio at ~{CDPAM_SR} samples, got {t_passed.shape[-1]}"
    )
    assert t_passed.abs().max().item() > 100, (
        "forward() should receive int16-scaled audio (> 100), got near-zero amplitude"
    )

    # Write and verify CSV
    csv_path = tmp_path / "cdpam.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "cdpam"])
        writer.writeheader()
        writer.writerows(per_file_results)

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == len(stems)
    assert set(rows[0].keys()) == {"track", "cdpam"}
