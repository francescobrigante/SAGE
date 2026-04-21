# ===============================================================
# test_compute_spectral.py — Integration tests for per-file
# SI-SDR and STFT loss computation (compute_spectral.py).
# Uses synthetic WAV files — no real checkpoints required.
# ===============================================================
import sys
import csv
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torchmetrics.audio.sdr import SignalDistortionRatio as SISDRMetric

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from eval_dataloader import PairedEvalDataset, batch_align
from ar_spectra.utils.audio import load_waveform
from ar_spectra.training.losses.signal import STFTLoss

SR = 44100


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_wav(path: Path, wav: torch.Tensor, sr: int = SR):
    """Save a tensor [C, T] as WAV."""
    torchaudio.save(str(path), wav.float(), sr)


def _sine_wav(duration: float = 1.0, freq: float = 440.0, sr: int = SR) -> torch.Tensor:
    """Generate a clean mono sine wave → [1, T]."""
    t = torch.linspace(0, duration, int(sr * duration))
    return (torch.sin(2 * torch.pi * freq * t) * 0.5).unsqueeze(0)


def _compute_pair(target: torch.Tensor, pred: torch.Tensor):
    """Run alignment + SI-SDR + STFT on a single [1, T] pair. Returns (stft, sisdr)."""
    device = torch.device("cpu")
    dtype  = torch.float64

    sisdr_metric = SISDRMetric().to(device)
    stft_loss_fn = STFTLoss(
        fft_size=2048, hop_size=512, win_length=2048,
        perceptual_weighting=True, w_log_mag=1.0, sample_rate=SR, reduction="none",
    ).to(device=device, dtype=dtype)

    t = target.unsqueeze(0).to(device=device, dtype=dtype)  # [1, 1, T]
    p = pred.unsqueeze(0).to(device=device, dtype=dtype)    # [1, 1, T]

    t_al, p_al, _ = batch_align(t, p, sr=SR)  # [1, 1, T]

    with torch.no_grad():
        stft  = stft_loss_fn(p_al, t_al).flatten().mean().item()
        sisdr = sisdr_metric(p_al.squeeze(1), t_al.squeeze(1)).item()

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
    """Adding noise should decrease SI-SDR compared to identical signals."""
    wav  = _sine_wav()
    _, sisdr_clean   = _compute_pair(wav, wav.clone())

    torch.manual_seed(42)
    noisy = wav + torch.randn_like(wav) * 0.3
    _, sisdr_noisy = _compute_pair(wav, noisy)

    assert sisdr_noisy < sisdr_clean, (
        f"Noisy pred (SI-SDR={sisdr_noisy:.1f}) should be worse than clean ({sisdr_clean:.1f})"
    )


def test_corrupted_signal_higher_stft():
    """Adding noise should increase STFT loss compared to identical signals."""
    wav = _sine_wav()
    stft_clean, _ = _compute_pair(wav, wav.clone())

    torch.manual_seed(42)
    noisy = wav + torch.randn_like(wav) * 0.3
    stft_noisy, _ = _compute_pair(wav, noisy)

    assert stft_noisy > stft_clean, (
        f"Noisy pred (STFT={stft_noisy:.4f}) should be worse than clean ({stft_clean:.4f})"
    )


# ── T3: end-to-end CSV output ─────────────────────────────────────────────────

def test_per_file_csv_correct_columns_and_count(tmp_path):
    """
    Full pipeline: write synthetic pairs → process file-by-file → verify CSV.
    Uses 4 pairs: 2 clean (pred=target) and 2 noisy, checks ordering of quality.
    """
    target_dir = tmp_path / "targets"
    pred_dir   = tmp_path / "preds"
    target_dir.mkdir()
    pred_dir.mkdir()

    torch.manual_seed(0)

    # Two clean pairs (pred ≈ target) and two noisy pairs
    clean_stems = ["clean_001", "clean_002"]
    noisy_stems = ["noisy_001", "noisy_002"]

    for stem in clean_stems:
        w = _sine_wav(freq=440.0)
        _write_wav(target_dir / f"{stem}.wav", w)
        _write_wav(pred_dir   / f"{stem}.wav", w.clone())          # identical

    for stem in noisy_stems:
        w = _sine_wav(freq=660.0)
        _write_wav(target_dir / f"{stem}.wav", w)
        _write_wav(pred_dir   / f"{stem}.wav", w + torch.randn_like(w) * 0.5)  # noisy

    device     = torch.device("cpu")
    dtype      = torch.float64
    sisdr_fn   = SISDRMetric().to(device)
    stft_fn    = STFTLoss(
        fft_size=2048, hop_size=512, win_length=2048,
        perceptual_weighting=True, w_log_mag=1.0, sample_rate=SR, reduction="none",
    ).to(device=device, dtype=dtype)

    dataset = PairedEvalDataset(
        target_dir=str(target_dir),
        preds_dir=str(pred_dir),
        extensions=[".wav"],
        max_files=0,
        fma_csv_path=None,
    )

    results = []
    for target_path, pred_path in dataset.pairs:
        t_wav, _, _, _ = load_waveform(target_path, target_sample_rate=SR, expected_channels=1)
        p_wav, _, _, _ = load_waveform(pred_path,   target_sample_rate=SR, expected_channels=1)

        t = t_wav.unsqueeze(0).to(device=device, dtype=dtype)
        p = p_wav.unsqueeze(0).to(device=device, dtype=dtype)
        t_al, p_al, _ = batch_align(t, p, sr=SR)

        with torch.no_grad():
            stft  = stft_fn(p_al, t_al).flatten().mean().item()
            sisdr = sisdr_fn(p_al.squeeze(1), t_al.squeeze(1)).item()

        results.append({"target_file": target_path.stem, "stft_loss": stft, "si_sdr": sisdr})

    # --- structural checks ---
    assert len(results) == len(clean_stems) + len(noisy_stems)
    assert set(results[0].keys()) == {"target_file", "stft_loss", "si_sdr"}

    # --- quality ordering ---
    by_stem = {r["target_file"]: r for r in results}

    for stem in clean_stems:
        assert by_stem[stem]["si_sdr"] > 30.0, (
            f"{stem}: expected SI-SDR > 30 dB for identical pair, got {by_stem[stem]['si_sdr']:.1f}"
        )
        assert by_stem[stem]["stft_loss"] < 0.05, (
            f"{stem}: expected STFT < 0.05 for identical pair, got {by_stem[stem]['stft_loss']:.4f}"
        )

    for stem in noisy_stems:
        assert by_stem[stem]["si_sdr"] < 20.0, (
            f"{stem}: expected SI-SDR < 20 dB for noisy pair, got {by_stem[stem]['si_sdr']:.1f}"
        )

    # --- CSV round-trip ---
    csv_path = tmp_path / "spectral.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_file", "stft_loss", "si_sdr"])
        writer.writeheader()
        writer.writerows(results)

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"target_file", "stft_loss", "si_sdr"}
