#!/usr/bin/env python
# ===============================================================
# compute_spectral.py — Per-file SI-SDR and perceptual STFT loss.
# Processes audio file-by-file (no DataLoader/zero-padding) to
# avoid batch-alignment artifacts from collate zero-padding.
# ===============================================================
import sys
import argparse
import csv
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torchmetrics.audio.sdr import SignalDistortionRatio as SISDRMetric

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from eval_dataloader import PairedEvalDataset, batch_align
from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE, FMA_METADATA
from ar_spectra.utils.audio import load_waveform
from ar_spectra.utils.console import ok, warn, err, info
from ar_spectra.training.losses.signal import STFTLoss

SR = 44100

def main():
    parser = argparse.ArgumentParser(description="Per-file spectral metrics (SI-SDR, STFT).")
    parser.add_argument("--target-dir",  type=str, default=str(DATA_PATH))
    parser.add_argument("--preds-dir",   default=str(RUNS_DIR / "inference"))
    parser.add_argument("--extensions",  type=str, default="wav,flac,mp3")
    parser.add_argument("--max-files",   type=int, default=0)
    parser.add_argument("--csv_out",     type=str, default="")
    args = parser.parse_args()

    device = torch.device(DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float32 if device.type == "mps" else torch.float64

    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir,
        preds_dir=args.preds_dir,
        sample_rate=SR,
        channels=1,
        extensions=exts,
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA,
    )
    info(f"Computing spectral metrics for {len(dataset)} files (file-by-file, no batching)")

    sisdr_metric = SISDRMetric().to(device)
    stft_loss_fn = STFTLoss(
        fft_size=2048,
        hop_size=512,
        win_length=2048,
        perceptual_weighting=True,
        w_log_mag=1.0,
        sample_rate=SR,
        reduction="none",
    ).to(device=device, dtype=compute_dtype)

    per_file_results = []
    stft_scores_all  = []
    sisdr_scores_all = []

    for target_path, pred_path in tqdm(dataset.pairs, desc="Spectral Metrics"):
        try:
            # Load at 44100 Hz, mono → [1, T]
            t_wav, _, _, _ = load_waveform(target_path, target_sample_rate=SR, expected_channels=1)
            p_wav, _, _, _ = load_waveform(pred_path,   target_sample_rate=SR, expected_channels=1)

            # batch_align expects [B, C, T]; unsqueeze adds the batch dim
            t = t_wav.unsqueeze(0).to(device=device, dtype=compute_dtype)  # [1, 1, T]
            p = p_wav.unsqueeze(0).to(device=device, dtype=compute_dtype)  # [1, 1, T]

            t_aligned, p_aligned, _ = batch_align(t, p, sr=SR)  # [1, 1, T]

            with torch.no_grad():
                # STFTLoss: [1, 1, T] → [1] with reduction="none"; flatten → scalar
                stft_score = stft_loss_fn(p_aligned, t_aligned).flatten().mean().item()
                # SISDRMetric: expects [B, T]; squeeze channel dim
                sisdr_score = sisdr_metric(
                    p_aligned.squeeze(1), t_aligned.squeeze(1)  # [1, T]
                ).item()

            per_file_results.append({
                "target_file": target_path.stem,
                "stft_loss":   stft_score,
                "si_sdr":      sisdr_score,
            })
            stft_scores_all.append(stft_score)
            sisdr_scores_all.append(sisdr_score)

        except Exception as e:
            warn(f"Failed on {target_path.stem}: {e}")

    if stft_scores_all:
        ok(f"Final Results ({len(stft_scores_all)} files):")
        ok(f"  STFTLoss: {np.mean(stft_scores_all):.6f}")
        ok(f"  SI-SDR:   {np.mean(sisdr_scores_all):.6f}")
    else:
        err("No spectral scores computed.")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["target_file", "stft_loss", "si_sdr"])
            writer.writeheader()
            writer.writerows(per_file_results)
        info(f"Saved CSV: {out_path}")

if __name__ == "__main__":
    main()
