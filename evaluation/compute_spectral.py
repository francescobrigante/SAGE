#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import csv
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.audio.sdr import SignalDistortionRatio as SISDRMetric

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import new evaluation utilities
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from eval_dataloader import PairedEvalDataset, collate_paired_eval, batch_align

from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE, FMA_METADATA
from ar_spectra.utils.console import ok, warn, err, info
from ar_spectra.training.losses.signal import STFTLoss
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Batch-optimized Spectral metrics (SI-SDR, STFT).")
    parser.add_argument("--target-dir", type=str, default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--extensions", type=str, default="wav,flac,mp3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--csv_out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float32 if device.type == "mps" else torch.float64
    
    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir, 
        preds_dir=args.preds_dir,
        sample_rate=44100, 
        channels=1, 
        extensions=exts, 
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_paired_eval)

    # Metrics setup
    sisdr_metric = SISDRMetric().to(device)
    stft_loss_fn = STFTLoss(
        fft_size=2048,
        hop_size=512,
        win_length=2048,
        perceptual_weighting=True,
        w_log_mag=1.0,
        sample_rate=44100,
        reduction="none", # We want per-file scores
    ).to(device=device, dtype=compute_dtype)

    num_batches = len(loader)
    info(f"Computing Spectral metrics for {len(dataset)} files in {num_batches} batches (BS={args.batch_size})")

    batch_stft_means = []
    batch_sisdr_means = []
    per_file_results = []

    pbar = tqdm(loader, desc="Spectral Metrics", total=num_batches)
    for i, (target_batch, pred_batch, stems) in enumerate(pbar):
        target_batch = target_batch.to(device=device, dtype=compute_dtype)
        pred_batch = pred_batch.to(device=device, dtype=compute_dtype)
        
        # 1. Batched Alignment
        target_aligned, pred_aligned, _ = batch_align(target_batch, pred_batch, sr=44100)
        
        # SI-SDR and STFT Loss expect [B, T] or [B, 1, T]
        # Our dataloader returns [B, C, T]. Since we forced channels=1, it's [B, 1, T].
        
        with torch.no_grad():
            # STFT Loss
            stft_scores = stft_loss_fn(pred_aligned, target_aligned) # [B, 1] or [B]
            stft_scores = stft_scores.flatten()
            
            # SI-SDR (requires [B, T])
            sisdr_scores = []
            for j in range(len(stems)):
                # SI-SDR is often unstable in batches if lengths differ or padding is heavy.
                # Even though we aligned, we might have padded zeros.
                # We'll compute it per-item in the batch for maximum robustness 
                # but it's still fast on GPU.
                s = sisdr_metric(pred_aligned[j:j+1].squeeze(1), target_aligned[j:j+1].squeeze(1))
                sisdr_scores.append(s)
            sisdr_scores = torch.stack(sisdr_scores).flatten()

        b_stft = stft_scores.mean().item()
        b_sisdr = sisdr_scores.mean().item()
        batch_stft_means.append(b_stft)
        batch_sisdr_means.append(b_sisdr)
        
        for stem, s_stft, s_sisdr in zip(stems, stft_scores.tolist(), sisdr_scores.tolist()):
            per_file_results.append({
                "target_file": stem,
                "stft_loss": s_stft,
                "si_sdr": s_sisdr
            })
            
        pbar.set_postfix({"batch_stft": f"{b_stft:.3f}", "batch_sisdr": f"{b_sisdr:.3f}"})

    final_stft = np.mean(batch_stft_means)
    final_sisdr = np.mean(batch_sisdr_means)
    
    ok(f"Final Results (Mean of Batch-Means):")
    ok(f"  STFTLoss: {final_stft:.6f}")
    ok(f"  SI-SDR:   {final_sisdr:.6f}")

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