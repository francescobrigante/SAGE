#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import new evaluation utilities from tests folder
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from eval_dataloader import PairedEvalDataset, collate_paired_eval, batch_align

from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE, FMA_METADATA
from ar_spectra.utils.console import ok, warn, err, info
import numpy as np
import csv
from tqdm import tqdm

# ── MONKEY-PATCH: cdpam compatibility fixes ───────────────────────────────────
_real_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _real_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import cdpam
if not hasattr(np, "float"):
    np.float = float
# ── END MONKEY-PATCH ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-optimized CDPAM evaluation.")
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--csv_out", type=str, default="")
    parser.add_argument("--extensions", type=str, default="wav,flac,mp3")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loss_fn = cdpam.CDPAM(dev=str(device))
    
    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir, 
        preds_dir=args.preds_dir, 
        extensions=exts, 
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_paired_eval)

    num_batches = len(loader)
    info(f"Computing CDPAM for {len(dataset)} files in {num_batches} batches (BS={args.batch_size})")

    batch_means = []
    per_file_results = []
    
    pbar = tqdm(loader, desc="CDPAM Metrics", total=num_batches)
    for i, (target_batch, pred_batch, stems) in enumerate(pbar):
        target_batch = target_batch.to(device)
        pred_batch = pred_batch.to(device)
        
        # 1. Batched Alignment
        target_aligned, pred_aligned, _ = batch_align(target_batch, pred_batch, sr=44100)
        
        # 2. Mix to mono: [B, C, T] -> [B, T]
        # CDPAM expects [B, T] and internally calls unsqueeze(1); stereo input
        # would produce [B, 1, C, T] (4D) and crash conv1d.
        target_aligned = target_aligned.mean(dim=1)
        pred_aligned = pred_aligned.mean(dim=1)
        
        # 3. Chunking logic (if requested, or full waveform)
        # For simplicity and speed in batching, we truncate to a multiple of chunk_size 
        # or just use the full aligned waveform if chunk_size=0
        if args.chunk_size > 0:
            T = target_aligned.shape[-1]
            usable = (T // args.chunk_size) * args.chunk_size
            target_aligned = target_aligned[..., :usable]
            pred_aligned = pred_aligned[..., :usable]
            
            # Reshape to [B * NumChunks, 1, ChunkSize]
            B = target_aligned.shape[0]
            target_chunks = target_aligned.reshape(-1, 1, args.chunk_size)
            pred_chunks = pred_aligned.reshape(-1, 1, args.chunk_size)
            
            # Sub-batching to avoid OOM
            INTERNAL_BS = 64
            chunk_scores = []
            for start in range(0, target_chunks.shape[0], INTERNAL_BS):
                end = start + INTERNAL_BS
                with torch.no_grad():
                    # cdpam expects [B, T]
                    s = loss_fn.forward(target_chunks[start:end].squeeze(1), pred_chunks[start:end].squeeze(1))
                chunk_scores.append(s.flatten())
            
            all_scores = torch.cat(chunk_scores) # [B * NumChunks]
            # Reshape back to [B, NumChunks] to get per-file results
            all_scores = all_scores.reshape(B, -1)
            file_scores = all_scores.mean(dim=-1) # [B]
        else:
            with torch.no_grad():
                # Direct forward on entire aligned tracks
                s = loss_fn.forward(target_aligned, pred_aligned)
            file_scores = s.flatten()

        batch_mean = file_scores.mean().item()
        batch_means.append(batch_mean)
        
        pbar.set_postfix({"batch_mean": f"{batch_mean:.4f}"})
        
        for stem, score in zip(stems, file_scores.tolist()):
            per_file_results.append({"track": stem, "cdpam": score})
            
        info(f"Batch {i+1} mean: {batch_mean:.4f}")

    final_mean = np.mean(batch_means)
    ok(f"CDPAM Final Score (Mean of Batch-Means): {final_mean:.6f}")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["track", "cdpam"])
            writer.writeheader()
            writer.writerows(per_file_results)
        info(f"Saved CSV: {out_path}")

if __name__ == "__main__":
    main()