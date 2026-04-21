#!/usr/bin/env python
# ===============================================================
# compute_cdpam.py — Per-file CDPAM perceptual distance evaluation.
# Fixes: (1) amplitude scaling ×32768 for BatchNorm compatibility,
# (2) resample to 22050 Hz (CDPAM native rate), (3) file-by-file
# processing to avoid zero-padding corruption of global avg pool.
# ===============================================================
import sys
import argparse
import csv
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

# ── MONKEY-PATCH: cdpam compatibility fixes ───────────────────────────────────
# Must happen before `import cdpam` — patches weights_only default and np.float
_real_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _real_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import cdpam
if not hasattr(np, "float"):
    np.float = float
# ── END MONKEY-PATCH ──────────────────────────────────────────────────────────

from eval_dataloader import PairedEvalDataset
from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE, FMA_METADATA
from ar_spectra.utils.audio import load_waveform
from ar_spectra.utils.console import ok, warn, err, info

# CDPAM was trained on audio at 22050 Hz with int16-scale amplitudes.
# Passing float32 [-1,1] without scaling causes BatchNorm collapse → scores ≈ 0.
CDPAM_SR    = 22050   # native sample rate expected by CDPAM
CDPAM_SCALE = 32768.0 # int16 amplitude scale required by CDPAM BatchNorm

def main():
    parser = argparse.ArgumentParser(description="Per-file CDPAM perceptual distance.")
    parser.add_argument("--target-dir",  default=str(DATA_PATH))
    parser.add_argument("--preds-dir",   default=str(RUNS_DIR / "inference"))
    parser.add_argument("--device",      default=DEFAULT_DEVICE)
    parser.add_argument("--max-files",   type=int, default=0)
    parser.add_argument("--csv_out",     type=str, default="")
    parser.add_argument("--extensions",  type=str, default="wav,flac,mp3")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loss_fn = cdpam.CDPAM(dev=str(device))

    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir,
        preds_dir=args.preds_dir,
        extensions=exts,
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA,
    )
    info(f"Computing CDPAM for {len(dataset)} files (file-by-file, no batching)")

    # Single resampler reused across all files (stateless transform)
    resampler = torchaudio.transforms.Resample(44100, CDPAM_SR).to(device)

    per_file_results = []
    all_scores = []

    for target_path, pred_path in tqdm(dataset.pairs, desc="CDPAM"):
        try:
            # Load at 44100 Hz, downmix to mono → [1, T]
            t_wav, _, _, _ = load_waveform(target_path, target_sample_rate=44100, expected_channels=1)
            p_wav, _, _, _ = load_waveform(pred_path,   target_sample_rate=44100, expected_channels=1)

            t = t_wav.to(device)  # [1, T_44k]
            p = p_wav.to(device)  # [1, T_44k]

            # Resample to CDPAM native rate
            t = resampler(t)      # [1, T_22k]
            p = resampler(p)      # [1, T_22k]

            # Scale to int16 amplitude — required for BatchNorm compatibility
            t = t * CDPAM_SCALE   # [1, T_22k]
            p = p * CDPAM_SCALE   # [1, T_22k]

            # Trim to equal length (no zero-padding)
            min_len = min(t.shape[-1], p.shape[-1])
            t = t[..., :min_len]  # [1, T_22k]
            p = p[..., :min_len]  # [1, T_22k]

            # CDPAM.forward expects [B, T]; channel dim doubles as batch dim here
            with torch.no_grad():
                score = loss_fn.forward(t, p)  # scalar tensor

            s = score.item()
            per_file_results.append({"track": target_path.stem, "cdpam": s})
            all_scores.append(s)

        except Exception as e:
            warn(f"Failed on {target_path.stem}: {e}")

    if all_scores:
        final_mean = float(np.mean(all_scores))
        ok(f"CDPAM Final Score (mean over {len(all_scores)} files): {final_mean:.6f}")
    else:
        err("No valid CDPAM scores computed.")

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
