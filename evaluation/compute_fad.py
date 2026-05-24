#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import new evaluation utilities
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from eval_dataloader import PairedEvalDataset, batch_embed_files

from config import DATA_PATH, RUNS_DIR
from ar_spectra.utils.console import ok, warn, err, info

import fadtk
from fadtk.fad import get_cache_embedding_path, calc_frechet_distance
from fadtk.model_loader import VGGishModel, CLAPLaionModel, MERTModel

def compute_stats(fad, files: list[Path], cache_dir: Path = None, subfolder: str = None):
    """Compute mean and covariance from cached embedding .npy files."""
    model_name = fad.ml.name
    embeddings = []
    
    for f in files:
        if not cache_dir:
            cp = get_cache_embedding_path(model_name, f)
        else:
            if subfolder:
                cp = cache_dir / model_name / subfolder / f"{f.stem}.npy"
            else:
                cp = cache_dir / model_name / f"{f.stem}.npy"
        
        if cp.exists():
            # (T, D) -> pool to (D,)
            emb = np.atleast_2d(np.load(cp)).astype(np.float32)
            embeddings.append(emb)
    
    if not embeddings:
        return None, None
        
    all_embs = np.concatenate(embeddings, axis=0)
    mu = np.mean(all_embs, axis=0)
    cov = np.cov(all_embs, rowvar=False)
    return mu, cov


def embed_mert(ml, wav: torch.Tensor, src_sr: int, device) -> np.ndarray:
    """In-memory MERT embedding from a [C, T] waveform tensor.

    Returns (N_chunks, D) float16 array.
    Imported by evaluate.py for the in-memory pipeline.
    """
    msr = 24000
    if src_sr != msr:
        wav = torchaudio.functional.resample(wav.cpu(), src_sr, msr)
    wav_m = wav.mean(0)  # (T,)

    cl, hl = 5 * msr, msr
    chunks = [
        F.pad(wav_m[s: s + cl], (0, max(0, cl - wav_m[s: s + cl].shape[0]))).cpu().numpy()
        for s in range(0, wav_m.shape[0], hl)
    ] or [np.zeros(cl, dtype=np.float32)]

    inputs = ml.processor(chunks, sampling_rate=msr, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = ml.model(**inputs, output_hidden_states=True)
    layer = getattr(ml, "layer", 12)
    return out.hidden_states[layer].mean(1).cpu().numpy().astype(np.float16)


def compute_fad_from_embeddings(
    ml,
    target_files:  list[Path],
    pred_emb_list: list[np.ndarray],
    shared_cache:  Path,
    csv_path:      Path,
    model_label:   str,
) -> Optional[float]:
    """Compute FAD from cached target .npy embeddings + in-memory pred embeddings.

    Used by evaluate.py — no WAV files are written to disk.
    """
    import csv as _csv
    model_name: str = ml.name
    t_embs = [
        np.load(shared_cache / model_name / "target" / f"{f.stem}.npy").astype(np.float32)
        for f in target_files
        if (shared_cache / model_name / "target" / f"{f.stem}.npy").exists()
    ]
    if not t_embs:
        warn(f"No cached target embeddings for {model_name} — FAD skipped.")
        return None

    all_t = np.concatenate(t_embs, axis=0)
    all_p = np.concatenate(pred_emb_list, axis=0).astype(np.float32)
    score = calc_frechet_distance(
        all_t.mean(0), np.cov(all_t, rowvar=False),
        all_p.mean(0), np.cov(all_p, rowvar=False),
    )
    ok(f"FAD ({model_label}): {score:.6f}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "score"])
        w.writeheader()
        w.writerow({"model": model_label, "score": score})
    return score

def main():
    parser = argparse.ArgumentParser(description="Batch-optimized FAD evaluation (fadtk).")
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--model", default="mert", choices=["vggish", "clap-laion", "clap-laion-audio", "mert"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, help="Per-checkpoint cache (predictions).")
    parser.add_argument("--shared-cache-dir", type=Path, default=None,
                        help="Shared cache for target embeddings (reused across checkpoints).")
    parser.add_argument("--extensions", default="wav,mp3,flac")
    parser.add_argument("--csv_out", type=Path, help="Path to save result CSV.")
    args = parser.parse_args()

    # 0. Load Data
    from config import FMA_METADATA
    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir, 
        preds_dir=args.preds_dir, 
        extensions=exts, 
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA
    )
    info(f"FAD fadtk: Matched {len(dataset)} pairs for distributional comparison.")

    # Select model
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        if args.model == "clap-laion":
            ml = CLAPLaionModel("music")
        elif args.model == "mert":
            ml = MERTModel(size='v1-95M', layer=12)
        elif args.model == "clap-laion-audio":
            ml = CLAPLaionModel("audio")
        else:
            ml = VGGishModel()

        fad = fadtk.FrechetAudioDistance(ml)
    
    # pred_subfolder is unique per checkpoint; target cache may be shared across checkpoints
    ckpt_name = Path(args.preds_dir).name
    pred_subfolder = f"preds_{ckpt_name}"
    target_cache = args.shared_cache_dir if args.shared_cache_dir else args.cache_dir

    # 1. Collect unique files
    target_files = sorted(list({p[0] for p in dataset.pairs}))
    pred_files = sorted(list({p[1] for p in dataset.pairs}))

    # 2. Batch Embed — targets go to shared cache, predictions to per-checkpoint cache
    info("Phase 1/2 - Embedding Target Files → shared cache...")
    batch_embed_files(fad, target_files, batch_size=args.batch_size, cache_dir=target_cache, subfolder="target")

    info(f"Phase 2/2 - Embedding Prediction Files into '{pred_subfolder}'...")
    batch_embed_files(fad, pred_files, batch_size=args.batch_size, cache_dir=args.cache_dir, subfolder=pred_subfolder)

    # 3. Compute Stats
    info("Computing Statistics...")
    mu_t, cov_t = compute_stats(fad, target_files, cache_dir=target_cache, subfolder="target")
    mu_p, cov_p = compute_stats(fad, pred_files, cache_dir=args.cache_dir, subfolder=pred_subfolder)

    if mu_t is None or mu_p is None:
        err("Failed to collect enough embeddings for FAD.")
        sys.exit(1)

    fad_score = calc_frechet_distance(mu_t, cov_t, mu_p, cov_p)
    ok(f"FAD ({args.model}): {fad_score:.6f}")

    if args.csv_out:
        import csv
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "score"])
            writer.writeheader()
            writer.writerow({"model": args.model, "score": fad_score})
        info(f"Saved FAD result to {args.csv_out}")

if __name__ == "__main__":
    main()