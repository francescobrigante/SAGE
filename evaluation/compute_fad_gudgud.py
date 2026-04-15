#!/usr/bin/env python
import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
import logging
from contextlib import contextmanager, redirect_stdout

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ar_spectra.utils.console import ok, warn, err, info
import fadtk
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import calc_frechet_distance

# Import optimized embedding logic
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from eval_dataloader import batch_embed_files
from compute_fad import compute_stats

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
_SKIP_DIRS = frozenset({"metrics", "convert", "embeddings", "cache"})

@contextmanager
def silence_output():
    """Context manager to silence stdout and logging."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            # Also silence logging
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                yield
            finally:
                logger.setLevel(old_level)

_CLAP_AUDIO_CFG: dict = {
    "model_name":    "clap",
    "sample_rate":   48000,
    "submodel_name": "630k-audioset",
    "enable_fusion": False,
}

def collect_audio_files(directory: str, max_files=0) -> list[Path]:
    root = Path(directory)
    results = []
    for f in root.rglob("*"):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        if _SKIP_DIRS.intersection(f.relative_to(root).parts[:-1]):
            continue
        results.append(f)
    results = sorted(results)
    if max_files > 0:
        results = results[:max_files]
    return results

def _make_symlink_dir(files: list[Path], parent: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(dir=parent))
    for f in files:
        link = tmp / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    return tmp

def main():
    parser = argparse.ArgumentParser(description="Batch-compatible FAD (gudgud96).")
    parser.add_argument("--target-dir", default=os.getcwd())
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, help="Unused for now by gudgud96 but accepted for API consistency.")
    parser.add_argument("--model", default="clap-audio")
    parser.add_argument("--csv_out", type=Path, help="Path to save result CSV.")
    args = parser.parse_args()

    # Collect and match
    pred_files = collect_audio_files(args.preds_dir, max_files=args.max_files)
    pred_stems = {f.stem for f in pred_files}
    target_files = collect_audio_files(args.target_dir)
    target_matched = [f for f in target_files if f.stem in pred_stems]

    if not target_matched or not pred_files:
        err(f"No matching audio pairs found.")
        sys.exit(1)

    info(f"FAD gudgud: Matched {len(target_matched)} target / {len(pred_files)} prediction files.")

    # 1. Initialize Model
    # We use CLAPLaionModel(audio) to match gudgud's default behavior
    info(f"Initializing CLAP model for FAD...")
    with silence_output():
        ml = CLAPLaionModel("audio")
        fad = fadtk.FrechetAudioDistance(ml)
    ok("CLAP model initialized successfully.")

    # 2. Batch Embed
    # Use a unique subfolder for predictions based on the stem of the parent directory
    ckpt_name = Path(args.preds_dir).parent.name
    pred_subfolder = f"preds_{ckpt_name}"

    info("Phase 1/2 - Embedding Target Files...")
    batch_embed_files(fad, target_matched, batch_size=args.batch_size, cache_dir=args.cache_dir, subfolder="target")

    info(f"Phase 2/2 - Embedding Prediction Files into '{pred_subfolder}'...")
    batch_embed_files(fad, pred_files, batch_size=args.batch_size, cache_dir=args.cache_dir, subfolder=pred_subfolder)

    # 3. Compute Stats
    info("Computing Statistics...")
    mu_t, cov_t = compute_stats(fad, target_matched, cache_dir=args.cache_dir, subfolder="target")
    mu_p, cov_p = compute_stats(fad, pred_files, cache_dir=args.cache_dir, subfolder=pred_subfolder)

    if mu_t is None or mu_p is None:
        err("Failed to collect enough embeddings for FAD.")
        sys.exit(1)

    fad_score = calc_frechet_distance(mu_t, cov_t, mu_p, cov_p)
    ok(f"FAD gudgud ({args.model}): {fad_score:.6f}")

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
