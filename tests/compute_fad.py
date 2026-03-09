## NOTE: FAD is a distributional metric — results are only meaningful with a large number of samples (hundreds+).
## With few samples (e.g. 10) the score is computed correctly but statistically unreliable.

import os
import sys
from pathlib import Path

# Add project root to path so we can import ar_spectra and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, RUNS_DIR
from argparse import ArgumentParser

from ar_spectra.utils.console import ok, warn, err, info

import numpy as np
import fadtk
from fadtk.fad import get_cache_embedding_path, calc_frechet_distance
from fadtk.model_loader import VGGishModel
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
_SKIP_DIRS = frozenset({"metrics", "convert", "embeddings"})


def compute_stats(npy_paths: list[Path]):
    """Compute mean and covariance from cached embedding .npy files (avoids multiprocessing)."""
    embeddings = np.concatenate([np.load(p).astype(np.float32) for p in npy_paths], axis=0)
    mu = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    return mu, cov


def collect_audio_files(directory: str) -> list[Path]:
    """Recursively collect all audio files under directory, skipping cache subdirectories."""
    root = Path(directory)
    results = []
    for f in root.rglob("*"):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        if _SKIP_DIRS.intersection(f.relative_to(root).parts[:-1]):
            continue
        results.append(f)
    return results


def embed_and_stats(fad: fadtk.FrechetAudioDistance, files: list[Path], desc: str = "Embedding"):
    """Compute and cache embeddings for all files, then return (mu, cov)."""
    npy_paths = []
    pbar = tqdm(files, desc=desc, unit="file")
    try:
        for f in pbar:
            fad.cache_embedding_file(f)
            npy_paths.append(get_cache_embedding_path(fad.ml.name, f))
    except KeyboardInterrupt:
        pbar.close()
        if not npy_paths:
            raise
        warn(f"[Interrupted] Using {len(npy_paths)}/{len(files)} embedded files for {desc}.")
    return compute_stats(npy_paths)


parser = ArgumentParser(description="Compute Frechet Audio Distance using fadtk")
parser.add_argument("--target-dir", type=str, default=str(DATA_PATH), required=False, help="Path to the target/reference audio directory")
parser.add_argument("--preds-dir", type=str, default=str(RUNS_DIR / "inference"), required=False, help="Path to the predicted/generated audio directory")
parser.add_argument("--model", type=str, default="vggish", choices=["vggish", "clap-laion"], help="Embedding model to use for FAD (default: vggish)")
parser.add_argument("--max-files", type=int, default=-1, help="Limit number of target files used (default: all)")

args = parser.parse_args()

target_dir = args.target_dir
preds_dir = args.preds_dir

info(f"Target directory:      {target_dir}")
info(f"Predictions directory: {preds_dir}")
info(f"Embedding model:       {args.model}")

if args.model == "clap-laion":
    from fadtk.model_loader import CLAPLaionModel
    model_loader = CLAPLaionModel("music")
else:
    model_loader = VGGishModel()

warn("FAD is a distributional metric. Results are only statistically meaningful with a large number of samples.")

fad = fadtk.FrechetAudioDistance(model_loader)

target_files = collect_audio_files(target_dir)
preds_files = collect_audio_files(preds_dir)

# Match target files to prediction files by stem so we compare the same tracks
pred_stems = {f.stem for f in preds_files}
target_files = [f for f in target_files if f.stem in pred_stems]

if args.max_files > 0:
    target_files = target_files[:args.max_files]
    preds_files = preds_files[:args.max_files]

info(f"Found {len(target_files)} target files and {len(preds_files)} prediction files.")

if len(target_files) == 0 or len(preds_files) == 0:
    err("No audio files found in one or both directories. Aborting.")
    sys.exit(1)

try:
    info("Computing embeddings for target files...")
    mu_t, cov_t = embed_and_stats(fad, target_files, desc="Target")

    info("Computing embeddings for prediction files...")
    mu_p, cov_p = embed_and_stats(fad, preds_files, desc="Predictions")

    fad_score = calc_frechet_distance(mu_t, cov_t, mu_p, cov_p)

    ok(f"FAD ({args.model}): {fad_score}")
except KeyboardInterrupt:
    warn("[Interrupted] FAD computation aborted before enough embeddings were collected.")