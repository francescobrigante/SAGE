# ===============================================================================
# compute_clap_score.py
# Per-pair CLAP-LAION cosine similarity between reference and reconstructed audio.
# Supports "music", "audio", or "both" flavours via fadtk's file-level .npy cache.
# ===============================================================================
import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, RUNS_DIR
from ar_spectra.utils.console import ok, warn, err, info

import fadtk
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import get_cache_embedding_path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
_SKIP_DIRS = frozenset({"metrics", "convert", "embeddings"})


def match_pairs(target_dir, preds_dir):
    target_dir = Path(target_dir)
    preds_dir = Path(preds_dir)

    target_map = {}
    for f in target_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            if _SKIP_DIRS.intersection(f.relative_to(target_dir).parts[:-1]):
                continue
            target_map.setdefault(f.stem, []).append(f)

    pairs = []
    for g in preds_dir.rglob("*"):
        if g.is_file() and g.suffix.lower() in AUDIO_EXTS:
            if _SKIP_DIRS.intersection(g.relative_to(preds_dir).parts[:-1]):
                continue
            if g.stem in target_map:
                pairs.append((target_map[g.stem][0], g))
    return pairs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_embedding(npy_path: Path) -> np.ndarray:
    """Load cached .npy embedding, mean-pool over time axis, return shape (D,)."""
    emb = np.atleast_2d(np.load(npy_path))  # (T, D) or already (1, D)
    return emb.mean(axis=0)                  # (D,)


def compute_clap_scores(flavour: str, pairs: list) -> list:
    """
    Embed all files for one CLAP flavour and compute per-pair cosine scores.

    Args:
        flavour: "music" or "audio"
        pairs: list of (target_path, pred_path) Path tuples

    Returns:
        List of (target_name, pred_name, score) tuples.
    """
    ml = CLAPLaionModel(flavour)
    fad = fadtk.FrechetAudioDistance(ml)
    model_name = fad.ml.name

    # Deduplicate — a target may appear in multiple pairs
    all_files = set()
    for t, p in pairs:
        all_files.add(t)
        all_files.add(p)

    info(f"[clap-{flavour}] Caching embeddings for {len(all_files)} files...")
    pbar = tqdm(sorted(all_files), desc=f"Embedding clap-{flavour}", unit="file")
    try:
        for f in pbar:
            fad.cache_embedding_file(f)
    except KeyboardInterrupt:
        pbar.close()
        warn("[Interrupted] Partial embeddings — cosine results may be incomplete.")

    results = []
    for tpath, ppath in pairs:
        try:
            npy_t = get_cache_embedding_path(model_name, tpath)
            npy_p = get_cache_embedding_path(model_name, ppath)
            score = cosine(load_embedding(npy_t), load_embedding(npy_p))
            results.append((tpath.name, ppath.name, score))
        except Exception as e:
            err(f"Error on pair {tpath.name}/{ppath.name}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLAP-LAION cosine similarity between reference and reconstructed audio."
    )
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--model", default="both", choices=["music", "audio", "both"])
    parser.add_argument("--csv_out", default="", help="Optional path for per-pair CSV output.")
    args = parser.parse_args()

    pairs = match_pairs(args.target_dir, args.preds_dir)
    if not pairs:
        warn("No matching pairs found.")
        return

    info(f"Found {len(pairs)} matched pairs.")

    flavours = ["music", "audio"] if args.model == "both" else [args.model]

    scores_by_flavour: dict[str, list] = {}
    for flavour in flavours:
        results = compute_clap_scores(flavour, pairs)
        scores_by_flavour[flavour] = results
        if results:
            mean_score = sum(r[2] for r in results) / len(results)
            ok(f"CLAP-{flavour} mean cosine score: {mean_score:.6f} over {len(results)} pairs.")
        else:
            warn(f"CLAP-{flavour}: no scores computed.")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge flavours into rows keyed by (target_name, pred_name)
        row_map: dict = {}
        for flavour, results in scores_by_flavour.items():
            for tname, pname, score in results:
                key = (tname, pname)
                row_map.setdefault(key, {"target_file": tname, "pred_file": pname})
                row_map[key][f"clap_{flavour}_score"] = score

        fieldnames = ["target_file", "pred_file"] + [f"clap_{f}_score" for f in flavours]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_map.values())
        info(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()
