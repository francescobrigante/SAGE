#!/usr/bin/env python3
"""
evaluation/maeb/plot_latent_umap.py

Embed MoisesDB instrument stems with SAGE encoder using the exact MAEB protocol
(same file list, same prepare_audio path, same standardize_bottleneck pooling),
then project the 64-dim embeddings to 2D with UMAP and save a scatter plot.

Run from C-VAE/ root with the maeb_dl venv:
    python evaluation/maeb/plot_latent_umap.py \
        --ckpt checkpoints/SAGE_e299.ckpt \
        --out  evaluation/maeb/figures

Dependencies (all in maeb_dl): umap-learn, matplotlib, pandas, torch, torchaudio.
Falls back to t-SNE (sklearn) if umap-learn is unavailable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Path setup (mirrors swin_encoder.py + maeb_swin.py) ─────────────────────
_EVAL_DIR = Path(__file__).resolve().parent          # evaluation/maeb/
_REPO_ROOT = _EVAL_DIR.parent.parent                 # C-VAE/
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR.parent))            # evaluation/ — for `from maeb import …`

from maeb import compatibility                        # noqa: E402 — side-effect
compatibility.add_ar_spectra_to_path(_REPO_ROOT)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
from tqdm.auto import tqdm                           # noqa: E402

from maeb.audio_prep import AudioDecodeError, prepare_audio   # noqa: E402
from maeb.moisesdb_tasks import (                             # noqa: E402
    _INSTRUMENT_CLASSES,
    _get_stems,
)
from maeb.swin_encoder import SwinEncoder            # noqa: E402

log = logging.getLogger(__name__)

# ── Colorblind-safe palette (Wong et al. 2011, 7-colour subset) ──────────────
# Assigned in alphabetical class order (bass drums guitar other_keys
# percussion piano vocals) so colours are deterministic regardless of run order.
_PALETTE = [
    "#0072B2",   # blue        → bass
    "#D55E00",   # vermillion  → drums
    "#009E73",   # bluish green → guitar
    "#CC79A7",   # reddish purple → other_keys
    "#56B4E9",   # sky blue    → percussion
    "#E69F00",   # orange      → piano
    "#F0E442",   # yellow      → vocals
]


# ─────────────────────────────────────────────────────────────────────────────

def _build_file_list():
    """Return (paths, string_labels, int_labels) for the instrument task.

    Replicates MoisesDBInstrumentClassification._build_dataset() stem selection
    exactly — same filter, same alphabetical category-code assignment, no cap
    (max_files=0 → full set).
    """
    stems = _get_stems()
    sub = stems[stems["stem_name"].isin(_INSTRUMENT_CLASSES)].copy()
    # Alphabetical category codes (matches pandas default in the task)
    sub["label_int"] = sub["stem_name"].astype("category").cat.codes
    paths = sub["_path"].tolist()
    str_labels = sub["stem_name"].tolist()
    int_labels = sub["label_int"].tolist()
    return paths, str_labels, int_labels


@torch.no_grad()
def embed_stems(encoder: SwinEncoder, paths: list[str]) -> np.ndarray:
    """Encode every stem path → (N, D) float32 array.

    Skips undecodable files (zero embedding), exactly like the MAEB harness.
    """
    embs: list[torch.Tensor] = []
    n_skipped = 0
    for p in tqdm(paths, desc="Encoding"):
        try:
            audio_item = {"path": p}
            wav = prepare_audio(
                audio_item,
                encoder.sampling_rate,
                encoder.audio_channels,
                encoder.max_audio_length_seconds,
            )
            emb = encoder._encode_item(wav)
        except AudioDecodeError as e:
            n_skipped += 1
            log.warning("Skipping %s: %s", p, e)
            emb = torch.zeros(encoder.embed_dim)
        embs.append(emb.cpu())
    if n_skipped:
        log.warning("Total skipped: %d", n_skipped)
    return torch.stack(embs).float().numpy()


def reduce_2d(X: np.ndarray, seed: int = 94) -> tuple[np.ndarray, str]:
    """Reduce (N, D) → (N, 2) with UMAP (falls back to t-SNE)."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, verbose=False)
        Z = reducer.fit_transform(X)
        method = "UMAP"
    except ImportError:
        log.warning("umap-learn not found — falling back to t-SNE (sklearn).")
        from sklearn.manifold import TSNE
        try:
            reducer = TSNE(n_components=2, random_state=seed, max_iter=1000)
        except TypeError:
            reducer = TSNE(n_components=2, random_state=seed, n_iter=1000)
        Z = reducer.fit_transform(X)
        method = "t-SNE"
    return Z, method


def make_figure(
    Z: np.ndarray,
    str_labels: list[str],
    int_labels: list[int],
    method: str,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    classes_sorted = sorted(_INSTRUMENT_CLASSES)   # alphabetical, matches int codes
    colour_map = {cls: _PALETTE[i] for i, cls in enumerate(classes_sorted)}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    counts = {cls: 0 for cls in classes_sorted}
    for cls in classes_sorted:
        mask = np.array([s == cls for s in str_labels])
        counts[cls] = int(mask.sum())
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=colour_map[cls],
            s=10, alpha=0.7, linewidths=0, rasterized=True,
        )

    patches = [
        mpatches.Patch(
            color=colour_map[cls],
            label=f"{cls} (n={counts[cls]})",
        )
        for cls in classes_sorted
    ]
    ax.legend(
        handles=patches,
        loc="best",
        framealpha=0.85,
        fontsize=8,
        title=f"{method} — 2D",
        title_fontsize=8,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "latent_umap_instruments"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Saved: %s", pdf_path)
    log.info("Saved: %s", png_path)
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/SAGE_e299.ckpt",
                   help="SAGE checkpoint path (default: checkpoints/SAGE_e299.ckpt)")
    p.add_argument("--out", default="evaluation/maeb/figures",
                   help="Output directory for PDF + PNG (default: evaluation/maeb/figures)")
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available, else cpu)")
    p.add_argument("--seed", type=int, default=94, help="UMAP random seed (default: 94)")
    p.add_argument("--max-audio-sec", type=float, default=30.0,
                   help="Max clip length in seconds, matching the MAEB default (default: 30)")
    p.add_argument("--embed-only", action="store_true",
                   help="Save embeddings to .npz and exit without running UMAP/plotting.")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # ── 1. File list (exact MAEB instrument task selection) ──────────────────
    paths, str_labels, int_labels = _build_file_list()
    classes_sorted = sorted(_INSTRUMENT_CLASSES)
    counts = {cls: str_labels.count(cls) for cls in classes_sorted}
    print(f"\nDataset: {len(paths)} stems, {len(classes_sorted)} classes")
    print(f"Chunk length used by harness: 30 s  (chunks_30s/ directory, not 10 s)")
    for cls in classes_sorted:
        print(f"  {cls:<15} {counts[cls]:>4} stems")

    # ── 2. Encode ─────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SwinEncoder(
        model_name=str(ckpt),
        device=device,
        max_audio_length_seconds=args.max_audio_sec,
        standardize_bottleneck=True,   # must match MAEB default
    )
    print(f"\nEncoder: embed_dim={encoder.embed_dim}, device={device}, "
          f"standardize_bottleneck=True")

    X = embed_stems(encoder, paths)
    print(f"Embeddings: {X.shape}")

    # ── 3. Save embeddings to .npz (so UMAP can be re-run locally) ───────────
    npz_path = Path(args.out) / "instruments_emb.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X,
        str_labels=np.array(str_labels),
        int_labels=np.array(int_labels),
    )
    print(f"Saved embeddings: {npz_path}")

    if args.embed_only:
        print("--embed-only: done. Transfer the .npz and run UMAP locally.")
        return

    # ── 4. 2-D projection ─────────────────────────────────────────────────────
    print("Running dimensionality reduction…")
    Z, method = reduce_2d(X, seed=args.seed)
    print(f"Method: {method}")

    # ── 4. Plot + save ────────────────────────────────────────────────────────
    make_figure(Z, str_labels, int_labels, method, Path(args.out))


if __name__ == "__main__":
    main()
