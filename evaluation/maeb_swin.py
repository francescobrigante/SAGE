#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb_swin.py
# CLI entry point: run the FMA-local MAEB suite with the Swin C-VAE encoder.
# Thin wrapper — all logic lives in evaluation/maeb/.
# =============================================================================
"""
MAEB evaluation for Swin C-VAE (complex and real) checkpoints, on the FMA suite.

Runs the 6 FMA music tasks (default) on the FMA large test set; le task musicali
MAEB upstream sui dataset originari (GTZAN/NSynth/JamAlt/…) sono opt-in via
--maeb-original-music-only. Run from the ISOLATED maeb_dl venv (see CINECA.md §5.4).
complextorch/complexpytorch are pure-Python on PyTorch 2.4.1 complex tensors —
no cineca-custom torch needed.

Usage examples:
  # Default: solo le 6 FMA:
  python evaluation/maeb_swin.py --ckpt checkpoints/swin_cplx_4s_x64/best.ckpt

  # SOLO le task MAEB musicali sui dataset originari (estende un summary FMA
  # già valutato — merge non-distruttivo):
  python evaluation/maeb_swin.py --ckpt model.ckpt --maeb-original-music-only

  # A subset of the suite, with a small per-task cap for a quick run:
  python evaluation/maeb_swin.py --ckpt model.ckpt \
      --tasks FMAGenreClustering FMAArtistPairClassification --max-files 200
"""
from __future__ import annotations

import sys
sys.stderr.write("MAEB Swin C-VAE: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import logging
from pathlib import Path

# compatibility MUST be the first project import: it routes the HF cache and
# registers the datasets 'List' alias BEFORE mteb / datasets are imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # put evaluation/ on path
from maeb import compatibility  # noqa: E402 — side-effect import

import torch  # noqa: E402

from maeb import tasks as maeb_tasks  # noqa: E402
from maeb.runner import run_maeb     # noqa: E402

log = logging.getLogger(__name__)

_ENCODER_LABEL = "SwinEncoder"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the FMA-local MAEB suite with a Swin C-VAE checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", default="checkpoints/O1_real_same_mrstft.ckpt",
                   help="Path to the Swin C-VAE .ckpt. Default: checkpoints/O1_real_same_mrstft.ckpt")
    p.add_argument("--tasks", nargs="*", default=None,
                   help=f"Explicit task subset (overrides the -only flags). "
                        f"Allowed: {maeb_tasks.ALLOWED_TASKS}.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--maeb-original-music-only", action="store_true",
                     help=f"Esegue SOLO le task MAEB musicali upstream sui dataset "
                          f"originari: {maeb_tasks.MAEB_ORIGINAL_MUSIC}.")
    grp.add_argument("--moisesdb-only", action="store_true",
                     help="Esegue SOLO i task MoisesDB (chunks_30s).")
    p.add_argument("--with-moisesdb", action="store_true",
                   help="Aggiunge i 7 task MoisesDB (chunks_30s) alla suite di default.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Per-FMA-task sample cap: min(task_samples, max_files). 0 = all.")
    p.add_argument("--output-dir", default=None,
                   help="Results directory. Default: ./maeb_results/<ckpt_stem>.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device. Default: cuda if available, else cpu.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="DataLoader batch size. Default: 1 (clips have variable length).")
    p.add_argument("--max-audio-sec", type=float, default=30.0,
                   help="Max clip length in seconds (truncated before encoding). Default: 30.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-task results. Default: resume from disk.")
    p.add_argument("--standardize-bottleneck", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Fold the latent freq axis into channels (C×F_lat=64) and pool "
                        "time only, matching SAO's protocol (DEFAULT — it wins on FMA). "
                        "Use --no-standardize-bottleneck for the legacy freq+time pool "
                        "(16-dim). Standardized runs land under a distinct '__std' name.")
    return p.parse_args()


def _resolve_tasks(args: argparse.Namespace) -> list:
    """Resolve the selected task names into objects."""
    names = maeb_tasks.select_task_names(
        args.tasks,
        maeb_original_music_only=args.maeb_original_music_only,
        with_moisesdb=args.with_moisesdb,
        moisesdb_only=args.moisesdb_only,
    )
    tasks = maeb_tasks.get_tasks_by_name(
        names, encoder_label=_ENCODER_LABEL, max_files=args.max_files,
    )
    log.info("MAEB tasks: %s (max_files=%d)",
             [t.metadata.name for t in tasks], args.max_files)
    return tasks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("MAEB Swin C-VAE evaluation starting.")
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("maeb_results") / ckpt_path.stem
    )

    tasks = _resolve_tasks(args)
    if not tasks:
        raise ValueError("No tasks resolved. Check --tasks.")

    # Import lazily — pulls sage + complextorch (heavy first time).
    from maeb.swin_encoder import SwinEncoder, build_swin_model_meta
    model = SwinEncoder(
        model_name=str(ckpt_path),
        device=args.device,
        max_audio_length_seconds=args.max_audio_sec,
        standardize_bottleneck=args.standardize_bottleneck,
    )
    model.mteb_model_meta = build_swin_model_meta(str(ckpt_path), model)

    run_maeb(
        model, tasks, output_dir,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite=args.overwrite,
        summary_extra={
            "ckpt": str(ckpt_path),
            "is_complex": model.is_complex,
            "embed_dim": model.embed_dim,
            "standardize_bottleneck": model.standardize_bottleneck,
            "f_lat": model.f_lat,
            "max_files": args.max_files,
        },
    )


if __name__ == "__main__":
    main()
