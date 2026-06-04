#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb_sao.py
# CLI entry point: run the FMA-local MAEB suite with the SAO-ACE encoder.
# Thin wrapper — all logic lives in evaluation/maeb/.
# =============================================================================
"""
MAEB evaluation for the Stable Audio Open ACE (SAO-ACE) autoencoder, FMA suite.

Runs the 6 FMA music tasks + NMSQAPairClassification on the FMA large test set.
Run from the ISOLATED maeb_dl venv (see CINECA.md §5.4); SAO is a plain conv
VAE and does NOT need cineca's custom torch.

Usage examples:
  # Full FMA suite, config from ckpt:
  python evaluation/maeb_sao.py --ckpt /path/to/alrurt2n_4330k.ckpt

  # Checkpoint without embedded model_config → pass the ACE JSON explicitly:
  python evaluation/maeb_sao.py --ckpt model.ckpt --model-config ace_vae.json

  # A subset of the suite with a quick cap:
  python evaluation/maeb_sao.py --ckpt model.ckpt \
      --tasks FMAGenreClassification --max-files 200 --batch-size 8

Embedding: deterministic VAE mean, single forward (no overlap-add), pooled over
valid latent frames. --batch-size 1 recovers strict per-clip encoding.
"""
from __future__ import annotations

import sys
sys.stderr.write("MAEB SAO-ACE: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import logging
from pathlib import Path

# compatibility MUST be the first project import: it routes the HF cache and
# registers the datasets 'List' alias BEFORE mteb / datasets are imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # put evaluation/ on path
from maeb import compatibility

import torch

from maeb import tasks as maeb_tasks
from maeb.runner import run_maeb

log = logging.getLogger(__name__)

_ENCODER_LABEL = "SAOACEEncoder"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the FMA-local MAEB suite with the SAO-ACE encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to the SAO-ACE .ckpt.")
    p.add_argument("--model-config", default=None,
                   help="ACE model JSON. Only needed if the ckpt lacks an embedded model_config.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help=f"Subset of the FMA suite to run. Default: all. "
                        f"Allowed: {maeb_tasks.ALLOWED_TASKS}.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Per-FMA-task sample cap: min(task_samples, max_files). 0 = all.")
    p.add_argument("--output-dir", default=None,
                   help="Results directory. Default: ./maeb_results/<ckpt_stem>.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device. Default: cuda if available, else cpu.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Encoding batch size. Default: 16 (=1 recovers strict per-clip).")
    p.add_argument("--max-audio-sec", type=float, default=30.0,
                   help="Max clip length in seconds. Default: 30.")
    p.add_argument("--pooling", choices=["mean", "max"], default="mean",
                   help="Temporal pooling over valid latent frames. Default: mean.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-task results. Default: resume from disk.")
    return p.parse_args()


def _resolve_tasks(args: argparse.Namespace) -> list:
    """Resolve the FMA suite (or a --tasks subset of it) into task objects."""
    names = args.tasks or (maeb_tasks.FMA_SUITE + maeb_tasks.FMA_SUITE_EXTRA)
    tasks = maeb_tasks.get_tasks_by_name(
        names, encoder_label=_ENCODER_LABEL, max_files=args.max_files,
    )
    log.info("FMA suite: %s (max_files=%d)",
             [t.metadata.name for t in tasks], args.max_files)
    return tasks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("MAEB SAO-ACE evaluation starting.")
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

    # Import the encoder lazily — it pulls stable_audio_tools (heavy).
    from maeb.sao_encoder import SAOACEEncoder, build_model_meta
    model = SAOACEEncoder(
        model_name=str(ckpt_path),
        device=args.device,
        model_config_path=args.model_config,
        max_audio_length_seconds=args.max_audio_sec,
        pooling=args.pooling,
    )
    model.mteb_model_meta = build_model_meta(str(ckpt_path), model)

    run_maeb(
        model, tasks, output_dir,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite=args.overwrite,
        summary_extra={
            "ckpt": str(ckpt_path),
            "embed_dim": model.embed_dim,
            "pooling": args.pooling,
            "max_files": args.max_files,
        },
    )


if __name__ == "__main__":
    main()
