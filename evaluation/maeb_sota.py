#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb_sota.py
# CLI entry point: run the FMA-local MAEB suite with a SOTA baseline codec
# (codicodec | music2latent) via the shared CodecAdapter. Thin wrapper — all
# logic lives in evaluation/maeb/. Run from the ISOLATED maeb_dl venv.
# =============================================================================
"""
MAEB evaluation for SOTA baseline codecs (codicodec, music2latent), FMA suite.

Embedding: deterministic latent, single per-clip forward, time-pooled → 64-d
(matches the SAO 64-d protocol). SAME is excluded (256-d latent).

Usage:
  python evaluation/maeb_sota.py --model codicodec
  python evaluation/maeb_sota.py --model music2latent --max-files 2000 --overwrite
"""
from __future__ import annotations

import sys
sys.stderr.write("MAEB SOTA: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import logging
from pathlib import Path

# compatibility MUST be the first project import (routes HF cache + datasets alias).
sys.path.insert(0, str(Path(__file__).resolve().parent))  # put evaluation/ on path
from maeb import compatibility  # noqa: F401

import torch

from maeb import tasks as maeb_tasks
from maeb.runner import run_maeb

log = logging.getLogger(__name__)

_ALLOWED_MODELS = ("codicodec", "music2latent", "sao-vae")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the FMA-local MAEB suite with a SOTA baseline codec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, choices=_ALLOWED_MODELS,
                   help="Adapter key (SAME excluded: 256-d latent).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help=f"Explicit task subset. Allowed: {maeb_tasks.ALLOWED_TASKS}.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--with-extra", action="store_true",
                     help=f"Add EXTRA music tasks to the FMA suite: {maeb_tasks.EXTRA_TASKS}.")
    grp.add_argument("--extra-only", action="store_true", help="Run ONLY the EXTRA tasks.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Per-FMA-task sample cap: min(task_samples, max_files). 0 = all.")
    p.add_argument("--output-dir", default=None,
                   help="Results dir. Default: runs/maeb_<model>.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-audio-sec", type=float, default=30.0)
    p.add_argument("--pooling", choices=["mean", "max"], default="mean")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-task results. Default: resume from disk.")
    return p.parse_args()


def _resolve_tasks(args: argparse.Namespace, encoder_label: str) -> list:
    names = maeb_tasks.select_task_names(
        args.tasks, with_extra=args.with_extra, extra_only=args.extra_only,
    )
    tasks = maeb_tasks.get_tasks_by_name(
        names, encoder_label=encoder_label, max_files=args.max_files,
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
    args = parse_args()
    log.info("MAEB SOTA evaluation starting — model=%s", args.model)

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"maeb_{args.model}"

    encoder_label = f"SOTACodecEncoder.{args.model}"
    tasks = _resolve_tasks(args, encoder_label)
    if not tasks:
        raise ValueError("No tasks resolved. Check --tasks.")

    from maeb.sota_encoder import SOTACodecEncoder, build_model_meta
    model = SOTACodecEncoder(
        model_name=args.model,
        device=args.device,
        max_audio_length_seconds=args.max_audio_sec,
        pooling=args.pooling,
    )
    model.mteb_model_meta = build_model_meta(args.model, model)

    run_maeb(
        model, tasks, output_dir,
        encode_kwargs={"batch_size": 1},
        overwrite=args.overwrite,
        summary_extra={
            "model": args.model,
            "embed_dim": model.embed_dim,
            "pooling": args.pooling,
            "max_files": args.max_files,
        },
    )


if __name__ == "__main__":
    main()
