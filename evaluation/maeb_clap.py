#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb_clap.py
# CLI entry point: run the MAEB suite with the FROZEN LAION-CLAP teacher as a
# reference "oracle" row. Thin wrapper — all logic lives in evaluation/maeb/.
# Run from the ISOLATED maeb_dl venv (laion_clap 1.1.7 already installed there).
# =============================================================================
"""
MAEB evaluation of the frozen distillation teacher (CLAP-oracle reference row).

This is a REFERENCE, not a baseline: CLAP is not invertible and reconstructs no
audio, so it never appears on the fidelity axis. It answers one question only —
how much of the teacher's semantic power survives inside SAGE's 64-d invertible
latent at x64 compression.

Checkpoint: the exact one distilled in training (music_audioset_epoch_15,
see loss_manager.py:391). Embedding: 512-d, mean of per-10s-window CLAP
embeddings (deterministic; see clap_encoder.py for why windowing is required).

Caveats to tabulate alongside: 512-d vs 64-d (not width-matched), and CLAP's
contrastive pretraining plausibly covers GTZAN / MusicGenre / NSynth, which
SAGE never saw labelled — both make the oracle an OPTIMISTIC upper bound.

Usage:
  python evaluation/maeb_clap.py --tasks FMAGenreClassification --max-files 200
  python evaluation/maeb_clap.py --with-moisesdb
  python evaluation/maeb_clap.py --maeb-original-music-only
"""
from __future__ import annotations

import sys
sys.stderr.write("MAEB CLAP-oracle: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import logging
import os
from pathlib import Path

# compatibility MUST be the first project import (routes HF cache + datasets alias).
sys.path.insert(0, str(Path(__file__).resolve().parent))  # put evaluation/ on path
from maeb import compatibility  # noqa: F401

import torch

from maeb import tasks as maeb_tasks
from maeb.runner import run_maeb

log = logging.getLogger(__name__)

# The teacher checkpoint, mirroring config.MODELS_DIR / "LAION_CLAP" / ... .
_DEFAULT_CKPT = (
    Path(os.environ.get("FAST", "/leonardo_scratch/fast/IscrC_AHNetBio"))
    / "models" / "LAION_CLAP" / "music_audioset_epoch_15_esc_90.14.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the MAEB suite with the frozen CLAP teacher (oracle row).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", default=str(_DEFAULT_CKPT),
                   help="LAION-CLAP checkpoint. Default: the distilled teacher.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help=f"Explicit task subset. Allowed: {maeb_tasks.ALLOWED_TASKS}.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--maeb-original-music-only", action="store_true",
                     help=f"Run ONLY the upstream MAEB music tasks: "
                          f"{maeb_tasks.MAEB_ORIGINAL_MUSIC}.")
    grp.add_argument("--moisesdb-only", action="store_true",
                     help="Run ONLY the MoisesDB tasks (chunks_30s).")
    p.add_argument("--with-moisesdb", action="store_true",
                   help="Add the 7 MoisesDB tasks (chunks_30s) to the default suite.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Per-FMA-task sample cap: min(task_samples, max_files). 0 = all.")
    p.add_argument("--output-dir", default=None,
                   help="Results dir. Default: runs/final_eval/maeb/maeb_clap_oracle.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-audio-sec", type=float, default=30.0)
    p.add_argument("--no-l2-normalize", action="store_true",
                   help="Skip re-normalizing the window-averaged embedding.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-task results. Default: resume from disk.")
    return p.parse_args()


def _resolve_tasks(args: argparse.Namespace, encoder_label: str) -> list:
    names = maeb_tasks.select_task_names(
        args.tasks,
        maeb_original_music_only=args.maeb_original_music_only,
        with_moisesdb=args.with_moisesdb,
        moisesdb_only=args.moisesdb_only,
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
    log.info("MAEB CLAP-oracle evaluation starting — ckpt=%s", args.ckpt)

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path("runs") / "final_eval" / "maeb" / "maeb_clap_oracle")

    encoder_label = "CLAPOracleEncoder.teacher"
    tasks = _resolve_tasks(args, encoder_label)
    if not tasks:
        raise ValueError("No tasks resolved. Check --tasks.")

    from maeb.clap_encoder import CLAPOracleEncoder, build_clap_model_meta
    model = CLAPOracleEncoder(
        ckpt_path=args.ckpt,
        device=args.device,
        max_audio_length_seconds=args.max_audio_sec,
        l2_normalize=not args.no_l2_normalize,
    )
    model.mteb_model_meta = build_clap_model_meta(model)

    run_maeb(
        model, tasks, output_dir,
        encode_kwargs={"batch_size": 1},
        overwrite=args.overwrite,
        summary_extra={
            "model": "clap-oracle-teacher",
            "role": "reference (not a baseline: CLAP is not invertible)",
            "ckpt": args.ckpt,
            "embed_dim": model.embed_dim,
            "pooling": "mean over 10s windows",
            "l2_normalize": model.l2_normalize,
            "max_files": args.max_files,
            "caveats": [
                "512-d vs SAGE 64-d: NOT width-matched (cf. SAME 256-d caveat).",
                "CLAP is contrastively pretrained on AudioSet + captioned music; "
                "GTZAN/MusicGenre/NSynth are plausibly in-distribution for it and "
                "are not for SAGE. The oracle is an OPTIMISTIC upper bound.",
            ],
        },
    )


if __name__ == "__main__":
    main()
