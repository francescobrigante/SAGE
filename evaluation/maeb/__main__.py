#!/usr/bin/env python3
# =============================================================================
# Semantic probing of a latent on the MAEB suite (paper Section 4):
#
#   python -m evaluation.maeb --encoder sage --checkpoint SAGE_FTe992.ckpt --with-moisesdb
#   python -m evaluation.maeb --encoder sage --checkpoint SAGE_FTe992.ckpt --maeb-original-music-only
#   python -m evaluation.maeb --encoder same-s --with-moisesdb
#   python -m evaluation.maeb --encoder clap --checkpoint music_audioset_epoch_15_esc_90.14.pt
#
# Suites: the 6 FMA tasks (default), + the 7 MoisesDB tasks (--with-moisesdb),
# or the 6 upstream MAEB music tasks (--maeb-original-music-only); runs into
# the same --output-dir merge, so the paper's 19 tasks are the first two calls.
# Encoders:
#   sage     deterministic latent μ (C × F_lat = 64-d, time-pooled)
#   <codec>  a baseline's deterministic latent at its native width (SAME: 256-d,
#            not width-matched — report its scores with that caveat)
#   clap     the frozen distillation teacher, an oracle reference row (512-d;
#            CLAP is not invertible, so it has no reconstruction counterpart)
# =============================================================================
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.maeb import compatibility  # noqa: F401  (must precede mteb / datasets)

import torch  # noqa: E402

from evaluation.maeb import tasks as maeb_tasks  # noqa: E402
from evaluation.maeb.runner import run_maeb  # noqa: E402

log = logging.getLogger(__name__)

BASELINES = ("codicodec", "music2latent", "sao-vae", "same", "same-s")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the MAEB suite with one encoder.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder", required=True, choices=("sage", "clap") + BASELINES)
    p.add_argument("--checkpoint", default=None,
                   help="sage: the SAGE checkpoint; clap: the LAION-CLAP teacher checkpoint")
    p.add_argument("--tasks", nargs="*", default=None,
                   help=f"Explicit task subset (overrides the suite flags). Allowed: {maeb_tasks.ALLOWED_TASKS}")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--maeb-original-music-only", action="store_true",
                     help=f"Only the upstream MAEB music tasks: {maeb_tasks.MAEB_ORIGINAL_MUSIC}")
    grp.add_argument("--moisesdb-only", action="store_true", help="Only the MoisesDB tasks")
    p.add_argument("--with-moisesdb", action="store_true", help="Add the 7 MoisesDB tasks to the FMA suite")
    p.add_argument("--max-files", type=int, default=0, help="Per-task sample cap (0 = all)")
    p.add_argument("--output-dir", default=None, help="Default: maeb_results/<encoder or checkpoint name>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-audio-sec", type=float, default=30.0, help="Clips are truncated to this length")
    p.add_argument("--overwrite", action="store_true", help="Recompute tasks that already have results")
    p.add_argument("--batch-size", type=int, default=1, help="sage: DataLoader batch size")
    p.add_argument("--standardize-bottleneck", action=argparse.BooleanOptionalAction, default=True,
                   help="sage: fold the latent frequency axis into channels (64-d) and pool time only")
    p.add_argument("--pooling", choices=["mean", "max"], default="mean", help="baselines: time pooling")
    p.add_argument("--no-l2-normalize", action="store_true", help="clap: keep the raw teacher embedding")
    args = p.parse_args(argv)
    if args.encoder in ("sage", "clap") and not args.checkpoint:
        p.error(f"--encoder {args.encoder} needs --checkpoint")
    return args


def _build_encoder(args: argparse.Namespace):
    """(MTEB encoder, summary fields) for the requested encoder."""
    if args.encoder == "sage":
        from evaluation.maeb.sage_encoder import SAGELatentEncoder, build_sage_model_meta
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = SAGELatentEncoder(model_name=str(args.checkpoint), device=args.device,
                                  max_audio_length_seconds=args.max_audio_sec,
                                  standardize_bottleneck=args.standardize_bottleneck)
        model.mteb_model_meta = build_sage_model_meta(str(args.checkpoint), model)
        return model, {"ckpt": str(args.checkpoint), "is_complex": model.is_complex,
                       "embed_dim": model.embed_dim, "standardize_bottleneck": model.standardize_bottleneck,
                       "f_lat": model.f_lat}
    if args.encoder == "clap":
        from evaluation.maeb.clap_encoder import CLAPOracleEncoder, build_clap_model_meta
        model = CLAPOracleEncoder(ckpt_path=args.checkpoint, device=args.device,
                                  max_audio_length_seconds=args.max_audio_sec,
                                  l2_normalize=not args.no_l2_normalize)
        model.mteb_model_meta = build_clap_model_meta(model)
        return model, {"model": "clap-oracle-teacher",
                       "role": "reference (not a baseline: CLAP is not invertible)",
                       "ckpt": args.checkpoint, "embed_dim": model.embed_dim,
                       "pooling": "mean over 10s windows", "l2_normalize": model.l2_normalize,
                       "caveats": ["512-d vs SAGE 64-d: not width-matched.",
                                   "CLAP is contrastively pretrained on AudioSet + captioned music; "
                                   "GTZAN/MusicGenre/NSynth are plausibly in-distribution for it: "
                                   "the oracle is an optimistic upper bound."]}
    from evaluation.maeb.sota_encoder import SOTACodecEncoder, build_model_meta
    model = SOTACodecEncoder(model_name=args.encoder, device=args.device,
                             max_audio_length_seconds=args.max_audio_sec, pooling=args.pooling)
    model.mteb_model_meta = build_model_meta(args.encoder, model)
    return model, {"model": args.encoder, "embed_dim": model.embed_dim, "pooling": args.pooling}


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args(argv)
    name = Path(args.checkpoint).stem if args.encoder == "sage" else args.encoder
    output_dir = Path(args.output_dir) if args.output_dir else Path("maeb_results") / name

    names = maeb_tasks.select_task_names(args.tasks, maeb_original_music_only=args.maeb_original_music_only,
                                         with_moisesdb=args.with_moisesdb, moisesdb_only=args.moisesdb_only)
    tasks = maeb_tasks.get_tasks_by_name(names, encoder_label=f"MAEB encoder '{args.encoder}'",
                                         max_files=args.max_files)
    if not tasks:
        raise ValueError("No tasks resolved. Check --tasks.")
    log.info("MAEB tasks: %s (max_files=%d)", [t.metadata.name for t in tasks], args.max_files)

    model, summary = _build_encoder(args)
    batch_size = args.batch_size if args.encoder == "sage" else 1
    run_maeb(model, tasks, output_dir, encode_kwargs={"batch_size": batch_size}, overwrite=args.overwrite,
             summary_extra={**summary, "max_files": args.max_files})


if __name__ == "__main__":
    main()
