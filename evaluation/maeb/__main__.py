#!/usr/bin/env python3
# =============================================================================
# Semantic probing of a latent on the MAEB tasks (paper Section 4, Tables 4 and 9):
#
#   python -m evaluation.maeb encoder=sage                  # the paper's 19 tasks
#   python -m evaluation.maeb encoder=same-s                # a baseline
#   python -m evaluation.maeb encoder=clap                  # CLAP oracle row
#
# Settings: configs/maeb.yaml (suite, pooling, ...); data and weights: configs/paths/.
# Encoders:
#   sage     deterministic latent μ (C × F_lat = 64-d, time-pooled)
#   <codec>  a baseline's deterministic latent at its native width (SAME: 256-d,
#            not width-matched — report its scores with that caveat)
#   clap     the frozen distillation teacher, an oracle reference row (512-d;
#            CLAP is not invertible, so it has no reconstruction counterpart)
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path

from evaluation.maeb import compatibility  # noqa: F401  (must precede mteb / datasets)

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from evaluation.maeb import fma_tasks, moisesdb_tasks  # noqa: E402
from evaluation.maeb import tasks as maeb_tasks  # noqa: E402
from evaluation.maeb.runner import run_maeb  # noqa: E402

log = logging.getLogger(__name__)

BASELINES = ("codicodec", "music2latent", "sao-vae", "same", "same-s")


def _args_from_cfg(cfg: DictConfig):
    """The settings of configs/maeb.yaml, checked; the checkpoint defaults to paths.*."""
    from types import SimpleNamespace
    if cfg.encoder not in ("sage", "clap") + BASELINES:
        raise SystemExit(f"encoder={cfg.encoder}: not one of {('sage', 'clap') + BASELINES}")
    checkpoint = cfg.checkpoint or {"sage": cfg.paths.sage_checkpoint, "clap": cfg.paths.clap_teacher}.get(cfg.encoder)
    if cfg.encoder in BASELINES and cfg.checkpoint:
        raise SystemExit(f"checkpoint applies to encoder=sage or clap only, not {cfg.encoder}")
    if checkpoint and not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint} (set checkpoint=... or configs/paths)")
    device = cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    return SimpleNamespace(encoder=cfg.encoder, checkpoint=str(checkpoint) if checkpoint else None, device=device,
                           max_audio_sec=float(cfg.max_audio_sec), standardize_bottleneck=bool(cfg.standardize_bottleneck),
                           pooling=cfg.pooling, no_l2_normalize=not cfg.l2_normalize,
                           adapter_kwargs={"model_dir": cfg.paths.sao_vae} if cfg.encoder == "sao-vae" else {})


def _build_encoder(args):
    """(MTEB encoder, summary fields) for the requested encoder."""
    if args.encoder == "sage":
        from evaluation.maeb.sage_encoder import SAGELatentEncoder, build_sage_model_meta
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
                             max_audio_length_seconds=args.max_audio_sec, pooling=args.pooling,
                             adapter_kwargs=args.adapter_kwargs)
    model.mteb_model_meta = build_model_meta(args.encoder, model)
    return model, {"model": args.encoder, "embed_dim": model.embed_dim, "pooling": args.pooling}


def run(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = _args_from_cfg(cfg)
    fma_tasks.FMA_METADATA, fma_tasks.FMA_AUDIO = cfg.paths.fma_metadata, cfg.paths.fma_audio
    moisesdb_tasks.MOISESDB_ROOT, moisesdb_tasks.MOISESDB_CHUNKS = cfg.paths.moisesdb_root, cfg.paths.moisesdb_chunks
    name = Path(args.checkpoint).stem if args.encoder == "sage" else args.encoder
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.paths.eval_output) / "maeb" / name

    names = maeb_tasks.select_task_names(list(cfg.tasks) if cfg.tasks else None, suite=cfg.suite)
    tasks = maeb_tasks.get_tasks_by_name(names, encoder_label=f"MAEB encoder '{args.encoder}'",
                                         max_files=int(cfg.max_files))
    if not tasks:
        raise ValueError("No tasks resolved. Check tasks=.")
    log.info("MAEB tasks: %s (max_files=%d)", [t.metadata.name for t in tasks], cfg.max_files)

    model, summary = _build_encoder(args)
    batch_size = int(cfg.batch_size) if args.encoder == "sage" else 1
    run_maeb(model, tasks, output_dir, encode_kwargs={"batch_size": batch_size}, overwrite=bool(cfg.overwrite),
             summary_extra={**summary, "max_files": int(cfg.max_files)})


@hydra.main(config_path="../../configs", config_name="maeb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
