#!/usr/bin/env python3
# =============================================================================
# Reconstruction metrics of one codec on one evaluation set (paper Table 2):
# SDR / SI-SDR, STFT and mel distances, CLAP cosine (music, audio), FAD-MERT,
# FAD-CLAP and FAD-PANN, plus CDPAM and, on request, the stereo-image metrics
# of Table 7. SAGE and every baseline go through the same code (codecs.py).
#
#   python -m evaluation.reconstruction dataset=fma model=sage
#   python -m evaluation.reconstruction dataset=musiccaps model=same-s
#
# Settings: configs/reconstruction.yaml, the evaluation sets in configs/dataset/,
# data and weights in configs/paths/. Protocols (set by the dataset):
#   fma    FMA test split, full-length tracks. Target embeddings are computed on
#          the fly and cached in the dataset's cache_dir, shared across models so
#          their FADs are measured against the same reference.
#   clips  a flat folder of 10 s clips (MoisesDB mixtures/stems, MusicCaps, Song
#          Describer) whose references were built once by evaluation.build_references.
#
# Scaling: under SLURM each task (SLURM_PROCID of SLURM_NTASKS) scores
# files[rank::world] on GPU SLURM_LOCALID and writes per-file rows as it goes;
# a relaunch resumes, and `merge=true` (automatic when there is one task) writes the
# final CSVs and the FADs. Outputs: <output_dir>/<name>/metrics/*.csv.
# =============================================================================
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.codecs import ADAPTERS, build_adapter
from evaluation.common import (CLAP_AUDIO_NAME, CLAP_GUD_NAME, CLAP_MUSIC_NAME, MERT_NAME, AudioDataset, append_csv,
                               atomic_save_npy, collect_clip_files, collect_fma_files, file_seed, first_item, load_or_embed,
                               seed_everything, set_metric_weights, silence_output, target_cache_path,
                               write_csv)
from evaluation.metrics.clap import cosine_sim, embed_clap, embed_clap_gud
from evaluation.metrics.fad import (PANN_NAME, compute_fad_from_embeddings, compute_incremental_stats,
                                    embed_mert_framewise, embed_pann, get_pann_model)
from evaluation.metrics.signal import cdpam_score, compute_sdr_and_sisdr, spectral_losses
from evaluation.metrics.stereo import MS_COLUMNS, ms_metrics_row
from sage.utils.console import err, info, ok, warn

# numpy 1.24+ removed np.float, which CDPAM still uses internally
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

# Per-file CSVs (appended per rank under metrics/parts/, merged into metrics/).
CSV_SCHEMA = {
    "spectral":   ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"],
    "cdpam":      ["file", "cdpam"],
    "clap_music": ["file", "cosine"],
    "clap_audio": ["file", "cosine"],
    "ms_metrics": MS_COLUMNS,               # only with compute_ms_metrics=true
}
# FAD embedding → output CSV.
FAD_CSV = {MERT_NAME: "fad_mert.csv", CLAP_GUD_NAME: "fad_gudgud.csv", PANN_NAME: "fad_pann.csv"}


# ── Data ─────────────────────────────────────────────────────────────────────

def _load_done(parts: Path) -> set[str]:
    """Stems already scored by any rank (resume)."""
    done: set[str] = set()
    for f in parts.glob("done.*.txt"):
        done |= {s for s in f.read_text().split() if s}
    return done


# Run settings that change the numbers: a run resumed into the same folder must keep them.
_MANIFEST_KEYS = ("model", "checkpoint", "varlen", "deterministic", "protocol", "data_dir", "fma_csv",
                  "max_files", "skip_cdpam", "sdr_only", "compute_ms_metrics", "seed")


def _check_manifest(parts: Path, args: SimpleNamespace, rank: int = 0, timeout: float = 600.0) -> None:
    """Write the run settings on the first launch; refuse to resume into a folder made with others.

    Every task of a sharded run starts at once: only rank 0 writes the manifest, atomically
    (the other ranks never see a half-written file), and the others wait for it."""
    current = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k in _MANIFEST_KEYS}
    path = parts / "run.json"
    if rank == 0 and not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(current, indent=2) + "\n")
        os.replace(tmp, path)
        return
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"{path} was not written by rank 0 within {timeout:.0f} s")
        time.sleep(1.0)
    saved = json.loads(path.read_text())
    diff = {k: (saved.get(k), current[k]) for k in current if saved.get(k) != current[k]}
    if diff:
        raise SystemExit(f"{path.parent.parent} holds a run with other settings {diff} (saved, requested); "
                         "use another output_dir or name.")


def _load_ref_emb(data_dir: Path, model_name: str, stem: str) -> Optional[np.ndarray]:
    """Reference embedding of a clip, written by evaluation.build_references."""
    p = data_dir / "embeddings" / model_name / f"{stem}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


# ── Settings ─────────────────────────────────────────────────────────────────

def _args_from_cfg(cfg: DictConfig) -> SimpleNamespace:
    """The run settings of configs/reconstruction.yaml, checked, as one flat namespace."""
    if cfg.get("dataset") is None:
        raise SystemExit("dataset=<name> is required: fma | moisesdb_mix | moisesdb_stems | musiccaps | song_describer")
    if cfg.model not in ADAPTERS:
        raise SystemExit(f"model={cfg.model}: not one of {sorted(ADAPTERS)}")
    ds = cfg.dataset
    if not ds.data_dir:
        raise SystemExit(f"dataset={ds.name}: its data folder is not set (configs/paths, key used by "
                         f"configs/dataset/{ds.name}.yaml)")
    if cfg.model == "sage":
        checkpoint = Path(cfg.checkpoint or cfg.paths.sage_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f"SAGE checkpoint not found: {checkpoint} (set checkpoint=... or paths.sage_checkpoint)")
    elif cfg.checkpoint or cfg.varlen or cfg.deterministic:
        raise SystemExit("checkpoint, varlen and deterministic apply to model=sage only")
    else:
        checkpoint = None
    return SimpleNamespace(
        model=cfg.model, checkpoint=checkpoint, varlen=cfg.varlen, deterministic=bool(cfg.deterministic),
        adapter_kwargs={"model_dir": cfg.paths.sao_vae} if cfg.model == "sao-vae" else {},
        protocol=ds.protocol, data_dir=Path(ds.data_dir), fma_csv=ds.fma_csv,
        extensions=ds.extensions or ".wav,.flac,.mp3,.ogg",
        cache_dir=Path(ds.cache_dir) if ds.cache_dir else None,
        output_dir=Path(cfg.output_dir), name=cfg.name, device=cfg.device, num_workers=int(cfg.num_workers),
        max_files=int(cfg.max_files), skip_cdpam=bool(cfg.skip_cdpam), sdr_only=bool(cfg.sdr_only),
        compute_ms_metrics=bool(cfg.compute_ms_metrics), seed=int(cfg.seed), resume=bool(cfg.resume),
        merge=bool(cfg.merge))


# ── Merge: per-rank rows → final CSVs, embeddings → FAD ──────────────────────

def _fad_from_stats(data_dir: Path, model_name: str, pred_emb_dir: Path, csv_path: Path) -> None:
    """FAD of the prediction embeddings against the reference statistics of a clip set."""
    stats_root = data_dir / "stats_ours" if (data_dir / "stats_ours" / model_name / "mu.npy").exists() \
        else data_dir / "stats"                    # stats/: statistics shipped with the set, if any
    ref_mu_path, ref_cov_path = stats_root / model_name / "mu.npy", stats_root / model_name / "cov.npy"
    if not ref_mu_path.exists() or not ref_cov_path.exists():
        warn(f"No reference stats for {model_name} — FAD skipped.", prefix="MERGE")
        return
    npys = sorted(pred_emb_dir.glob("*.npy")) if pred_emb_dir.is_dir() else []
    if not npys:
        warn(f"No pred embeddings for {model_name} — FAD skipped.", prefix="MERGE")
        return
    from fadtk.fad import calc_frechet_distance
    ref_mu, ref_cov = np.load(ref_mu_path).astype(np.float64), np.load(ref_cov_path).astype(np.float64)
    pred_mu, pred_cov = compute_incremental_stats(npys)
    if pred_mu is None:
        pred_mu, pred_cov = np.zeros_like(ref_mu), np.zeros_like(ref_cov)
    score = calc_frechet_distance(ref_mu, ref_cov, pred_mu, pred_cov)
    ok(f"FAD {model_name}: {score:.6f}", prefix="MERGE")
    write_csv(csv_path, ["model", "score"], [{"model": model_name, "score": score}])


def _merge(metrics_dir: Path, args: SimpleNamespace) -> None:
    parts = metrics_dir / "parts"
    _check_manifest(parts, args)
    n_done = len(_load_done(parts))
    for name, cols in CSV_SCHEMA.items():
        seen: dict[str, dict] = {}
        for f in sorted(parts.glob(f"{name}.*.csv")):
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    seen.setdefault(r["file"], dict(r))           # dedup by file, keep the first row
        if seen:
            write_csv(metrics_dir / f"{name}.csv", cols, list(seen.values()))
            ok(f"{name}.csv ({len(seen)} files)", prefix="MERGE")

    if not args.sdr_only:
        for emb_name, csv_name in FAD_CSV.items():
            emb_dir = parts / "pred" / emb_name
            if args.protocol == "clips":
                _fad_from_stats(args.data_dir.expanduser().resolve(), emb_name, emb_dir, metrics_dir / csv_name)
                continue
            npys = sorted(emb_dir.glob("*.npy")) if emb_dir.is_dir() else []
            if npys and len(npys) != n_done:
                warn(f"{emb_name}: {len(npys)} embeddings for {n_done} scored files", prefix="MERGE")
            if args.cache_dir is None or not npys:
                warn(f"No {'cache dir' if args.cache_dir is None else 'pred embeddings'} — "
                     f"{csv_name} skipped.", prefix="MERGE")
                continue
            compute_fad_from_embeddings(SimpleNamespace(name=emb_name), [Path(f.stem) for f in npys], npys,
                                        args.cache_dir.expanduser().resolve(), metrics_dir / csv_name, emb_name)
    (metrics_dir / "_done").touch()
    ok(f"_done → {metrics_dir}", prefix="MERGE")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(cfg: DictConfig) -> None:
    set_metric_weights(cfg.paths)
    args = _args_from_cfg(cfg)
    name = args.name or (args.checkpoint.stem if args.model == "sage" else args.model)
    metrics_dir = args.output_dir.expanduser().resolve() / name / "metrics"
    if args.merge:
        if not (metrics_dir / "parts").is_dir():
            raise SystemExit(f"nothing to merge under {metrics_dir}")
        _merge(metrics_dir, args)
        return
    if args.resume and (metrics_dir / "_done").exists():
        info("[RESUME] already done — skipping.", prefix="EVAL")
        return

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    device = (torch.device("cpu") if args.device == "cpu" or not torch.cuda.is_available()
              else torch.device(f"cuda:{local}"))
    if device.type == "cuda":
        torch.cuda.set_device(local)    # fadtk/laion embedders use the *current* CUDA device

    data_dir = args.data_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    if args.protocol == "fma":
        exts = {(e if e.startswith(".") else f".{e}").lower() for e in args.extensions.split(",")}
        audio_files = collect_fma_files(data_dir, exts, args.fma_csv, args.max_files)
    else:
        if cache_dir is not None:
            warn("cache_dir is ignored by the clips protocol (references live in the data folder).", prefix="EVAL")
            cache_dir = None
        audio_files = collect_clip_files(data_dir, args.max_files)
    if not audio_files:
        raise SystemExit(f"No audio files found in {data_dir}")

    parts_dir = metrics_dir / "parts"
    pred_root = parts_dir / "pred"
    parts_dir.mkdir(parents=True, exist_ok=True)
    _check_manifest(parts_dir, args, rank)
    done = _load_done(parts_dir)
    my_files = [f for f in audio_files[rank::world] if f.stem not in done]
    done_file = parts_dir / f"done.{rank}.txt"
    info(f"Rank {rank}/{world} on {device} — {len(my_files)} files "
         f"(skip {len(audio_files[rank::world]) - len(my_files)} done)", prefix="EVAL")

    adapter = build_adapter(args.model, device=str(device), checkpoint=args.checkpoint,
                            varlen=args.varlen, deterministic=args.deterministic, **args.adapter_kwargs)
    sr, ch = adapter.sample_rate, adapter.audio_channels
    ok(f"Model '{args.model}' ready: sr={sr} ch={ch}", prefix="EVAL")

    if not args.sdr_only:
        from fadtk.model_loader import CLAPLaionModel, MERTModel
        mert_ml, clap_music_ml, clap_audio_ml = MERTModel(layer=4), CLAPLaionModel("music"), CLAPLaionModel("audio")
        for ml in (mert_ml, clap_music_ml, clap_audio_ml):
            with silence_output():
                ml.load_model()
            ml.model.to(device)
        get_pann_model(device)                      # fail fast on a missing checkpoint
        ok("Embedding models ready.", prefix="EVAL")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(AudioDataset(my_files, sr, target_channels=ch, cache_dir=cache_dir),
                        batch_size=1, num_workers=args.num_workers, collate_fn=first_item,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    def reference(ml, embed_fn, wav_ref, stem, model_name):
        """Target embedding: cached (fma) or shipped with the clip set (clips)."""
        if args.protocol == "fma":
            return load_or_embed(ml, embed_fn, wav_ref, sr, device, target_cache_path(cache_dir, model_name, stem))
        return _load_ref_emb(data_dir, model_name, stem)

    processed = skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for wav, stem in tqdm(loader, desc=f"{name}.r{rank}", dynamic_ncols=True,
                              mininterval=30.0, file=sys.stdout):
            if wav is None:
                skipped += 1
                continue
            seed = file_seed(args.seed, stem)
            try:
                seed_everything(seed)                                     # SAGE's z, SAME's noise
                pred = adapter.reconstruct(wav)                           # [C, T'] cpu float
            except Exception as e:
                err(f"Inference error {stem}: {e}", prefix="EVAL")
                skipped += 1
                continue
            n = min(wav.shape[-1], pred.shape[-1])
            wav_ref, pred = wav[..., :n], pred[..., :n]                   # [C, n]

            sdr, si_sdr = compute_sdr_and_sisdr(wav_ref, pred)
            append_csv(parts_dir / f"spectral.{rank}.csv", CSV_SCHEMA["spectral"],
                       {"file": stem, "si_sdr": si_sdr, "sdr": sdr, **spectral_losses(wav_ref, pred)})

            if args.compute_ms_metrics and wav_ref.shape[0] == 2 and pred.shape[0] == 2:
                try:
                    append_csv(parts_dir / f"ms_metrics.{rank}.csv", CSV_SCHEMA["ms_metrics"],
                               {"file": stem, **ms_metrics_row(wav_ref, pred, sr)})
                except Exception as e:
                    warn(f"MS metrics {stem}: {e}", prefix="MS")

            if not args.skip_cdpam and not args.sdr_only:
                try:
                    append_csv(parts_dir / f"cdpam.{rank}.csv", CSV_SCHEMA["cdpam"],
                               {"file": stem, "cdpam": cdpam_score(wav_ref, pred, sr, device=device)})
                except Exception as e:
                    warn(f"CDPAM {stem}: {e}", prefix="CDPAM")

            complete = True
            if not args.sdr_only:
                try:
                    for ml, csv_name, emb_name in ((clap_music_ml, "clap_music", CLAP_MUSIC_NAME),
                                                   (clap_audio_ml, "clap_audio", CLAP_AUDIO_NAME)):
                        target = reference(ml, embed_clap, wav_ref, stem, emb_name)
                        p_emb = embed_clap(ml, pred, sr, device)
                        if target is not None:
                            append_csv(parts_dir / f"{csv_name}.{rank}.csv", CSV_SCHEMA[csv_name],
                                       {"file": stem, "cosine": float(cosine_sim(target.mean(0), p_emb.mean(0)))})
                    # FAD sources; PANN is written last, after every other artefact of the file.
                    if args.protocol == "fma":
                        reference(mert_ml, embed_mert_framewise, wav_ref, stem, MERT_NAME)
                    atomic_save_npy(pred_root / MERT_NAME / f"{stem}.npy",
                                    embed_mert_framewise(mert_ml, pred, sr, device).astype(np.float16))
                    # LAION-CLAP embeds a random 10 s crop of longer audio (np.random): reference and
                    # prediction draw their crops from separate seeds, whether or not the reference is cached.
                    if args.protocol == "fma":
                        np.random.seed(seed % 2**32)
                        reference(None, lambda ml, w, s, d: embed_clap_gud(w, s, d), wav_ref, stem, CLAP_GUD_NAME)
                    np.random.seed((seed + 1) % 2**32)
                    atomic_save_npy(pred_root / CLAP_GUD_NAME / f"{stem}.npy",
                                    embed_clap_gud(pred, sr, device).astype(np.float16))
                    if args.protocol == "fma":
                        reference(None, lambda ml, w, s, d: embed_pann(w, s, d), wav_ref, stem, PANN_NAME)
                    atomic_save_npy(pred_root / PANN_NAME / f"{stem}.npy",
                                    embed_pann(pred, sr, device).astype(np.float16))
                except Exception as e:
                    warn(f"Embedding error {stem}: {e} — the file will be retried on resume", prefix="EMBED")
                    complete = False
            if not complete:
                skipped += 1
                continue
            with open(done_file, "a") as df:
                df.write(stem + "\n")
            processed += 1
            if processed % 200 == 0:
                info(f"r{rank}: {processed}/{len(my_files)} ({time.time() - t0:.0f}s)", prefix="EVAL")

    ok(f"Rank {rank} done — processed: {processed}  skipped: {skipped}", prefix="EVAL")
    if rank == 0 and hasattr(adapter, "provenance"):
        (metrics_dir / "varlen.json").write_text(json.dumps(adapter.provenance(), indent=2) + "\n")
    if world == 1:
        _merge(metrics_dir, args)


@hydra.main(config_path="../configs", config_name="reconstruction", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
