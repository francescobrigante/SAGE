#!/usr/bin/env python3
# =============================================================================
# evaluation/sota_models/evaluate_sota.py
# Model-agnostic reconstruction eval for SOTA baselines. One --model flag selects
# a CodecAdapter (adapters.py); the per-file loop, metrics, caching and CSV
# outputs are IDENTICAL to evaluate_sao.py so results drop straight into the same
# runs/<model>/metrics/ layout and results_viz.ipynb.
#
# Scales to the full FMA-large test split via SLURM file-sharding: each rank
# (SLURM_PROCID of SLURM_NTASKS) processes files[rank::world] on cuda:LOCALID.
# Outputs are written INCREMENTALLY per file (CSV rows appended, pred embeddings
# saved per stem) so a relaunch RESUMES from where it stopped. A final --merge
# pass concatenates the per-file CSVs and computes FAD over all pred embeddings
# vs the shared target cache. Single-process runs (world==1) merge inline.
# =============================================================================
from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_SOTA_DIR  = Path(__file__).parent.resolve()
_EVAL_DIR  = _SOTA_DIR.parent
_PROJ_ROOT = _EVAL_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "src"), str(_EVAL_DIR), str(_SOTA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ar_spectra.utils.console import ok, warn, info, err
from losses import compute_sdr_and_sisdr, stft_loss, cdpam_score
from utils import (write_csv, collect_fma_files, atomic_save_npy,
                   load_or_embed, target_cache_path, silence_output)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import embed_mert, compute_fad_from_embeddings
from fadtk.model_loader import CLAPLaionModel, MERTModel

from adapters import build_adapter

# numpy 1.24+ removed np.float — patch for CDPAM internals (mirrors evaluate_sao)
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

# Embedder names == their target-cache subdir names (shared with Swin/SAO).
_MERT_NAME = "MERT-v1-95M"
_CLAP_AUDIO_NAME = "clap-laion-audio"

# Per-metric CSV schema (used for partial writes and the merge concat).
_CSV_SCHEMA = {
    "spectral":   ["file", "si_sdr", "sdr", "stft_loss"],
    "cdpam":      ["file", "cdpam"],
    "clap_music": ["file", "cosine"],
    "clap_audio": ["file", "cosine"],
}


# ── Dataset ───────────────────────────────────────────────────
# Identical to evaluate_sao._AudioDataset: load + resample + normalise channels,
# with optional .npy cache of the resampled waveform (shared across models).
class _AudioDataset(Dataset):
    def __init__(self, files: list[Path], target_sr: int,
                 target_channels: int = 2, cache_dir: Optional[Path] = None):
        self.files           = files            # this rank's (resume-filtered) file shard
        self.target_sr       = target_sr        # adapter.sample_rate
        self.target_channels = target_channels  # adapter.audio_channels
        self.cache_dir       = cache_dir        # shared resampled-wav cache

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            cache_file = (
                self.cache_dir /
                f"wav_{p.stem}_ch{self.target_channels}_sr{self.target_sr}.npy"
            ) if self.cache_dir is not None else None

            if cache_file is not None and cache_file.exists():
                return torch.from_numpy(np.load(cache_file).astype(np.float32)), p.stem

            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            C = wav.shape[0]
            if C > self.target_channels:
                wav = wav[:self.target_channels]
            elif C < self.target_channels:
                wav = wav.repeat((self.target_channels + C - 1) // C, 1)[:self.target_channels]

            if cache_file is not None:
                atomic_save_npy(cache_file, wav.numpy())
            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Incremental partial I/O (enables file-level resume) ────────
def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append one row to a CSV, writing the header if the file is new/empty."""
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _load_done(parts: Path) -> set[str]:
    """Stems already fully processed by any rank (union of done.*.txt)."""
    done: set[str] = set()
    for f in parts.glob("done.*.txt"):
        done |= {s for s in f.read_text().split() if s}
    return done


# ── Argument parsing ──────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model-agnostic SOTA-baseline reconstruction eval on FMA test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Adapter key: identity | codicodec | music2latent | same")
    p.add_argument("--target-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--fma-csv-path", default=None)
    p.add_argument("--extensions", default=".wav,.flac,.mp3,.ogg")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Shared cache of resampled waveforms + target embeddings "
                        "(reuse the SAME dir across all models for comparable FAD).")
    p.add_argument("--skip-cdpam", action="store_true")
    p.add_argument("--sdr-only", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip entirely if metrics/_done exists (per-file resume is automatic).")
    p.add_argument("--merge", action="store_true",
                   help="Merge per-file partial outputs into final CSVs + FAD (no GPU).")
    return p.parse_args()


# ── Merge partials → final CSVs + FAD ──────────────────────────
def _merge(metrics_dir: Path, cache_dir: Optional[Path], sdr_only: bool) -> None:
    """Concatenate metrics/parts/{name}.*.csv into final CSVs (dedup by file) and
    compute FAD over all per-stem pred embeddings vs the shared target cache."""
    parts = metrics_dir / "parts"

    for name, cols in _CSV_SCHEMA.items():
        seen: dict[str, dict] = {}
        for f in sorted(parts.glob(f"{name}.*.csv")):
            with open(f, newline="") as fh:
                for r in _csv.DictReader(fh):
                    seen.setdefault(r["file"], dict(r))           # dedup by file, keep first
        if seen:
            write_csv(metrics_dir / f"{name}.csv", cols, list(seen.values()))
            ok(f"{name}.csv merged ({len(seen)} files)", prefix="MERGE")

    if not sdr_only and cache_dir:
        for emb_name, csv_name in ((_MERT_NAME, "fad_mert.csv"),
                                   (_CLAP_AUDIO_NAME, "fad_gudgud.csv")):
            emb_dir = parts / "pred" / emb_name
            npys = sorted(emb_dir.glob("*.npy")) if emb_dir.is_dir() else []
            if not npys:
                warn(f"No pred embeddings for {emb_name} — FAD skipped.", prefix="MERGE")
                continue
            pred = np.concatenate([np.load(f).astype(np.float32) for f in npys], axis=0)
            fad_files = [Path(f.stem) for f in npys]               # .stem -> cache lookup
            compute_fad_from_embeddings(
                SimpleNamespace(name=emb_name), fad_files, [pred],
                cache_dir, metrics_dir / csv_name, emb_name,
            )
            ok(f"{csv_name} merged (FAD over {len(npys)} files)", prefix="MERGE")

    (metrics_dir / "_done").touch()
    ok(f"_done written → {metrics_dir}", prefix="MERGE")


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_root / args.model / "metrics"
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    if args.merge:
        _merge(metrics_dir, cache_dir, args.sdr_only)
        return

    if args.resume and (metrics_dir / "_done").exists():
        info("[RESUME] already done — skipping.", prefix="EVAL"); return

    # ── SLURM file-sharding (mirrors evaluate_swin rank→GPU) ──
    rank  = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    device = (torch.device("cpu") if args.device == "cpu" or not torch.cuda.is_available()
              else torch.device(f"cuda:{local}"))

    target_dir = Path(args.target_dir).expanduser().resolve()
    audio_exts = {(e if e.startswith(".") else f".{e}").lower()
                  for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files in {target_dir}"); return

    parts_dir = metrics_dir / "parts"
    pred_root = parts_dir / "pred"
    parts_dir.mkdir(parents=True, exist_ok=True)
    stem_to_file: dict[str, Path] = {f.stem: f for f in audio_files}

    done = _load_done(parts_dir)                                  # already-processed stems (resume)
    my_files = [f for f in audio_files[rank::world] if f.stem not in done]
    done_file = parts_dir / f"done.{rank}.txt"
    info(f"Rank {rank}/{world} on {device} — {len(my_files)} files "
         f"(skipping {len(audio_files[rank::world]) - len(my_files)} already done)", prefix="EVAL")

    info(f"Building adapter '{args.model}' on {device}...", prefix="EVAL")
    adapter = build_adapter(args.model, device=str(device))
    sr, ch = adapter.sample_rate, adapter.audio_channels
    ok(f"Adapter ready: sr={sr} ch={ch}", prefix="EVAL")

    if not args.sdr_only:
        info("Loading CLAP (music/audio) and MERT models...", prefix="EVAL")
        clap_music_ml = CLAPLaionModel("music")
        clap_audio_ml = CLAPLaionModel("audio")
        mert_ml       = MERTModel()
        for _ml in (clap_music_ml, clap_audio_ml, mert_ml):
            with silence_output():
                _ml.load_model()
            _ml.model.to(device)
        ok("Embedding models ready.", prefix="EVAL")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = _AudioDataset(my_files, sr, target_channels=ch, cache_dir=cache_dir)
    loader  = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    processed = skipped = 0
    t0 = time.time()

    with torch.no_grad():
        for i, (wav, stem) in enumerate(tqdm(loader, desc=f"{args.model}.r{rank}",
                                             dynamic_ncols=True, mininterval=30.0,
                                             file=sys.stdout)):
            if wav is None:
                skipped += 1; continue
            orig_len = wav.shape[-1]

            try:
                pred = adapter.reconstruct(wav)            # [C, T'] cpu float
            except Exception as e:
                err(f"Inference error {stem}: {e}", prefix="EVAL")
                skipped += 1; continue

            n       = min(orig_len, pred.shape[-1])
            wav_ref = wav[..., :n]                          # [C, n]
            pred    = pred[..., :n]                         # [C, n]

            sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
            _append_csv(parts_dir / f"spectral.{rank}.csv", _CSV_SCHEMA["spectral"],
                        {"file": stem, "si_sdr": sisdr_val, "sdr": sdr_val,
                         "stft_loss": float(stft_loss(wav_ref, pred))})

            if not args.skip_cdpam and not args.sdr_only:
                try:
                    _append_csv(parts_dir / f"cdpam.{rank}.csv", _CSV_SCHEMA["cdpam"],
                                {"file": stem, "cdpam": cdpam_score(wav_ref, pred, sr, device=device)})
                except Exception as e:
                    warn(f"CDPAM {stem}: {e}", prefix="CDPAM")

            if not args.sdr_only:
                try:
                    t_cm = load_or_embed(clap_music_ml, embed_clap, wav_ref, sr, device,
                                         target_cache_path(cache_dir, clap_music_ml.name, stem))
                    p_cm = embed_clap(clap_music_ml, pred, sr, device)
                    _append_csv(parts_dir / f"clap_music.{rank}.csv", _CSV_SCHEMA["clap_music"],
                                {"file": stem, "cosine": float(cosine_sim(t_cm.mean(0), p_cm.mean(0)))})

                    t_ca = load_or_embed(clap_audio_ml, embed_clap, wav_ref, sr, device,
                                         target_cache_path(cache_dir, clap_audio_ml.name, stem))
                    p_ca = embed_clap(clap_audio_ml, pred, sr, device)
                    _append_csv(parts_dir / f"clap_audio.{rank}.csv", _CSV_SCHEMA["clap_audio"],
                                {"file": stem, "cosine": float(cosine_sim(t_ca.mean(0), p_ca.mean(0)))})

                    load_or_embed(mert_ml, embed_mert, wav_ref, sr, device,
                                  target_cache_path(cache_dir, mert_ml.name, stem))
                    p_mert = embed_mert(mert_ml, pred, sr, device)
                    # per-stem pred embeddings → resume-safe FAD inputs
                    atomic_save_npy(pred_root / _CLAP_AUDIO_NAME / f"{stem}.npy", p_ca.astype(np.float16))
                    atomic_save_npy(pred_root / _MERT_NAME / f"{stem}.npy", p_mert.astype(np.float16))
                except Exception as e:
                    warn(f"Embedding error {stem}: {e}", prefix="EMBED")

            with open(done_file, "a") as df:                # mark file done (resume marker)
                df.write(stem + "\n")
            processed += 1
            if processed % 200 == 0:
                info(f"r{rank}: {processed}/{len(my_files)} ({time.time()-t0:.0f}s)", prefix="EVAL")

    ok(f"Rank {rank} done — processed: {processed}  skipped: {skipped}", prefix="EVAL")

    if world == 1:                                          # single-process → merge inline
        _merge(metrics_dir, cache_dir, args.sdr_only)


if __name__ == "__main__":
    main()
