#!/usr/bin/env python3
# =============================================================================
# evaluation/sota_models/evaluate_sota_10s.py
# Reconstruction eval for SOTA baselines on the colleague's chunks_mix_original
# dataset (1998 × 10s WAVs). Identical metric set to evaluate_sota.py but uses
# pre-computed reference embeddings/stats bundled with the dataset instead of
# the shared eval_cache, and MERTModel(layer=4) for prediction embeddings.
#
# Reference data layout (data_dir = chunks_mix_original/original/):
#   embeddings/clap-laion-audio/{stem}.npy  (10, 512) — per-file ref CLAP audio
#   embeddings/clap-laion-music/{stem}.npy  (10, 512) — per-file ref CLAP music
#   embeddings/MERT-v1-95M-4/{stem}.npy    (N, 768)  — per-file ref MERT l4
#   stats/{model}/mu.npy + cov.npy          — pre-computed Fréchet reference stats
#
# Per-file resume identical to evaluate_sota.py (parts/done.{rank}.txt).
# SLURM file-sharding supported (SLURM_PROCID / SLURM_NTASKS).
# =============================================================================
from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_SOTA_DIR  = Path(__file__).parent.resolve()
_EVAL_DIR  = _SOTA_DIR.parent
_PROJ_ROOT = _EVAL_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "src"), str(_EVAL_DIR), str(_SOTA_DIR),
           str(_EVAL_DIR / "stereo_diagnosis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sage.utils.console import ok, warn, info, err
from losses import compute_sdr_and_sisdr, stft_loss, spectral_losses, cdpam_score
from utils import (atomic_save_npy, silence_output, ch_name,
                   CHANNEL_MID, CHANNEL_SIDE, CHANNELS)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import (embed_mert_framewise, embed_pann, get_pann_model, PANN_NAME)
from fadtk.model_loader import CLAPLaionModel, MERTModel
from fadtk.fad import calc_frechet_distance

from adapters import build_adapter
from ms_eval import MS_COLUMNS, ms_metrics_row   # torch-only; shared with evaluate_swin_10s

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

_MERT_NAME       = "MERT-v1-95M-4"
_CLAP_AUDIO_NAME = "clap-laion-audio"
_CLAP_MUSIC_NAME = "clap-laion-music"
_PANN_NAME       = PANN_NAME               # pann-cnn14-16k (from compute_fad)

_SKIP_DIRS = frozenset({"embeddings", "stats", "stats_ours", "convert", "metrics", "parts"})

_CSV_SCHEMA = {
    "spectral":   ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"],
    "cdpam":      ["file", "cdpam"],
    "clap_music": ["file", "cosine"],
    "clap_audio": ["file", "cosine"],
    # Pure inference time (audio→latent→reconstruction), no metric/loading cost.
    "timing":     ["file", "infer_sec", "audio_sec"],
    # Stereo-imaging metrics (--ms-only): the baseline row SAGE is measured against.
    # All four adapters are audio_channels=2, so the M/S pair is well defined for
    # every SOTA model here. Absent CSV → reader reports NaN; old runs unchanged.
    "ms_metrics": MS_COLUMNS,
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class _AudioDataset(Dataset):
    def __init__(self, files: list[Path], target_sr: int, target_channels: int = 2):
        self.files           = files
        self.target_sr       = target_sr
        self.target_channels = target_channels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            C = wav.shape[0]
            if C > self.target_channels:
                wav = wav[:self.target_channels]
            elif C < self.target_channels:
                wav = wav.repeat((self.target_channels + C - 1) // C, 1)[:self.target_channels]
            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Incremental I/O helpers ──────────────────────────────────────────────────

def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _load_done(parts: Path, marker: str = "done") -> set[str]:
    """Stems already processed. --new-metrics-only uses a separate marker (done_new)
    so it never collides with the full-run marker (which lists every stem)."""
    done: set[str] = set()
    for f in parts.glob(f"{marker}.*.txt"):
        done |= {s for s in f.read_text().split() if s}
    return done


def _load_ref_emb(data_dir: Path, model_name: str, stem: str) -> Optional[np.ndarray]:
    """Load pre-computed reference embedding; return None if missing."""
    p = data_dir / "embeddings" / model_name / f"{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


# ── File collection ──────────────────────────────────────────────────────────

def _collect_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(
        p for p in data_dir.glob("*.wav")
        if not _SKIP_DIRS.intersection(p.relative_to(data_dir).parts[:-1])
    )
    if max_files > 0:
        files = files[:max_files]
    return files


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOTA eval on chunks_mix_original (pre-computed reference embeddings).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Adapter key: identity | codicodec | music2latent | sao-vae | same | same-s")
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Path to chunks_mix_original/original/ (contains *.wav + embeddings/ + stats/)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--skip-cdpam", action="store_true")
    p.add_argument("--sdr-only", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip entirely if metrics/_done exists.")
    p.add_argument("--fad-gud-only", action="store_true",
                   help="Compute ONLY fad_gudgud (whole-file CLAP, 48kHz, float): skip "
                        "all other metrics. Overwrites fad_gudgud.csv.")
    p.add_argument("--new-metrics-only", action="store_true",
                   help="Compute ONLY fad_pann (PANN Cnn14 whole-file FAD): skip everything "
                        "else. Uses a separate resume marker (done_new.*); writes only fad_pann.csv.")
    p.add_argument("--ms-only", action="store_true",
                   help="Compute ONLY the stereo-imaging metrics (ms_metrics.csv): one decode "
                        "pass, NO embedding model loaded. Backfill twin of --new-metrics-only; "
                        "uses its own resume marker (done_ms.*) and leaves every other CSV "
                        "and _done untouched.")
    p.add_argument("--channel", default=CHANNEL_MID, choices=list(CHANNELS),
                   help="Channel fed to the FAD embedders and to CDPAM. 'mid' = (L+R)/2 "
                        "(historical mono downmix, default -> results unchanged); 'side' = "
                        "(L-R)/2. In 'side' mode ONLY the three FAD sources and CDPAM are "
                        "computed. Embeddings and ref stats live under '<model>-side/'. "
                        "Use a SEPARATE --output-dir: the CSV names are unchanged.")
    p.add_argument("--merge", action="store_true",
                   help="Merge per-file partial outputs into final CSVs + FAD (no GPU).")
    return p.parse_args()


# ── FAD from pre-computed reference stats ────────────────────────────────────

def _fad_from_stats(
    data_dir: Path,
    model_name: str,
    pred_emb_dir: Path,
    csv_path: Path,
    label: str,
) -> None:
    """Load all per-file pred embeddings, compute FAD against pre-computed ref stats."""
    # Prefer OUR self-consistent stats (recompute_ref_stats_10s.py) over the
    # colleague's pre-computed stats/ — see evaluate_swin_10s for rationale.
    if (data_dir / "stats_ours" / model_name / "mu.npy").exists():
        stats_root = data_dir / "stats_ours"
        info(f"Using OUR recomputed ref stats for {model_name}.", prefix="MERGE")
    else:
        stats_root = data_dir / "stats"
    ref_mu_path  = stats_root / model_name / "mu.npy"
    ref_cov_path = stats_root / model_name / "cov.npy"
    if not ref_mu_path.exists() or not ref_cov_path.exists():
        warn(f"No reference stats for {model_name} — FAD skipped.", prefix="MERGE")
        return

    npys = sorted(pred_emb_dir.glob("*.npy")) if pred_emb_dir.is_dir() else []
    if not npys:
        warn(f"No pred embeddings for {model_name} — FAD skipped.", prefix="MERGE")
        return

    ref_mu  = np.load(ref_mu_path).astype(np.float64)
    ref_cov = np.load(ref_cov_path).astype(np.float64)

    from compute_fad import compute_incremental_stats
    pred_mu, pred_cov = compute_incremental_stats(npys)
    
    if pred_mu is None:
        pred_mu = np.zeros_like(ref_mu)
        pred_cov = np.zeros_like(ref_cov)

    score = calc_frechet_distance(ref_mu, ref_cov, pred_mu, pred_cov)
    ok(f"FAD {label}: {score:.6f}", prefix="MERGE")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "score"])
        w.writeheader()
        w.writerow({"model": label, "score": score})


# ── Merge partials → final CSVs + FAD ───────────────────────────────────────

def _merge(metrics_dir: Path, data_dir: Path, sdr_only: bool,
           new_metrics_only: bool = False, ms_only: bool = False,
           channel: str = CHANNEL_MID) -> None:
    parts = metrics_dir / "parts"

    # --new-metrics-only writes no per-file CSV (only the scalar fad_pann below);
    # --ms-only writes exactly one and computes no FAD.
    merge_names = ([] if new_metrics_only
                   else ["ms_metrics"] if ms_only
                   else list(_CSV_SCHEMA))
    for name in merge_names:
        cols = _CSV_SCHEMA[name]
        seen: dict[str, dict] = {}
        for f in sorted(parts.glob(f"{name}.*.csv")):
            with open(f, newline="") as fh:
                for r in _csv.DictReader(fh):
                    seen.setdefault(r["file"], dict(r))
        if seen:
            out = metrics_dir / f"{name}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(seen.values())
            ok(f"{name}.csv merged ({len(seen)} files)", prefix="MERGE")

    pred_root = parts / "pred"
    if not sdr_only and not ms_only:
        # In merge, we just run FAD for whatever pred embeddings exist on disk.
        _mert_ch = ch_name(_MERT_NAME, channel)
        _gud_ch  = ch_name("clap-laion-audio-gud", channel)
        _pann_ch = ch_name(_PANN_NAME, channel)
        if not new_metrics_only and (pred_root / _mert_ch).exists():
            _fad_from_stats(data_dir, _mert_ch, pred_root / _mert_ch,
                            metrics_dir / "fad_mert.csv", _mert_ch)

        if not new_metrics_only and (pred_root / _gud_ch).exists():
            _fad_from_stats(data_dir, _gud_ch, pred_root / _gud_ch,
                            metrics_dir / "fad_gudgud.csv", _gud_ch)

        if (pred_root / _pann_ch).exists():
            _fad_from_stats(data_dir, _pann_ch, pred_root / _pann_ch,
                            metrics_dir / "fad_pann.csv", _pann_ch)

    # Partial refresh runs (fad_gud_only / new_metrics_only / ms_only) must NOT write
    # the full-run sentinel.
    if not new_metrics_only and not ms_only and (not (pred_root / _gud_ch).exists()
                                 or (pred_root / _mert_ch).exists()):
        (metrics_dir / "_done").touch()
    ok(f"_done/merge → {metrics_dir}", prefix="MERGE")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    # Modalità Side: solo le tre sorgenti FAD e CDPAM (vedi --channel).
    side_mode = args.channel == CHANNEL_SIDE
    CH        = args.channel
    data_dir    = args.data_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    metrics_dir = output_root / args.model / "metrics"

    if args.merge:
        _merge(metrics_dir, data_dir, args.sdr_only, args.new_metrics_only, args.ms_only,
               args.channel)
        return

    if args.fad_gud_only and args.resume and (metrics_dir / "fad_gudgud.csv").exists():
        info("[RESUME] fad_gudgud.csv exists — skipping.", prefix="EVAL"); return
    elif (args.new_metrics_only and args.resume
          and (metrics_dir / "fad_pann.csv").exists()):
        info("[RESUME] fad_pann.csv exists — skipping.", prefix="EVAL"); return
    elif args.ms_only and args.resume and (metrics_dir / "ms_metrics.csv").exists():
        info("[RESUME] ms_metrics.csv exists — skipping.", prefix="EVAL"); return
    elif (not args.fad_gud_only and not args.new_metrics_only and not args.ms_only
          and args.resume and (metrics_dir / "_done").exists()):
        info("[RESUME] already done — skipping.", prefix="EVAL"); return

    rank  = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    device = (torch.device("cpu") if args.device == "cpu" or not torch.cuda.is_available()
              else torch.device(f"cuda:{local}"))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(local)

    audio_files = _collect_files(data_dir, args.max_files)
    if not audio_files:
        err(f"No WAV files found in {data_dir}"); return

    parts_dir = metrics_dir / "parts"
    pred_root = parts_dir / "pred"
    parts_dir.mkdir(parents=True, exist_ok=True)

    marker   = ("done_new" if args.new_metrics_only
                else "done_ms" if args.ms_only else "done")
    done     = _load_done(parts_dir, marker)
    my_files = [f for f in audio_files[rank::world] if f.stem not in done]
    done_file = parts_dir / f"{marker}.{rank}.txt"
    info(f"Rank {rank}/{world} on {device} — {len(my_files)} files "
         f"(skip {len(audio_files[rank::world]) - len(my_files)} done)", prefix="EVAL")

    info(f"Building adapter '{args.model}' on {device}...", prefix="EVAL")
    adapter = build_adapter(args.model, device=str(device))
    sr, ch  = adapter.sample_rate, adapter.audio_channels
    ok(f"Adapter ready: sr={sr} ch={ch}", prefix="EVAL")

    if not args.sdr_only and not args.ms_only:
        info("Loading embedding models...", prefix="EVAL")
        embed_models = []
        if not args.fad_gud_only and not args.new_metrics_only:
            # MERT (layer 4) drives fad_mert → full run only (not --new-metrics-only,
            # which computes only fad_pann, nor --fad-gud-only).
            mert_ml       = MERTModel(layer=4)
            embed_models.append(mert_ml)
            # CLAP only feeds the CLAP cosine metric → full run only.
            if not args.new_metrics_only and not side_mode:
                clap_audio_ml = CLAPLaionModel("audio")
                clap_music_ml = CLAPLaionModel("music")
                embed_models.extend([clap_audio_ml, clap_music_ml])

        for _ml in embed_models:
            with silence_output():
                _ml.load_model()
            _ml.model.to(device)
        # PANN drives fad_pann → full run + --new-metrics-only. Fail-fast on missing ckpt.
        if not args.fad_gud_only:
            get_pann_model(device)
        ok("Embedding models ready.", prefix="EVAL")

    dataset = _AudioDataset(my_files, sr, target_channels=ch)
    loader  = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    processed = skipped = 0
    t0 = time.time()

    with torch.no_grad():
        for wav, stem in tqdm(loader, desc=f"{args.model}.r{rank}",
                              dynamic_ncols=True, mininterval=30.0, file=sys.stdout):
            if wav is None:
                skipped += 1; continue

            orig_len = wav.shape[-1]
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                pred = adapter.reconstruct(wav)              # pure inference (enc→dec)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_sec = time.perf_counter() - _t0
            except Exception as e:
                err(f"Inference error {stem}: {e}", prefix="EVAL")
                skipped += 1; continue

            n       = min(orig_len, pred.shape[-1])
            wav_ref = wav[..., :n]
            pred    = pred[..., :n]

            if (not args.fad_gud_only and not args.new_metrics_only
                    and not args.ms_only and not side_mode):
                _append_csv(parts_dir / f"timing.{rank}.csv", _CSV_SCHEMA["timing"],
                            {"file": stem, "infer_sec": infer_sec, "audio_sec": orig_len / sr})
                sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
                _append_csv(parts_dir / f"spectral.{rank}.csv", _CSV_SCHEMA["spectral"],
                            {"file": stem, "si_sdr": sisdr_val, "sdr": sdr_val,
                             **spectral_losses(wav_ref, pred)})

            # Stereo-imaging metrics — the SOTA target row for the Side. Same function
            # evaluate_swin_10s calls, so SAGE and the baselines stay comparable.
            if (args.ms_only and not side_mode
                    and wav_ref.shape[0] == 2 and pred.shape[0] == 2):
                try:
                    _append_csv(parts_dir / f"ms_metrics.{rank}.csv", _CSV_SCHEMA["ms_metrics"],
                                {"file": stem, **ms_metrics_row(wav_ref, pred, sr)})
                except Exception as e:
                    warn(f"MS metrics {stem}: {e}", prefix="MS")

            if (not args.skip_cdpam and not args.sdr_only and not args.fad_gud_only
                    and not args.new_metrics_only and not args.ms_only):
                try:
                    _append_csv(parts_dir / f"cdpam.{rank}.csv", _CSV_SCHEMA["cdpam"],
                                {"file": stem,
                                 "cdpam": cdpam_score(wav_ref, pred, sr, device=device,
                                                       channel=CH)})
                except Exception as e:
                    warn(f"CDPAM {stem}: {e}", prefix="CDPAM")

            if not args.sdr_only and not args.ms_only:
                try:
                    if not args.fad_gud_only:
                        # CLAP audio/music cosine + pred embeddings → full run only.
                        # in modalità Side i modelli CLAP non sono caricati: la
                        # cosine vive solo sul Mid ed è già misurata.
                        if not args.new_metrics_only and not side_mode:
                            ref_ca_emb = _load_ref_emb(data_dir, _CLAP_AUDIO_NAME, stem)
                            p_ca = embed_clap(clap_audio_ml, pred, sr, device)
                            if ref_ca_emb is not None:
                                _append_csv(parts_dir / f"clap_audio.{rank}.csv",
                                            _CSV_SCHEMA["clap_audio"],
                                            {"file": stem,
                                             "cosine": float(cosine_sim(ref_ca_emb.mean(0), p_ca.mean(0)))})
                            atomic_save_npy(pred_root / _CLAP_AUDIO_NAME / f"{stem}.npy",
                                            p_ca.astype(np.float16))

                            ref_cm_emb = _load_ref_emb(data_dir, _CLAP_MUSIC_NAME, stem)
                            p_cm = embed_clap(clap_music_ml, pred, sr, device)
                            if ref_cm_emb is not None:
                                _append_csv(parts_dir / f"clap_music.{rank}.csv",
                                            _CSV_SCHEMA["clap_music"],
                                            {"file": stem,
                                             "cosine": float(cosine_sim(ref_cm_emb.mean(0), p_cm.mean(0)))})
                            atomic_save_npy(pred_root / _CLAP_MUSIC_NAME / f"{stem}.npy",
                                            p_cm.astype(np.float16))

                        # MERT l4 framewise → fad_mert source (full run only).
                        if not args.new_metrics_only:
                            p_mert = embed_mert_framewise(mert_ml, pred, sr, device, CH)
                            atomic_save_npy(pred_root / ch_name(_MERT_NAME, CH) / f"{stem}.npy",
                                            p_mert.astype(np.float16))

                    # CLAP-gud whole-file → fad_gudgud source (full + --fad-gud-only).
                    if not args.new_metrics_only:
                        from compute_clap_score import embed_clap_gud
                        p_clap_gud = embed_clap_gud(pred, sr, device, CH)
                        atomic_save_npy(pred_root / ch_name("clap-laion-audio-gud", CH) / f"{stem}.npy", p_clap_gud.astype(np.float16))

                    # PANN Cnn14 whole-file → fad_pann source (full + --new-metrics-only).
                    if not args.fad_gud_only:
                        p_pann = embed_pann(pred, sr, device, CH)
                        atomic_save_npy(pred_root / ch_name(_PANN_NAME, CH) / f"{stem}.npy", p_pann.astype(np.float16))

                except Exception as e:
                    warn(f"Embedding error {stem}: {e}", prefix="EMBED")

            with open(done_file, "a") as df:
                df.write(stem + "\n")
            processed += 1
            if processed % 200 == 0:
                info(f"r{rank}: {processed}/{len(my_files)} ({time.time()-t0:.0f}s)", prefix="EVAL")

    ok(f"Rank {rank} done — processed: {processed}  skipped: {skipped}", prefix="EVAL")

    if world == 1:
        _merge(metrics_dir, data_dir, args.sdr_only, args.new_metrics_only, args.ms_only,
               args.channel)


if __name__ == "__main__":
    main()
