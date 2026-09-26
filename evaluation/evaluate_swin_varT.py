#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate_swin_varT.py
# Parallel multi-checkpoint evaluation for Swin C-VAE (variable-length audio).
#
# With srun --ntasks-per-node=N, each rank handles every N-th checkpoint
# on its own GPU (SLURM_LOCALID). Audio is padded to the next temporal
# multiple of 2^num_downsamples and processed in a single forward pass —
# no fixed-chunk splitting. Metrics: SI-SDR, STFT, CDPAM, CLAP, FAD.
# =============================================================================

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_EVAL_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from ar_spectra.models.inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, info, err
from c_vae.swin.varlen import resolve
from config import DATA_PATH, DEFAULT_AUDIO_EXTENSIONS, DEFAULT_MAX_FILES, FMA_METADATA
from losses import compute_sdr_and_sisdr, stft_loss, spectral_losses, cdpam_score

from utils import (write_csv, collect_fma_files, atomic_save_npy,
                   load_or_embed, target_cache_path, silence_output)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import (embed_mert_framewise, compute_fad_from_embeddings,
                         embed_pann, get_pann_model, PANN_NAME)
from fadtk.model_loader import CLAPLaionModel, MERTModel

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

# Opt-in stereo-imaging metrics (--compute-ms-metrics). Columns + lazily-loaded
# (align, stereo_imaging_distance) from evaluation/stereo_diagnosis/ — imported
# ONLY when the flag is set, so old runs stay byte-for-byte unchanged.
_MS_COLS = ["file", "width_bias", "d_width", "sisdr_s", "sisdr_m"]
_STEREO_FNS = None


def _si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-10) -> float:
    """Scale-invariant SDR (dB), matching stereo_diagnosis/measure_imaging.si_sdr."""
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est * ref).sum() / (ref.square().sum() + eps)
    proj = a * ref
    return float((10 * torch.log10(proj.square().sum() / ((est - proj).square().sum() + eps))).item())


def _compute_ms_metrics(ref: torch.Tensor, pred: torch.Tensor, sr: int) -> dict:
    """Four stereo-imaging metrics on an aligned (2, T) ref/pred pair.

    width_bias (→0; <0 squash, >0 over-widening) and d_width (→0, unsigned image
    error) from stereo_imaging_distance; sisdr_s = SI-SDR on the side S=(L−R)/√2 —
    the channel SAGE collapses; sisdr_m = SI-SDR on the mid M=(L+R)/√2 (guardrail:
    fixing Side must not degrade Mid). Matches evaluation/stereo_diagnosis definitions.
    """
    global _STEREO_FNS
    if _STEREO_FNS is None:
        sys.path.insert(0, str(_EVAL_DIR / "stereo_diagnosis"))
        from stereo_imaging import align, stereo_imaging_distance   # torch-only, no heavy deps
        _STEREO_FNS = (align, stereo_imaging_distance)
    align, stereo_imaging_distance = _STEREO_FNS

    ref_a, pred_a = align(ref.float(), pred.float())                 # delay-compensate (≈no-op for our recon)
    d = stereo_imaging_distance(ref_a, pred_a, sample_rate=sr)       # width_bias, d_width, …
    Sr = (ref_a[0] - ref_a[1]) / 2 ** 0.5                            # (T,) reference side
    Sx = (pred_a[0] - pred_a[1]) / 2 ** 0.5                          # (T,) predicted side
    Mr = (ref_a[0] + ref_a[1]) / 2 ** 0.5                            # (T,) reference mid
    Mx = (pred_a[0] + pred_a[1]) / 2 ** 0.5                          # (T,) predicted mid
    return {"width_bias": d["width_bias"], "d_width": d["d_width"],
            "sisdr_s": _si_sdr(Sx, Sr), "sisdr_m": _si_sdr(Mx, Mr)}


def _purge_pred_embeddings(pred_root: Path, embedder_name: str) -> None:
    """Delete the per-file FAD *prediction* embeddings of one embedder after its
    FAD score has been written.

    ``parts/pred/<embedder>/*.npy`` are model-specific, single-use caches: FAD needs
    the whole set on disk to compute mean/covariance, but they are dead weight
    (~36 GB/model) once the CSV exists. The reusable *target*-side cache lives in
    ``cache_dir`` and is NEVER touched here. Now-empty ``pred``/``parts`` parents are
    pruned so no stale skeleton is left behind.
    """
    d = pred_root / embedder_name
    if not d.is_dir():
        return
    shutil.rmtree(d, ignore_errors=True)
    for parent in (pred_root, pred_root.parent):          # pred/, then parts/
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
    info(f"purged pred embeddings ({embedder_name}) — kept CSV, target cache untouched",
         prefix="EVAL")


# ── Dataset ───────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Load audio files; returns (waveform [C, T], stem).

    Resampled waveforms are cached as float32 .npy when cache_dir is set,
    avoiding repeated MP3 decode + resample for every checkpoint.
    """

    def __init__(self, files: list[Path], target_sr: int, target_ch: int,
                 cache_dir: Optional[Path] = None):
        self.files     = files
        self.target_sr = target_sr
        self.target_ch = target_ch
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            cache_file = (
                self.cache_dir /
                f"wav_{p.stem}_ch{self.target_ch}_sr{self.target_sr}.npy"
            ) if self.cache_dir is not None else None

            if cache_file is not None and cache_file.exists():
                return torch.from_numpy(np.load(cache_file).astype(np.float32)), p.stem

            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            C = wav.shape[0]
            if C > self.target_ch:
                wav = wav[:self.target_ch]
            elif C < self.target_ch:
                wav = wav.repeat((self.target_ch + C - 1) // C, 1)[:self.target_ch]

            if cache_file is not None:
                atomic_save_npy(cache_file, wav.numpy())

            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Padding ───────────────────────────────────────────────────

def pad_audio_for_swin(
    wav: torch.Tensor, hop_length: int, num_downsamples: int = 2
) -> tuple[torch.Tensor, int]:
    """Pad waveform so STFT frame count T is a multiple of 2^num_downsamples."""
    orig_samples = wav.shape[-1]
    w_multiple   = 2 ** num_downsamples
    W_current    = (orig_samples // hop_length) + 1
    pad_w        = (w_multiple - W_current % w_multiple) % w_multiple
    target_samples = max(orig_samples, (W_current + pad_w - 1) * hop_length)
    if target_samples > orig_samples:
        wav = F.pad(wav, (0, target_samples - orig_samples))
    return wav, orig_samples


# ── File-sharding helpers (single-checkpoint 2-GPU mode) ──────

def _file_barrier(metrics_dir: Path, local_rank: int, world_size: int,
                  timeout: int = 7200) -> bool:
    """Each rank writes a sentinel; rank 0 waits until all are present."""
    import time as _time
    (metrics_dir / f"_rk{local_rank}_barrier").touch()
    if local_rank != 0:
        return True
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if all((metrics_dir / f"_rk{r}_barrier").exists() for r in range(world_size)):
            for r in range(world_size):
                (metrics_dir / f"_rk{r}_barrier").unlink(missing_ok=True)
            return True
        _time.sleep(5)
    warn(f"Barrier timeout after {timeout}s — not all ranks finished", prefix="EVAL")
    return False


def _merge_shards(metrics_dir: Path, stem: str, fieldnames: list[str],
                  world_size: int) -> None:
    """Merge per-rank CSV shards into final CSV and delete shard files."""
    all_rows: list[dict] = []
    for r in range(world_size):
        p = metrics_dir / f"{stem}_rk{r}.csv"
        if p.exists():
            with open(p, newline="") as f:
                all_rows.extend(list(_csv.DictReader(f)))
            p.unlink()
    if all_rows:
        write_csv(metrics_dir / f"{stem}.csv", fieldnames, all_rows)


# ── Per-file resume ──────────────────────────────────────────
# Same pattern as evaluate_swin_10s.py: per-file rows are appended (not
# overwritten) to a "parts" CSV as each file finishes, so a job killed by a
# wall-time limit loses nothing — a resumed run just keeps appending. The
# "parts" file is dedup-merged into the real metrics CSV (first occurrence
# per stem wins) every time a run's own loop completes uninterrupted, so
# reruns of the single-metric modes (--cdpam-only etc., which don't gate on
# --resume) stay idempotent instead of duplicating rows.
#
# FAD is NOT per-file (it's a Frechet distance over the whole embedding
# distribution), so it can't be resumed the same way — but its per-file
# *embeddings* already were (atomic_save_npy). The fix: only feed
# compute_fad_from_embeddings the files whose embedding actually exists on
# disk RIGHT NOW, checked fresh after the loop — this is correct whether
# that embedding was written by this run or an earlier, killed one. PANN is
# the last embedding written per file (see the embedding block below), so
# its presence is the single completion sentinel — same choice already made
# in evaluate_swin_10s.py.

def _write_varlen_stamp(metrics_dir: Path, codec) -> None:
    """Record which attention mode actually produced these numbers.

    The output directory is named after the checkpoint symlink, which is a naming
    convention and proves nothing about runtime — the same weights can be evaluated
    with any ``--varlen`` preset. Rule (CLAUDE.md, VARLEN_SEAMS.md §8.4): a metrics/
    dir WITHOUT this file is a baseline (single-phase) run. Mirrors the stamp
    evaluate_swin_10s.py already writes.
    """
    vl = resolve(codec.varlen_mode)
    (metrics_dir / "varlen.json").write_text(json.dumps(
        {"mode": codec.varlen_mode,
         "blocks": codec.varlen_blocks,
         "phases": list(vl.phases) if vl else [],
         "combine": vl.combine if vl else None}, indent=2) + "\n")


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)


def _finalize_csv(parts_dir: Path, metrics_dir: Path, name: str,
                  fieldnames: list[str], suf: str) -> int:
    """Dedup-merge one incrementally-appended parts CSV into the final metrics CSV."""
    src = parts_dir / f"{name}{suf}.csv"
    if not src.exists():
        return 0
    seen: dict[str, dict] = {}
    with open(src, newline="") as f:
        for r in _csv.DictReader(f):
            seen.setdefault(r["file"], dict(r))
    if not seen:
        return 0
    write_csv(metrics_dir / f"{name}{suf}.csv", fieldnames, list(seen.values()))
    return len(seen)


# ── Argument parsing ──────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Variable-T Swin C-VAE evaluation — SI-SDR, STFT, CDPAM, CLAP, FAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",      nargs="+", required=True, type=Path,
                   help="One or more Swin .ckpt files; distributed across SLURM ranks.")
    p.add_argument("--target-dir",      default=str(DATA_PATH))
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--num-workers",     type=int, default=4)
    p.add_argument("--max-files",       type=int, default=DEFAULT_MAX_FILES)
    p.add_argument("--dataset",         choices=["fma", "moisesdb"], default="fma")
    p.add_argument("--moisesdb-split",  choices=["mixtures", "stems"], default="mixtures")
    p.add_argument("--fma-csv-path",    default=FMA_METADATA)
    p.add_argument("--extensions",      default=",".join(DEFAULT_AUDIO_EXTENSIONS))
    p.add_argument("--cache-dir",       type=Path, default=None,
                   help="Cache resampled waveforms and target embeddings across checkpoints.")
    p.add_argument("--num-downsamples", type=int, default=None,
                   help="Override PatchMerging count (auto-detected from checkpoint if omitted).")
    p.add_argument("--skip-cdpam",      action="store_true")
    p.add_argument("--cdpam-only",      action="store_true",
                   help="Only calculate CDPAM and write cdpam.csv (skips _done check if cdpam.csv is missing)")
    p.add_argument("--sdr-only",        action="store_true",
                   help="Only compute SI-SDR, SDR and STFT loss, skipping CDPAM, CLAP, and FAD.")
    p.add_argument("--fad-only",        action="store_true",
                   help="Compute ONLY fad_mert (framewise MERT, layer 4): skip "
                        "SI-SDR/STFT, CDPAM, CLAP cosine and fad_gudgud. Overwrites "
                        "fad_mert.csv, leaves every other metric CSV and _done untouched.")
    p.add_argument("--fad-gud-only", action="store_true",
                   help="Compute ONLY fad_gudgud (whole-file CLAP, 48kHz, float): skip "
                        "all other metrics. Overwrites fad_gudgud.csv.")
    p.add_argument("--new-metrics-only", action="store_true",
                   help="Compute ONLY fad_pann (PANN Cnn14 whole-file FAD): skip SI-SDR/STFT, "
                        "CDPAM, CLAP cosine, fad_mert and fad_gudgud. Decodes each file once; "
                        "overwrites fad_pann.csv, leaves every other CSV and _done untouched.")
    p.add_argument("--compute-ms-metrics", action="store_true",
                   help="ADD stereo-imaging metrics (width_bias, d_width, sisdr_s, sisdr_m) to the "
                        "default per-file metrics, computed in-memory on the (ref, recon) pair. "
                        "Opt-in: off by default → old runs unchanged. Writes ms_metrics.csv.")
    p.add_argument("--resume",          action="store_true",
                   help="Skip checkpoint if metrics/_done already exists. In the full "
                        "metric mode (no --*-only flag) this ALSO skips individual files "
                        "whose embeddings are already on disk, so a run killed by a wall-time "
                        "limit can be finished by resubmitting the same command.")
    p.add_argument("--deterministic",   action="store_true",
                   help="Encode with VAE posterior mean μ instead of sampled z. "
                        "Improves SI-SDR/STFT at inference time. Default: False.")
    p.add_argument("--shard-files",     action="store_true",
                   help="When a single checkpoint is evaluated across multiple ranks, "
                        "shard audio files across ranks instead of leaving rank 1+ idle. "
                        "Gives ~Nx speedup with N GPUs. All metrics are bit-identical to "
                        "the single-rank run (FAD is merged on rank 0 after a barrier). "
                        "Default: False (backward-compatible).")
    p.add_argument("--varlen",          default=None,
                   help="Variable-length seam fix on the collapsed Swin stages: a preset "
                        "from config/inference/varlen.yaml (tri2, hard2, tri4, ...) or 'off' "
                        "for the original single-phase attention. Default = the project "
                        "default in that file. See VARLEN_SEAMS.md.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── SLURM rank → GPU assignment ────────────────────────────
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Pin the default CUDA device to this rank's GPU. Critical for --shard-files
    # on >1 GPU: fadtk/laion embedders place their internal tensors on the
    # *current* device (their self.device is the indexless torch.device('cuda')),
    # so without this rank 1 would mix cuda:1 weights with cuda:0 inputs and throw
    # "tensors on cuda:0 and cuda:1". Do NOT instead reassign self.device to an
    # indexed device — fadtk compares `self.device == torch.device('cuda')`.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Use .absolute() instead of .resolve() to keep the symlink name
    all_ckpts = [Path(c).expanduser().absolute() for c in args.checkpoint]

    # Single-checkpoint file-sharding: both ranks process the same checkpoint on
    # separate GPUs, splitting the file list.  All metrics are bit-identical to
    # single-rank evaluation (FAD merged on rank 0 after a barrier).
    single_ckpt_sharding = (
        args.shard_files and len(all_ckpts) == 1 and world_size > 1
    )
    if single_ckpt_sharding:
        my_ckpts = all_ckpts  # every rank processes the same checkpoint
        info(f"Rank {local_rank}/{world_size} on {device} — "
             f"file-sharding mode (1 ckpt, {world_size} GPUs)", prefix="EVAL")
    else:
        my_ckpts = all_ckpts[local_rank::world_size]
        if not my_ckpts:
            info(f"Rank {local_rank}: no checkpoints assigned — exiting.", prefix="EVAL")
            return
        info(f"Rank {local_rank}/{world_size} on {device} — "
             f"{len(my_ckpts)}/{len(all_ckpts)} checkpoints", prefix="EVAL")

    if args.dataset == "moisesdb" and args.target_dir == str(DATA_PATH):
        # Auto-resolve MoisesDB path if target_dir is still the FMA default
        moisesdb_root = os.getenv("MOISESDB_CHUNKS_ROOT", "/leonardo_scratch/fast/IscrC_AHNetBio/datasets/moisesdb/chunks_30s")
        target_dir = Path(moisesdb_root).expanduser().resolve()
    else:
        target_dir  = Path(args.target_dir).expanduser().resolve()
        
    output_root = Path(args.output_dir).expanduser().resolve()
    cache_dir   = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    if not target_dir.is_dir():
        err(f"Target directory not found: {target_dir}"); return
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        info(f"Cache dir: {cache_dir}", prefix="EVAL")

    if args.dataset == "fma":
        audio_exts  = {(e if e.startswith(".") else f".{e}").lower()
                       for e in args.extensions.split(",")}
        audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    elif args.dataset == "moisesdb":
        from utils import collect_moisesdb_files
        audio_files = collect_moisesdb_files(target_dir, args.moisesdb_split, args.max_files)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not audio_files:
        err(f"No audio files found in {target_dir}"); return
    ok(f"Files to evaluate: {len(audio_files)}", prefix="EVAL")

    if not args.cdpam_only and not args.sdr_only:
        # Load MERT (layer 4, framewise/fadtk-standard) once per rank. CLAP only
        # for the non-FAD metrics → skipped in --fad-only.
        info("Loading embedding models...", prefix="EVAL")
        embed_models = []
        # MERT (layer 4) drives fad_mert → needed in full run / --fad-only (NOT
        # --new-metrics-only, which computes only fad_pann, nor --fad-gud-only).
        if not args.fad_gud_only and not args.new_metrics_only:
            mert_ml       = MERTModel(layer=4)   # framewise/fadtk-standard, layer 4
            embed_models.append(mert_ml)
            # CLAP only feeds the CLAP cosine metric → full run only.
            if not args.fad_only and not args.new_metrics_only:
                clap_music_ml = CLAPLaionModel("music")
                clap_audio_ml = CLAPLaionModel("audio")
                embed_models.extend([clap_music_ml, clap_audio_ml])

        for _ml in embed_models:
            with silence_output():
                _ml.load_model()
            _ml.model.to(device)
        # PANN drives fad_pann → full run + --new-metrics-only. Load once now so a
        # missing checkpoint aborts loudly instead of being swallowed per-file.
        if not args.fad_only and not args.fad_gud_only:
            get_pann_model(device)
        ok("Embedding models ready.", prefix="EVAL")

    for ckpt_path in my_ckpts:
        metrics_dir = output_root / ckpt_path.stem / "metrics"
        ok(f"=== {ckpt_path.name} (rank {local_rank}) ===", prefix="EVAL")

        if args.cdpam_only and (metrics_dir / "cdpam.csv").exists():
            info("[RESUME] cdpam.csv already exists — skipping.", prefix="EVAL"); continue
        elif args.fad_only and args.resume and (metrics_dir / "fad_mert.csv").exists():
            info("[RESUME] fad_mert.csv exists — skipping.", prefix="EVAL"); continue
        elif args.fad_gud_only and args.resume and (metrics_dir / "fad_gudgud.csv").exists():
            info("[RESUME] fad_gudgud.csv exists — skipping.", prefix="EVAL"); continue
        elif (args.new_metrics_only and args.resume
              and (metrics_dir / "fad_pann.csv").exists()):
            info("[RESUME] fad_pann.csv exists — skipping.", prefix="EVAL"); continue
        elif not args.cdpam_only and not args.fad_only and not args.fad_gud_only and not args.new_metrics_only and args.resume and (metrics_dir / "_done").exists():
            info("[RESUME] already done — skipping.", prefix="EVAL"); continue

        try:
            codec = EuleroEncodeDecode(ckpt_path, device=device, varlen=args.varlen)
        except Exception as e:
            err(f"Failed to load {ckpt_path.name}: {e}"); continue

        sr: int = codec.sample_rate    or 44100
        ch: int = codec.audio_channels or 2

        stft_cfg = getattr(codec.autoencoder, "_stft_config", None)
        if stft_cfg is None:
            err(f"No STFT config in {ckpt_path.name}."); continue
        hop_length: int = stft_cfg.hop_length

        if args.num_downsamples is not None:
            num_downsamples = args.num_downsamples
        else:
            try:
                num_downsamples = len(codec.autoencoder.encoder.depths) - 1
            except AttributeError:
                warn(f"Cannot read encoder.depths; defaulting num_downsamples=2", prefix="EVAL")
                num_downsamples = 2
        ok(f"sr={sr} ch={ch} hop={hop_length} "
           f"depths={getattr(codec.autoencoder.encoder, 'depths', '?')} "
           f"downsamples={num_downsamples}", prefix="EVAL")

        metrics_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_rk{local_rank}" if single_ckpt_sharding else ""
        parts_dir = metrics_dir / "parts"
        pred_root = parts_dir / "pred"
        parts_dir.mkdir(parents=True, exist_ok=True)

        # In sharding mode each rank processes a disjoint file slice. This
        # (pre-resume-filter) list is this rank's FULL responsibility — kept
        # separately so the FAD completeness check below always looks at the
        # whole shard, not just whatever a single resumed link processed.
        my_full_shard = (
            audio_files[local_rank::world_size] if single_ckpt_sharding else audio_files
        )
        my_audio_files = my_full_shard
        if single_ckpt_sharding:
            info(f"Rank {local_rank}: {len(my_audio_files)}/{len(audio_files)} files", prefix="EVAL")

        # Per-file resume: skip files whose PANN embedding (the last artifact
        # written per file, see below) is already on disk from an earlier,
        # killed run. Only in the full metric mode — the single-metric-only
        # modes keep their existing whole-checkpoint _done gate.
        _full_mode = not (args.cdpam_only or args.sdr_only or args.fad_only
                          or args.fad_gud_only or args.new_metrics_only)
        if args.resume and _full_mode:
            _pann_dir = pred_root / PANN_NAME
            _finished = {f.stem for f in _pann_dir.glob("*.npy")} if _pann_dir.is_dir() else set()
            if _finished:
                _before = len(my_audio_files)
                my_audio_files = [f for f in my_audio_files if f.stem not in _finished]
                info(f"resume: {_before - len(my_audio_files)} già fatti, "
                     f"restano {len(my_audio_files)}", prefix="EVAL")

        dataset = _AudioDataset(my_audio_files, sr, ch, cache_dir=cache_dir)
        loader  = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=lambda b: b[0],
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        processed = 0
        skipped = 0
        t0 = time.time()

        with torch.no_grad():
            for wav, stem in tqdm(loader, desc=ckpt_path.stem,
                                  dynamic_ncols=True, mininterval=10.0,
                                  file=sys.stdout):
                if wav is None:
                    skipped += 1; continue

                try:
                    wav_padded, orig_len = pad_audio_for_swin(wav, hop_length, num_downsamples)
                    wav_gpu = wav_padded.unsqueeze(0).to(device)             # [1, C, T_pad]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                    latents = codec.encode(wav_gpu, deterministic=args.deterministic)  # [1, D, T_lat]
                    decoded = codec.decode(latents,
                                           target_length=wav_padded.shape[-1])  # [1, C, T_pad]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    infer_sec = time.perf_counter() - _t0    # pure encode→decode, no metric cost
                    n       = min(orig_len, decoded.shape[-1])
                    wav_ref = wav[..., :n]                                   # [C, n]  cpu
                    pred    = decoded[0, ..., :n].cpu().float()              # [C, n]
                except Exception as e:
                    err(f"Inference error {stem}: {e}", prefix="EVAL")
                    skipped += 1; continue

                if (not args.cdpam_only and not args.fad_only and not args.fad_gud_only
                        and not args.new_metrics_only):
                    _append_csv(parts_dir / f"timing{suf}.csv",
                               ["file", "infer_sec", "audio_sec"],
                               {"file": stem, "infer_sec": infer_sec, "audio_sec": orig_len / sr})
                    sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
                    _append_csv(parts_dir / f"spectral{suf}.csv",
                               ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"],
                               {"file": stem, "si_sdr": sisdr_val, "sdr": sdr_val,
                                **spectral_losses(wav_ref, pred)})
                    # Opt-in stereo-imaging metrics (needs true stereo ref+pred).
                    if (args.compute_ms_metrics
                            and wav_ref.shape[0] == 2 and pred.shape[0] == 2):
                        try:
                            _append_csv(parts_dir / f"ms_metrics{suf}.csv", _MS_COLS,
                                       {"file": stem, **_compute_ms_metrics(wav_ref, pred, sr)})
                        except Exception as e:
                            warn(f"MS metrics {stem}: {e}", prefix="MS")

                # CDPAM (perceptual) — ported from evaluate_swin.py. Runs in full
                # runs and in --cdpam-only; skipped by --skip-cdpam / --sdr-only /
                # --fad-only / --fad-gud-only.
                if (not args.skip_cdpam and not args.sdr_only
                        and not args.fad_only and not args.fad_gud_only
                        and not args.new_metrics_only):
                    try:
                        _append_csv(parts_dir / f"cdpam{suf}.csv", ["file", "cdpam"],
                                   {"file": stem, "cdpam": cdpam_score(wav_ref, pred, sr, device=device)})
                    except Exception as e:
                        warn(f"CDPAM error {stem}: {e}", prefix="CDPAM")

                # FAD / CLAP embeddings run under their OWN guard (mirrors
                # evaluate_sota.py): the inner `if`s below select which embeddings
                # to compute, so --fad-only / --fad-gud-only work instead of being
                # silently skipped by the spectral guard above.
                if not args.cdpam_only and not args.sdr_only:
                    # CLAP cosine (skipped in --fad-only and --fad-gud-only) + FAD embeddings
                    try:
                        from compute_clap_score import embed_clap_gud
                        from types import SimpleNamespace
                        
                        from utils import atomic_save_npy

                        if not args.fad_only and not args.fad_gud_only and not args.new_metrics_only:
                            t_cm = load_or_embed(clap_music_ml, embed_clap, wav_ref, sr, device,
                                                 target_cache_path(cache_dir, clap_music_ml.name, stem))
                            p_cm = embed_clap(clap_music_ml, pred, sr, device)
                            _append_csv(parts_dir / f"clap_music{suf}.csv", ["file", "cosine"],
                                       {"file": stem,
                                        "cosine": float(cosine_sim(t_cm.mean(0), p_cm.mean(0)))})

                            t_ca = load_or_embed(clap_audio_ml, embed_clap, wav_ref, sr, device,
                                                 target_cache_path(cache_dir, clap_audio_ml.name, stem))
                            p_ca = embed_clap(clap_audio_ml, pred, sr, device)
                            _append_csv(parts_dir / f"clap_audio{suf}.csv", ["file", "cosine"],
                                       {"file": stem,
                                        "cosine": float(cosine_sim(t_ca.mean(0), p_ca.mean(0)))})

                        # MERT layer-4 framewise: feeds fad_mert (populate target cache +
                        # save pred .npy). Full run + --fad-only (not --new-metrics-only /
                        # --fad-gud-only).
                        if not args.fad_gud_only and not args.new_metrics_only:
                            load_or_embed(mert_ml, embed_mert_framewise, wav_ref, sr, device,
                                          target_cache_path(cache_dir, mert_ml.name, stem))
                            p_mert = embed_mert_framewise(mert_ml, pred, sr, device)
                            atomic_save_npy(pred_root / mert_ml.name / f"{stem}.npy", p_mert.astype(np.float16))

                        if not args.fad_only and not args.new_metrics_only:
                            load_or_embed(None, lambda ml, w, s, d: embed_clap_gud(w, s, d), wav_ref, sr, device,
                                          target_cache_path(cache_dir, "clap-laion-audio-gud", stem))
                            p_gud = embed_clap_gud(pred, sr, device)
                            atomic_save_npy(pred_root / "clap-laion-audio-gud" / f"{stem}.npy", p_gud.astype(np.float16))

                        # PANN Cnn14 whole-file: feeds fad_pann. Full run + --new-metrics-only.
                        if not args.fad_only and not args.fad_gud_only:
                            load_or_embed(None, lambda ml, w, s, d: embed_pann(w, s, d), wav_ref, sr, device,
                                          target_cache_path(cache_dir, PANN_NAME, stem))
                            p_pann = embed_pann(pred, sr, device)
                            atomic_save_npy(pred_root / PANN_NAME / f"{stem}.npy", p_pann.astype(np.float16))
                    except Exception as e:
                        warn(f"Embedding error {stem}: {e}", prefix="EMBED")

                processed += 1
                if (processed + skipped) % 50 == 0:
                    info(f"{processed+skipped}/{len(my_audio_files)} "
                         f"({time.time()-t0:.0f}s)", prefix="EVAL")

        ok(f"Done — processed: {processed}  skipped: {skipped}", prefix="EVAL")

        # ── Dedup-merge the incrementally-appended parts into the real metrics
        # CSVs (per-rank suffix in sharding mode). Safe under resume (a file's
        # row was appended at most once, since resume skips it once done) and
        # under reruns of the non-resumable single-metric modes (dedup keeps
        # the first occurrence, so a rerun's fresh rows don't duplicate).
        for _name, _cols in (
            ("spectral", ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"]),
            ("cdpam", ["file", "cdpam"]),
            ("clap_music", ["file", "cosine"]),
            ("clap_audio", ["file", "cosine"]),
            ("timing", ["file", "infer_sec", "audio_sec"]),
            ("ms_metrics", _MS_COLS),
        ):
            _n = _finalize_csv(parts_dir, metrics_dir, _name, _cols, suf)
            if _n:
                ok(f"{_name}{suf}.csv written ({_n} rows)", prefix="EVAL")

        # FAD is a corpus-level Frechet distance, not a per-file metric, so it
        # can't be resumed the same way as the CSVs above — it can only be
        # computed once every file's embedding actually exists on disk. Check
        # fresh (not from what THIS run processed): correct whether an
        # embedding was written now or by an earlier, killed run, and
        # naturally excludes any file whose embedding failed this run.
        def _completed(candidates: list[Path]) -> list[Path]:
            if args.fad_only:
                d = pred_root / mert_ml.name
            elif args.fad_gud_only:
                d = pred_root / "clap-laion-audio-gud"
            else:                       # full mode + --new-metrics-only
                d = pred_root / PANN_NAME
            have = {p.stem for p in d.glob("*.npy")} if d.is_dir() else set()
            return [f for f in candidates if f.stem in have]

        fad_files = _completed(my_full_shard) if not (args.cdpam_only or args.sdr_only) else []

        if single_ckpt_sharding:
            if fad_files:
                # This rank's completed files (embedding on disk right now) —
                # correct regardless of which link, this one or an earlier
                # killed one, actually finished which file.
                import json as _json
                (metrics_dir / f"_shard_rk{local_rank}_files.json").write_text(
                    _json.dumps([str(p) for p in fad_files])
                )

            ok(f"Rank {local_rank}: waiting at barrier…", prefix="EVAL")
            _file_barrier(metrics_dir, local_rank, world_size)

            if local_rank == 0:
                _merge_shards(metrics_dir, "spectral",
                              ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"], world_size)
                _merge_shards(metrics_dir, "cdpam",      ["file", "cdpam"],  world_size)
                _merge_shards(metrics_dir, "clap_music", ["file", "cosine"], world_size)
                _merge_shards(metrics_dir, "clap_audio", ["file", "cosine"], world_size)
                _merge_shards(metrics_dir, "timing", ["file", "infer_sec", "audio_sec"], world_size)
                _merge_shards(metrics_dir, "ms_metrics", _MS_COLS, world_size)
                ok("CSV shards merged.", prefix="EVAL")

                # FAD: compute on all saved embeddings
                if cache_dir:
                    all_fad_files: list[Path]  = []
                    for r in range(world_size):
                        ff = metrics_dir / f"_shard_rk{r}_files.json"
                        if ff.exists():
                            import json as _json
                            all_fad_files.extend(
                                [Path(p) for p in _json.loads(ff.read_text())]
                            )
                            ff.unlink()
                    if all_fad_files:
                        from types import SimpleNamespace
                        if not args.fad_gud_only and not args.new_metrics_only:
                            m_paths = [pred_root / mert_ml.name / f"{f.stem}.npy" for f in all_fad_files]
                            compute_fad_from_embeddings(
                                mert_ml, all_fad_files, m_paths,
                                cache_dir, metrics_dir / "fad_mert.csv", "MERT-v1-95M-4")
                            ok("fad_mert.csv written (merged)", prefix="EVAL")
                            _purge_pred_embeddings(pred_root, mert_ml.name)
                        if not args.fad_only and not args.new_metrics_only:
                            c_paths = [pred_root / "clap-laion-audio-gud" / f"{f.stem}.npy" for f in all_fad_files]
                            compute_fad_from_embeddings(
                                SimpleNamespace(name="clap-laion-audio-gud"), all_fad_files, c_paths,
                                cache_dir, metrics_dir / "fad_gudgud.csv", "clap-laion-audio-gud")
                            ok("fad_gudgud.csv written (merged)", prefix="EVAL")
                            _purge_pred_embeddings(pred_root, "clap-laion-audio-gud")
                        if not args.fad_only and not args.fad_gud_only:
                            pann_paths = [pred_root / PANN_NAME / f"{f.stem}.npy" for f in all_fad_files]
                            compute_fad_from_embeddings(
                                SimpleNamespace(name=PANN_NAME), all_fad_files, pann_paths,
                                cache_dir, metrics_dir / "fad_pann.csv", PANN_NAME)
                            ok("fad_pann.csv written (merged)", prefix="EVAL")
                            _purge_pred_embeddings(pred_root, PANN_NAME)

                if (not args.cdpam_only and not args.fad_only and not args.fad_gud_only
                        and not args.new_metrics_only):
                    _write_varlen_stamp(metrics_dir, codec)
                    (metrics_dir / "_done").touch()
                    ok(f"_done → {metrics_dir}  (varlen={codec.varlen_mode})", prefix="EVAL")
        else:
            # Normal (non-sharding) path: compute FAD directly, write _done
            if cache_dir and fad_files:
                from types import SimpleNamespace
                if not args.fad_gud_only and not args.new_metrics_only:
                    m_paths = [pred_root / mert_ml.name / f"{f.stem}.npy" for f in fad_files]
                    compute_fad_from_embeddings(mert_ml, fad_files, m_paths, cache_dir,
                                                metrics_dir / "fad_mert.csv", "MERT-v1-95M-4")
                    ok("fad_mert.csv written", prefix="EVAL")
                    _purge_pred_embeddings(pred_root, mert_ml.name)
                if not args.fad_only and not args.new_metrics_only:
                    c_paths = [pred_root / "clap-laion-audio-gud" / f"{f.stem}.npy" for f in fad_files]
                    compute_fad_from_embeddings(SimpleNamespace(name="clap-laion-audio-gud"), fad_files, c_paths, cache_dir,
                                                metrics_dir / "fad_gudgud.csv", "clap-laion-audio-gud")
                    ok("fad_gudgud.csv written", prefix="EVAL")
                    _purge_pred_embeddings(pred_root, "clap-laion-audio-gud")
                if not args.fad_only and not args.fad_gud_only:
                    pann_paths = [pred_root / PANN_NAME / f"{f.stem}.npy" for f in fad_files]
                    compute_fad_from_embeddings(SimpleNamespace(name=PANN_NAME), fad_files, pann_paths, cache_dir,
                                                metrics_dir / "fad_pann.csv", PANN_NAME)
                    ok("fad_pann.csv written", prefix="EVAL")
                    _purge_pred_embeddings(pred_root, PANN_NAME)
            if (not args.cdpam_only and not args.fad_only and not args.fad_gud_only
                    and not args.new_metrics_only):
                _write_varlen_stamp(metrics_dir, codec)
                (metrics_dir / "_done").touch()
                ok(f"_done → {metrics_dir}  (varlen={codec.varlen_mode})", prefix="EVAL")

        del codec
        torch.cuda.empty_cache()

    ok(f"Rank {local_rank} done.", prefix="EVAL")


if __name__ == "__main__":
    main()
