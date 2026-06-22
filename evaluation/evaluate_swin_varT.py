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
import os
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
from config import DATA_PATH, DEFAULT_AUDIO_EXTENSIONS, DEFAULT_MAX_FILES, FMA_METADATA
from losses import compute_sdr_and_sisdr, stft_loss, cdpam_score

from utils import (write_csv, collect_fma_files, atomic_save_npy,
                   load_or_embed, target_cache_path, silence_output)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import embed_mert_framewise, compute_fad_from_embeddings
from fadtk.model_loader import CLAPLaionModel, MERTModel

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]


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
    import csv as _csv
    all_rows: list[dict] = []
    for r in range(world_size):
        p = metrics_dir / f"{stem}_rk{r}.csv"
        if p.exists():
            with open(p, newline="") as f:
                all_rows.extend(list(_csv.DictReader(f)))
            p.unlink()
    if all_rows:
        write_csv(metrics_dir / f"{stem}.csv", fieldnames, all_rows)


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
    p.add_argument("--resume",          action="store_true",
                   help="Skip checkpoint if metrics/_done already exists.")
    p.add_argument("--deterministic",   action="store_true",
                   help="Encode with VAE posterior mean μ instead of sampled z. "
                        "Improves SI-SDR/STFT at inference time. Default: False.")
    p.add_argument("--shard-files",     action="store_true",
                   help="When a single checkpoint is evaluated across multiple ranks, "
                        "shard audio files across ranks instead of leaving rank 1+ idle. "
                        "Gives ~Nx speedup with N GPUs. All metrics are bit-identical to "
                        "the single-rank run (FAD is merged on rank 0 after a barrier). "
                        "Default: False (backward-compatible).")
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

    stem_to_file: dict[str, Path] = {f.stem: f for f in audio_files}

    if not args.cdpam_only and not args.sdr_only:
        # Load MERT (layer 4, framewise/fadtk-standard) once per rank. CLAP only
        # for the non-FAD metrics → skipped in --fad-only.
        info("Loading embedding models...", prefix="EVAL")
        embed_models = []
        if not args.fad_gud_only:
            mert_ml       = MERTModel(layer=4)   # framewise/fadtk-standard, layer 4
            embed_models.append(mert_ml)
            if not args.fad_only:
                clap_music_ml = CLAPLaionModel("music")
                clap_audio_ml = CLAPLaionModel("audio")
                embed_models.extend([clap_music_ml, clap_audio_ml])
        
        for _ml in embed_models:
            with silence_output():
                _ml.load_model()
            _ml.model.to(device)
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
        elif not args.cdpam_only and not args.fad_only and not args.fad_gud_only and args.resume and (metrics_dir / "_done").exists():
            info("[RESUME] already done — skipping.", prefix="EVAL"); continue

        try:
            codec = EuleroEncodeDecode(ckpt_path, device=device)
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

        # In sharding mode each rank processes a disjoint file slice
        my_audio_files = (
            audio_files[local_rank::world_size] if single_ckpt_sharding else audio_files
        )
        if single_ckpt_sharding:
            info(f"Rank {local_rank}: {len(my_audio_files)}/{len(audio_files)} files", prefix="EVAL")

        dataset = _AudioDataset(my_audio_files, sr, ch, cache_dir=cache_dir)
        loader  = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=lambda b: b[0],
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        spectral_rows:    list[dict]       = []
        cdpam_rows:       list[dict]       = []
        clap_music_rows:  list[dict]       = []
        clap_audio_rows:  list[dict]       = []
        fad_files:        list[Path]       = []
        skipped = 0
        t0 = time.time()
        
        parts_dir = metrics_dir / "parts"
        pred_root = parts_dir / "pred"
        parts_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for wav, stem in tqdm(loader, desc=ckpt_path.stem,
                                  dynamic_ncols=True, mininterval=10.0,
                                  file=sys.stdout):
                if wav is None:
                    skipped += 1; continue

                try:
                    wav_padded, orig_len = pad_audio_for_swin(wav, hop_length, num_downsamples)
                    wav_gpu = wav_padded.unsqueeze(0).to(device)             # [1, C, T_pad]
                    latents = codec.encode(wav_gpu, deterministic=args.deterministic)  # [1, D, T_lat]
                    decoded = codec.decode(latents,
                                           target_length=wav_padded.shape[-1])  # [1, C, T_pad]
                    n       = min(orig_len, decoded.shape[-1])
                    wav_ref = wav[..., :n]                                   # [C, n]  cpu
                    pred    = decoded[0, ..., :n].cpu().float()              # [C, n]
                except Exception as e:
                    err(f"Inference error {stem}: {e}", prefix="EVAL")
                    skipped += 1; continue

                if not args.cdpam_only and not args.fad_only and not args.fad_gud_only:
                    sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
                    spectral_rows.append({
                        "file":      stem,
                        "si_sdr":    sisdr_val,
                        "sdr":       sdr_val,
                        "stft_loss": float(stft_loss(wav_ref, pred)),
                    })

                # CDPAM (perceptual) — ported from evaluate_swin.py. Runs in full
                # runs and in --cdpam-only; skipped by --skip-cdpam / --sdr-only /
                # --fad-only / --fad-gud-only.
                if (not args.skip_cdpam and not args.sdr_only
                        and not args.fad_only and not args.fad_gud_only):
                    try:
                        cdpam_rows.append({"file": stem,
                                           "cdpam": cdpam_score(wav_ref, pred, sr, device=device)})
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
                        
                        if not args.fad_only and not args.fad_gud_only:
                            t_cm = load_or_embed(clap_music_ml, embed_clap, wav_ref, sr, device,
                                                 target_cache_path(cache_dir, clap_music_ml.name, stem))
                            p_cm = embed_clap(clap_music_ml, pred, sr, device)
                            clap_music_rows.append({"file": stem,
                                                    "cosine": float(cosine_sim(t_cm.mean(0), p_cm.mean(0)))})

                            t_ca = load_or_embed(clap_audio_ml, embed_clap, wav_ref, sr, device,
                                                 target_cache_path(cache_dir, clap_audio_ml.name, stem))
                            p_ca = embed_clap(clap_audio_ml, pred, sr, device)
                            clap_audio_rows.append({"file": stem,
                                                    "cosine": float(cosine_sim(t_ca.mean(0), p_ca.mean(0)))})

                        if not args.fad_gud_only:
                            load_or_embed(mert_ml, embed_mert_framewise, wav_ref, sr, device,
                                          target_cache_path(cache_dir, mert_ml.name, stem))
                            p_mert = embed_mert_framewise(mert_ml, pred, sr, device)
                            from utils import atomic_save_npy
                            atomic_save_npy(pred_root / mert_ml.name / f"{stem}.npy", p_mert.astype(np.float16))
                        
                        if not args.fad_only:
                            load_or_embed(None, lambda ml, w, s, d: embed_clap_gud(w, s, d), wav_ref, sr, device,
                                          target_cache_path(cache_dir, "clap-laion-audio-gud", stem))
                            p_gud = embed_clap_gud(pred, sr, device)
                            from utils import atomic_save_npy
                            atomic_save_npy(pred_root / "clap-laion-audio-gud" / f"{stem}.npy", p_gud.astype(np.float16))
                        
                        fad_files.append(stem_to_file[stem])
                    except Exception as e:
                        warn(f"Embedding error {stem}: {e}", prefix="EMBED")

                if (len(spectral_rows) + skipped) % 50 == 0:
                    info(f"{len(spectral_rows)+skipped}/{len(audio_files)} "
                         f"({time.time()-t0:.0f}s)", prefix="EVAL")

        ok(f"Done — processed: {len(spectral_rows)}  skipped: {skipped}", prefix="EVAL")

        # ── Write per-file metrics (per-rank suffix in sharding mode) ──────────
        suf = f"_rk{local_rank}" if single_ckpt_sharding else ""
        if spectral_rows:
            write_csv(metrics_dir / f"spectral{suf}.csv",
                      ["file", "si_sdr", "sdr", "stft_loss"], spectral_rows)
            ok(f"spectral{suf}.csv written ({len(spectral_rows)} rows)", prefix="EVAL")
        if cdpam_rows:
            write_csv(metrics_dir / f"cdpam{suf}.csv", ["file", "cdpam"], cdpam_rows)
            ok(f"cdpam{suf}.csv written", prefix="EVAL")
        if clap_music_rows:
            write_csv(metrics_dir / f"clap_music{suf}.csv", ["file", "cosine"], clap_music_rows)
            ok(f"clap_music{suf}.csv written", prefix="EVAL")
        if clap_audio_rows:
            write_csv(metrics_dir / f"clap_audio{suf}.csv", ["file", "cosine"], clap_audio_rows)
            ok(f"clap_audio{suf}.csv written", prefix="EVAL")

        if single_ckpt_sharding:
            if fad_files:
                import json as _json
                (metrics_dir / f"_shard_rk{local_rank}_files.json").write_text(
                    _json.dumps([str(p) for p in fad_files])
                )

            ok(f"Rank {local_rank}: waiting at barrier…", prefix="EVAL")
            _file_barrier(metrics_dir, local_rank, world_size)

            if local_rank == 0:
                _merge_shards(metrics_dir, "spectral",
                              ["file", "si_sdr", "sdr", "stft_loss"], world_size)
                _merge_shards(metrics_dir, "cdpam",      ["file", "cdpam"],  world_size)
                _merge_shards(metrics_dir, "clap_music", ["file", "cosine"], world_size)
                _merge_shards(metrics_dir, "clap_audio", ["file", "cosine"], world_size)
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
                        if not args.fad_gud_only:
                            m_paths = [pred_root / mert_ml.name / f"{f.stem}.npy" for f in all_fad_files]
                            compute_fad_from_embeddings(
                                mert_ml, all_fad_files, m_paths,
                                cache_dir, metrics_dir / "fad_mert.csv", "MERT-v1-95M-4")
                            ok("fad_mert.csv written (merged)", prefix="EVAL")
                        if not args.fad_only:
                            c_paths = [pred_root / "clap-laion-audio-gud" / f"{f.stem}.npy" for f in all_fad_files]
                            from types import SimpleNamespace
                            compute_fad_from_embeddings(
                                SimpleNamespace(name="clap-laion-audio-gud"), all_fad_files, c_paths,
                                cache_dir, metrics_dir / "fad_gudgud.csv", "clap-laion-audio-gud")
                            ok("fad_gudgud.csv written (merged)", prefix="EVAL")

                if not args.cdpam_only and not args.fad_only and not args.fad_gud_only:
                    (metrics_dir / "_done").touch()
        else:
            # Normal (non-sharding) path: compute FAD directly, write _done
            if cache_dir and fad_files:
                if not args.fad_gud_only:
                    m_paths = [pred_root / mert_ml.name / f"{f.stem}.npy" for f in fad_files]
                    compute_fad_from_embeddings(mert_ml, fad_files, m_paths, cache_dir,
                                                metrics_dir / "fad_mert.csv", "MERT-v1-95M-4")
                    ok("fad_mert.csv written", prefix="EVAL")
                if not args.fad_only:
                    c_paths = [pred_root / "clap-laion-audio-gud" / f"{f.stem}.npy" for f in fad_files]
                    from types import SimpleNamespace
                    compute_fad_from_embeddings(SimpleNamespace(name="clap-laion-audio-gud"), fad_files, c_paths, cache_dir,
                                                metrics_dir / "fad_gudgud.csv", "clap-laion-audio-gud")
                    ok("fad_gudgud.csv written", prefix="EVAL")
            if not args.cdpam_only and not args.fad_only and not args.fad_gud_only:
                (metrics_dir / "_done").touch()

        del codec
        torch.cuda.empty_cache()

    ok(f"Rank {local_rank} done.", prefix="EVAL")


if __name__ == "__main__":
    main()
