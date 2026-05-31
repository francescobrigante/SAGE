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
from compute_fad import embed_mert, compute_fad_from_embeddings
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
    p.add_argument("--resume",          action="store_true",
                   help="Skip checkpoint if metrics/_done already exists.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── SLURM rank → GPU assignment ────────────────────────────
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    all_ckpts = [Path(c).expanduser().resolve() for c in args.checkpoint]
    my_ckpts  = all_ckpts[local_rank::world_size]

    if not my_ckpts:
        info(f"Rank {local_rank}: no checkpoints assigned — exiting.", prefix="EVAL")
        return

    info(f"Rank {local_rank}/{world_size} on {device} — "
         f"{len(my_ckpts)}/{len(all_ckpts)} checkpoints", prefix="EVAL")

    target_dir  = Path(args.target_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    cache_dir   = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    if not target_dir.is_dir():
        err(f"Target directory not found: {target_dir}"); return
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        info(f"Cache dir: {cache_dir}", prefix="EVAL")

    audio_exts  = {(e if e.startswith(".") else f".{e}").lower()
                   for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files found in {target_dir}"); return
    ok(f"Files to evaluate: {len(audio_files)}", prefix="EVAL")

    stem_to_file: dict[str, Path] = {f.stem: f for f in audio_files}

    if not args.cdpam_only and not args.sdr_only:
        # Load CLAP + MERT once per rank — heavy, reused across all checkpoints on this rank
        info("Loading CLAP (music/audio) and MERT models...", prefix="EVAL")
        clap_music_ml = CLAPLaionModel("music")
        clap_audio_ml = CLAPLaionModel("audio")
        mert_ml       = MERTModel()
        for _ml in (clap_music_ml, clap_audio_ml, mert_ml):
            with silence_output():
                _ml.load_model()
            _ml.model.to(device)
        ok("Embedding models ready.", prefix="EVAL")

    for ckpt_path in my_ckpts:
        metrics_dir = output_root / ckpt_path.stem / "metrics"
        ok(f"=== {ckpt_path.name} (rank {local_rank}) ===", prefix="EVAL")

        if args.cdpam_only and (metrics_dir / "cdpam.csv").exists():
            info("[RESUME] cdpam.csv already exists — skipping.", prefix="EVAL"); continue
        elif not args.cdpam_only and args.resume and (metrics_dir / "_done").exists():
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

        dataset = _AudioDataset(audio_files, sr, ch, cache_dir=cache_dir)
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
        mert_pred_embs:   list[np.ndarray] = []
        clap_a_pred_embs: list[np.ndarray] = []
        fad_files:        list[Path]       = []
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
                    latents = codec.encode(wav_gpu)                           # [1, D, T_lat]
                    decoded = codec.decode(latents,
                                           target_length=wav_padded.shape[-1])  # [1, C, T_pad]
                    n       = min(orig_len, decoded.shape[-1])
                    wav_ref = wav[..., :n]                                   # [C, n]  cpu
                    pred    = decoded[0, ..., :n].cpu().float()              # [C, n]
                except Exception as e:
                    err(f"Inference error {stem}: {e}", prefix="EVAL")
                    skipped += 1; continue

                if not args.cdpam_only:
                    sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
                    spectral_rows.append({
                        "file":      stem,
                        "si_sdr":    sisdr_val,
                        "sdr":       sdr_val,
                        "stft_loss": float(stft_loss(wav_ref, pred)),
                    })

                if not args.skip_cdpam and not args.sdr_only:
                    try:
                        cdpam_rows.append({"file": stem,
                                           "cdpam": cdpam_score(wav_ref, pred, sr, device=device)})
                    except Exception as e:
                        warn(f"CDPAM error {stem}: {e}", prefix="EVAL")

                if not args.cdpam_only and not args.sdr_only:
                    # CLAP cosine + FAD embeddings
                    try:
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
                        clap_a_pred_embs.append(p_ca)
    
                        load_or_embed(mert_ml, embed_mert, wav_ref, sr, device,
                                      target_cache_path(cache_dir, mert_ml.name, stem))
                        mert_pred_embs.append(embed_mert(mert_ml, pred, sr, device))
                        fad_files.append(stem_to_file[stem])
    
                    except Exception as e:
                        warn(f"Embedding error {stem}: {e}", prefix="EMBED")

                if (len(spectral_rows) + skipped) % 50 == 0:
                    info(f"{len(spectral_rows)+skipped}/{len(audio_files)} "
                         f"({time.time()-t0:.0f}s)", prefix="EVAL")

        ok(f"Done — processed: {len(spectral_rows)}  skipped: {skipped}", prefix="EVAL")

        if spectral_rows:
            write_csv(metrics_dir / "spectral.csv",
                      ["file", "si_sdr", "sdr", "stft_loss"], spectral_rows)
            ok(f"spectral.csv written ({len(spectral_rows)} rows)", prefix="EVAL")
        if cdpam_rows:
            write_csv(metrics_dir / "cdpam.csv", ["file", "cdpam"], cdpam_rows)
            ok(f"cdpam.csv written", prefix="EVAL")
        if clap_music_rows:
            write_csv(metrics_dir / "clap_music.csv", ["file", "cosine"], clap_music_rows)
            ok(f"clap_music.csv written", prefix="EVAL")
        if clap_audio_rows:
            write_csv(metrics_dir / "clap_audio.csv", ["file", "cosine"], clap_audio_rows)
            ok(f"clap_audio.csv written", prefix="EVAL")
        if cache_dir and mert_pred_embs:
            compute_fad_from_embeddings(mert_ml, fad_files, mert_pred_embs, cache_dir,
                                        metrics_dir / "fad_mert.csv", "MERT-v1-95M")
            ok(f"fad_mert.csv written", prefix="EVAL")
        if cache_dir and clap_a_pred_embs:
            compute_fad_from_embeddings(clap_audio_ml, fad_files, clap_a_pred_embs, cache_dir,
                                        metrics_dir / "fad_gudgud.csv", "clap-laion-audio")
            ok(f"fad_gudgud.csv written", prefix="EVAL")

        if not args.cdpam_only:
            (metrics_dir / "_done").touch()
        del codec
        torch.cuda.empty_cache()

    ok(f"Rank {local_rank} done.", prefix="EVAL")


if __name__ == "__main__":
    main()
