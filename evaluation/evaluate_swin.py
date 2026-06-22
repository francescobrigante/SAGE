#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate_swin.py
# Hann-WOLA evaluation for Swin C-VAE — SI-SDR, STFT loss, CDPAM,
# CLAP cosine similarity, FAD (MERT + CLAP-audio).
#
# Each file is chunked with 25% overlap; all chunks are batched into a single
# GPU encode/decode call, then stitched with Weighted Overlap-Add using Hann
# windows. Metrics are computed on the full stitched reconstruction — no
# per-chunk averaging.
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
from config import (
    DATA_PATH,
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FILES,
    FMA_METADATA,
)
from losses import compute_sdr_and_sisdr, stft_loss, cdpam_score

from utils import (write_csv, collect_fma_files, get_expected_frames,
                   atomic_save_npy, load_or_embed, target_cache_path,
                   silence_output)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import embed_mert_framewise, compute_fad_from_embeddings
from fadtk.model_loader import CLAPLaionModel, MERTModel

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]


# ── Dataset ───────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Load audio files; returns (waveform [C, T], stem).

    Resampled waveforms are cached as float32 .npy files when cache_dir is set,
    avoiding repeated MP3 decode + resample for every checkpoint.
    """

    def __init__(self, files: list[Path], target_sr: int,
                 target_ch: int, cache_dir: Optional[Path] = None):
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


# ── WOLA inference ────────────────────────────────────────────

def swin_infer_wola(
    codec,
    wav: torch.Tensor,
    chunk_samples: int,
    overlap_samples: int,
    deterministic: bool = False,
) -> torch.Tensor:
    """Batch encode+decode [C, T] with Hann WOLA stitching.

    All chunks from this file are stacked into one GPU call, then stitched
    with Weighted Overlap-Add (synthesis window: Hann fade at edges, flat
    center). Since fade_in[k] + fade_out[k] == 1 for every k in the overlap
    region, the normalised sum recovers the signal without amplitude loss.

    Args:
        deterministic: if True, encode returns μ instead of sampled z.

    Returns reconstructed [C, T] on CPU (same length as input).
    """
    C, T = wav.shape
    ae   = codec.autoencoder

    chunk_ranges, T_padded = ae._plan_chunks(T, chunk_samples, overlap_samples)
    if not chunk_ranges:
        return wav.clone()

    wav_padded = F.pad(wav, (0, T_padded - T)) if T_padded > T else wav.clone()

    # Stack all chunks → single GPU forward pass              [N, C, chunk_samples]
    chunks_gpu = torch.stack(
        [wav_padded[:, s:e] for s, e in chunk_ranges]
    ).to(codec.device)

    latents = codec.encode(chunks_gpu, deterministic=deterministic)     # [N, D, T_lat]
    decoded = codec.decode(
        latents, target_length=chunk_samples
    ).cpu().float()                                                     # [N, C, chunk_samples]

    # Synthesis window: Hann fade at edges, ones in the flat centre
    fade_in, fade_out = ae._hann_crossfade_windows(overlap_samples)
    win = torch.ones(chunk_samples)
    if fade_in is not None:
        win[:overlap_samples]                 = fade_in
        win[chunk_samples - overlap_samples:] = fade_out

    out  = torch.zeros(C, T_padded)
    norm = torch.zeros(T_padded)

    for i, (s, e) in enumerate(chunk_ranges):
        out[:, s:e]  += decoded[i] * win.unsqueeze(0)                  # [C, chunk_samples]
        norm[s:e]    += win

    out = out / norm.clamp(min=1e-8).unsqueeze(0)
    return out[:, :T]                                                   # [C, T]


# ── Argument parsing ──────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hann-WOLA Swin C-VAE evaluation — SI-SDR, STFT, CDPAM, CLAP, FAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True, nargs="+", type=Path,
                   help="One or more Swin C-VAE .ckpt files (evaluated sequentially).")
    p.add_argument("--target-dir",   default=str(DATA_PATH))
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--device",       default=DEFAULT_DEVICE)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--max-files",    type=int, default=DEFAULT_MAX_FILES)
    p.add_argument("--fma-csv-path", default=FMA_METADATA)
    p.add_argument("--extensions",   default=",".join(DEFAULT_AUDIO_EXTENSIONS))
    p.add_argument("--cache-dir",    type=Path, default=None,
                   help="Cache resampled waveforms and target embeddings (reused across checkpoints).")
    p.add_argument("--skip-cdpam",   action="store_true")
    p.add_argument("--fad-only",     action="store_true",
                   help="Compute ONLY fad_mert (framewise MERT, layer 4): skip "
                        "SI-SDR/STFT, CDPAM, CLAP cosine and fad_gudgud. Overwrites "
                        "fad_mert.csv, leaves every other metric CSV untouched.")
    p.add_argument("--fad-gud-only", action="store_true",
                   help="Compute ONLY fad_gudgud (whole-file CLAP, 48kHz, float): skip "
                        "all other metrics. Overwrites fad_gudgud.csv.")
    p.add_argument("--resume",       action="store_true",
                   help="Skip checkpoint if metrics/_done already exists.")
    p.add_argument("--deterministic", action="store_true",
                   help="Encode with VAE posterior mean μ instead of sampled z. "
                        "Improves SI-SDR/STFT at the cost of a tiny stochasticity "
                        "that the decoder was trained to ignore. Default: False.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── SLURM rank → GPU assignment (mirrors evaluate_swin_varT.py) ──
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

    if not target_dir.is_dir():
        err(f"Target directory not found: {target_dir}"); return

    audio_exts  = {(e if e.startswith(".") else f".{e}").lower()
                   for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files found in {target_dir}"); return
    ok(f"Files to evaluate: {len(audio_files)}", prefix="EVAL")

    stem_to_file: dict[str, Path] = {f.stem: f for f in audio_files}

    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        info(f"Cache dir: {cache_dir}", prefix="EVAL")

    # Load MERT (layer 4, framewise/fadtk-standard) once — heavy, reused across all
    # checkpoints. CLAP is only needed for the non-FAD metrics → skipped in --fad-only.
    info("Loading embedding models...", prefix="EVAL")
    embed_models = []
    if not args.fad_gud_only:
        mert_ml       = MERTModel(layer=4)
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
        ok(f"=== {ckpt_path.name} ===", prefix="EVAL")

        if args.fad_only:
            if args.resume and (metrics_dir / "fad_mert.csv").exists():
                info("[RESUME] fad_mert.csv exists — skipping.", prefix="EVAL"); continue
        elif args.fad_gud_only:
            if args.resume and (metrics_dir / "fad_gudgud.csv").exists():
                info("[RESUME] fad_gudgud.csv exists — skipping.", prefix="EVAL"); continue
        elif args.resume and (metrics_dir / "_done").exists():
            info("[RESUME] already done — skipping.", prefix="EVAL"); continue

        try:
            codec = EuleroEncodeDecode(ckpt_path, device=device)
        except Exception as e:
            err(f"Failed to load {ckpt_path.name}: {e}"); continue

        sr: int = codec.sample_rate    or 44100
        ch: int = codec.audio_channels or 2

        expected_frames = (get_expected_frames(codec.autoencoder.encoder)
                           if hasattr(codec.autoencoder, "encoder") else None)
        stft_cfg      = getattr(codec.autoencoder, "_stft_config", None)
        hop           = stft_cfg.hop_length if (expected_frames and stft_cfg) else None
        chunk_samples = ((expected_frames - 1) * hop) if (expected_frames and hop) else None

        if chunk_samples is None:
            err(f"Cannot determine chunk_samples from {ckpt_path.name}."); continue

        overlap_samples = chunk_samples // 4  # 25% overlap — standard WOLA
        ok(f"sr={sr} ch={ch}  chunk={chunk_samples}sa ({chunk_samples/sr:.2f}s)  "
           f"overlap={overlap_samples}sa ({overlap_samples/sr:.3f}s)", prefix="EVAL")

        metrics_dir.mkdir(parents=True, exist_ok=True)

        dataset = _AudioDataset(audio_files, sr, ch, cache_dir=cache_dir)
        loader  = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=lambda b: b[0],
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        spectral_rows:    list[dict]       = []
        cdpam_rows:       list[dict]       = []
        clap_music_rows:  list[dict]       = []
        clap_audio_rows:  list[dict]       = []
        mert_pred_embs:   list[np.ndarray] = []
        clap_gud_pred_embs: list[np.ndarray] = []
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
                    # Single GPU call — all overlapping chunks batched together
                    pred = swin_infer_wola(codec, wav, chunk_samples, overlap_samples,
                                           deterministic=args.deterministic)
                    # pred: [C, T]  cpu, same length as wav
                except Exception as e:
                    err(f"Inference error {stem}: {e}", prefix="EVAL")
                    skipped += 1; continue

                if not args.fad_only and not args.fad_gud_only:
                    sdr_val, sisdr_val = compute_sdr_and_sisdr(wav, pred)
                    spectral_rows.append({
                        "file":      stem,
                        "si_sdr":    sisdr_val,
                        "sdr":       sdr_val,
                        "stft_loss": float(stft_loss(wav, pred)),
                    })

                    if not args.skip_cdpam:
                        try:
                            cdpam_rows.append({"file": stem,
                                               "cdpam": cdpam_score(wav, pred, sr, device=device)})
                        except Exception as e:
                            warn(f"CDPAM error {stem}: {e}", prefix="CDPAM")

                # CLAP cosine (skipped in --fad-only and --fad-gud-only) + FAD embeddings (full audio)
                try:
                    from compute_clap_score import embed_clap_gud
                    from types import SimpleNamespace
                    
                    if not args.fad_only and not args.fad_gud_only:
                        t_cm = load_or_embed(clap_music_ml, embed_clap, wav, sr, device,
                                             target_cache_path(cache_dir, clap_music_ml.name, stem))
                        p_cm = embed_clap(clap_music_ml, pred, sr, device)
                        clap_music_rows.append({"file": stem,
                                                "cosine": float(cosine_sim(t_cm.mean(0), p_cm.mean(0)))})

                        t_ca = load_or_embed(clap_audio_ml, embed_clap, wav, sr, device,
                                             target_cache_path(cache_dir, clap_audio_ml.name, stem))
                        p_ca = embed_clap(clap_audio_ml, pred, sr, device)
                        clap_audio_rows.append({"file": stem,
                                                "cosine": float(cosine_sim(t_ca.mean(0), p_ca.mean(0)))})

                    if not args.fad_gud_only:
                        load_or_embed(mert_ml, embed_mert_framewise, wav, sr, device,
                                      target_cache_path(cache_dir, mert_ml.name, stem))
                        mert_pred_embs.append(embed_mert_framewise(mert_ml, pred, sr, device))
                        
                    if not args.fad_only:
                        load_or_embed(None, lambda ml, w, s, d: embed_clap_gud(w, s, d), wav, sr, device,
                                      target_cache_path(cache_dir, "clap-laion-audio-gud", stem))
                        clap_gud_pred_embs.append(embed_clap_gud(pred, sr, device))
                        
                    fad_files.append(stem_to_file[stem])

                except Exception as e:
                    warn(f"Embedding error {stem}: {e}", prefix="EMBED")

                if (len(spectral_rows) + skipped) % 50 == 0:
                    elapsed = time.time() - t0
                    info(f"{len(spectral_rows)+skipped}/{len(audio_files)} files  ({elapsed:.0f}s)", prefix="EVAL")

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
                                        metrics_dir / "fad_mert.csv", "MERT-v1-95M-4")
            ok(f"fad_mert.csv written", prefix="EVAL")
        if cache_dir and clap_gud_pred_embs:
            from types import SimpleNamespace
            compute_fad_from_embeddings(SimpleNamespace(name="clap-laion-audio-gud"), fad_files, clap_gud_pred_embs, cache_dir,
                                        metrics_dir / "fad_gudgud.csv", "clap-laion-audio-gud")
            ok(f"fad_gudgud.csv written", prefix="EVAL")

        if not args.fad_only and not args.fad_gud_only:
            (metrics_dir / "_done").touch()

        del codec
        torch.cuda.empty_cache()

    ok("All checkpoints done.", prefix="EVAL")


if __name__ == "__main__":
    main()
