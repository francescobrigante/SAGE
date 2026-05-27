#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate_sao.py
# Spectral evaluation for SAO checkpoints — matches training validation_step.
#
# Pipeline per file:
#   preprocess_audio_for_encoder → encode_audio → decode_audio → si_sdr / stft_loss
#
# No cross-correlation alignment. No autocast. One file at a time.
# Identical to stable_audio_tools AutoencoderTrainingWrapper.validation_step.
# =============================================================================

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_EVAL_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_PROJ_ROOT / "src"))
sys.path.insert(0, str(_PROJ_ROOT / "stable_audio_baseline"))

from ar_spectra.models.inference import _extract_autoencoder_state
from ar_spectra.utils.console import ok, warn, info, err
from losses import si_sdr, stft_loss, cdpam_score
from utils import (write_csv, collect_fma_files, atomic_save_npy,
                   load_or_embed, target_cache_path, silence_output)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import embed_mert, compute_fad_from_embeddings
from fadtk.model_loader import CLAPLaionModel, MERTModel

# numpy 1.24+ removed np.float — patch for CDPAM internals
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]


# ── Model loading ─────────────────────────────────────────────

def _load_sao(checkpoint_path: Path, device: torch.device):
    """Build AudioAutoencoder from model_config and load state dict."""
    from stable_audio_tools.models.factory import create_model_from_config

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_config = ckpt.get("model_config")
    if model_config is None:
        raise KeyError(
            f"{checkpoint_path}: no 'model_config' key. "
            "Use evaluate_swin.py for Swin checkpoints."
        )

    autoencoder = create_model_from_config(model_config)
    cleaned = _extract_autoencoder_state(ckpt.get("state_dict", ckpt))
    missing, unexpected = autoencoder.load_state_dict(cleaned, strict=False)
    if missing:
        warn(f"Missing keys: {missing}", prefix="CHECKPOINT")
    if unexpected:
        warn(f"Unexpected keys: {unexpected}", prefix="CHECKPOINT")

    sr = model_config.get("sample_rate") or 44100
    ch = model_config.get("audio_channels") or model_config.get("model", {}).get("io_channels") or 2
    autoencoder.to(device).eval()
    ok(f"Loaded {checkpoint_path.name}  sr={sr}  ch={ch}", prefix="CHECKPOINT")
    return autoencoder, int(sr), int(ch)


# ── Dataset ───────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Load and resample audio files; returns (waveform [C, T], stem).

    If cache_dir is provided, the resampled waveform is persisted as float32
    numpy on first access and reloaded from cache on subsequent runs/checkpoints,
    avoiding repeated MP3 decode + resample for every checkpoint.
    """

    def __init__(self, files: list[Path], target_sr: int,
                 target_channels: int = 2,
                 cache_dir: Optional[Path] = None):
        self.files           = files
        self.target_sr       = target_sr
        self.target_channels = target_channels
        self.cache_dir       = cache_dir

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            # Cache key includes channel count to avoid stale stereo/mono hits
            cache_file = (
                self.cache_dir /
                f"wav_{p.stem}_ch{self.target_channels}_sr{self.target_sr}.npy"
            ) if self.cache_dir is not None else None

            if cache_file is not None and cache_file.exists():
                wav = torch.from_numpy(
                    np.load(cache_file).astype(np.float32)
                )
                return wav, p.stem

            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)

            # Normalise channel count — some FMA-large MP3s have inconsistent
            # stereo/mono headers that cause torchaudio to return wrong shapes.
            C = wav.shape[0]
            if C > self.target_channels:
                wav = wav[:self.target_channels]
            elif C < self.target_channels:
                wav = wav.repeat(
                    (self.target_channels + C - 1) // C, 1
                )[:self.target_channels]

            if cache_file is not None:
                atomic_save_npy(cache_file, wav.numpy())

            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Argument parsing ──────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--fma-csv-path", default=None)
    p.add_argument("--extensions", default=".wav,.flac,.mp3,.ogg")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory to cache resampled target waveforms (.npy). "
                        "Written on first pass; reused for subsequent checkpoints and runs.")
    p.add_argument("--skip-cdpam", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip checkpoint if spectral.csv already exists.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    target_dir  = Path(args.target_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    audio_exts = {(e if e.startswith(".") else f".{e}").lower()
                  for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files in {target_dir}"); return
    ok(f"Files: {len(audio_files)}", prefix="EVAL")

    stem_to_file: dict[str, Path] = {f.stem: f for f in audio_files}

    # Load embedding models once — heavy, reused across all checkpoints
    info("Loading CLAP (music/audio) and MERT models...", prefix="EVAL")
    clap_music_ml = CLAPLaionModel("music")
    clap_audio_ml = CLAPLaionModel("audio")
    mert_ml       = MERTModel()
    for _ml in (clap_music_ml, clap_audio_ml, mert_ml):
        with silence_output():
            _ml.load_model()
        _ml.model.to(device)
    ok("Embedding models ready.", prefix="EVAL")

    for ckpt_path in args.checkpoint:
        ckpt_path   = Path(ckpt_path).expanduser().resolve()
        metrics_dir = output_root / ckpt_path.stem / "metrics"
        ok(f"=== {ckpt_path.name} ===", prefix="EVAL")

        if args.resume and (metrics_dir / "_done").exists():
            info(f"[RESUME] already done — skipping.", prefix="EVAL"); continue

        metrics_dir.mkdir(parents=True, exist_ok=True)
        autoencoder, sr, ch = _load_sao(ckpt_path, device)

        cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            info(f"Target audio cache: {cache_dir}", prefix="EVAL")

        dataset = _AudioDataset(audio_files, sr,
                                target_channels=ch, cache_dir=cache_dir)
        loader  = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=lambda b: b[0],          # returns single (wav, stem)
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
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
            for i, (wav, stem) in enumerate(tqdm(loader, desc=ckpt_path.stem,
                                                  dynamic_ncols=True, mininterval=10.0)):
                if wav is None:
                    skipped += 1; continue

                orig_len = wav.shape[-1]    # samples before padding

                try:
                    # Preprocessing: pad to multiple of downsampling_ratio
                    # (resampling already done in DataLoader)
                    audio_in = autoencoder.preprocess_audio_for_encoder(
                        wav, in_sr=sr         # [1, C, T_padded]
                    ).to(device)

                    # Encode → decode (no autocast, matches training validation_step)
                    latents = autoencoder.encode_audio(audio_in, chunked=False)
                    decoded = autoencoder.decode_audio(latents, chunked=False)  # [1, C, T_padded]

                except Exception as e:
                    err(f"Inference error {stem}: {e}", prefix="EVAL")
                    skipped += 1; continue

                # Trim both to original length — identical to trim_to_shortest in training
                n       = min(orig_len, decoded.shape[-1])
                wav_ref = wav[..., :n]                       # [C, n]
                pred    = decoded[0, ..., :n].cpu().float()  # [C, n]

                spectral_rows.append({
                    "file":      stem,
                    "si_sdr":    si_sdr(wav_ref, pred),
                    "stft_loss": stft_loss(wav_ref, pred),
                })

                if not args.skip_cdpam:
                    try:
                        score = cdpam_score(wav_ref, pred, sr, device=device)
                        cdpam_rows.append({"file": stem, "cdpam": score})
                    except Exception as e:
                        warn(f"CDPAM {stem}: {e}", prefix="CDPAM")

                # CLAP cosine similarity + FAD embeddings (target cached, pred in-memory)
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

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    info(f"{i+1}/{len(audio_files)} files  ({elapsed:.0f}s)", prefix="EVAL")

        ok(f"Done — processed: {len(spectral_rows)}  skipped: {skipped}", prefix="EVAL")

        if spectral_rows:
            write_csv(metrics_dir / "spectral.csv",
                      ["file", "si_sdr", "stft_loss"], spectral_rows)
            ok(f"spectral.csv written", prefix="EVAL")
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

        # Mark this checkpoint as fully evaluated — used by --resume to skip.
        (metrics_dir / "_done").touch()
        ok(f"Sentinel _done written", prefix="EVAL")

        del autoencoder
        torch.cuda.empty_cache()

    ok("All checkpoints done.", prefix="EVAL")


if __name__ == "__main__":
    main()
