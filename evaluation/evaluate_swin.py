#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate_swin.py
# Chunk-based evaluation for Swin C-VAE: SI-SDR, STFT loss, CDPAM.
#
# Audio is split into non-overlapping full chunks; partial last chunk is
# discarded. Metrics computed per chunk, averaged per file.
# Outputs: metrics/spectral.csv and metrics/cdpam.csv (same format as SAO).
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_EVAL_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from ar_spectra.models.inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, info, err
from config import (
    DATA_PATH,
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FILES,
    FMA_METADATA,
)
from losses import si_sdr, stft_loss, cdpam_score
from utils import write_csv, collect_fma_files, get_expected_frames

# numpy 1.24+ removed np.float — patch for CDPAM internals
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]


# ── Dataset ───────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Load audio files; returns (waveform [C, T], stem)."""

    def __init__(self, files: list[Path], target_sr: int, target_ch: int):
        self.files     = files
        self.target_sr = target_sr
        self.target_ch = target_ch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            if wav.shape[0] < self.target_ch:
                wav = wav.repeat(self.target_ch, 1)
            elif wav.shape[0] > self.target_ch:
                wav = wav.mean(0, keepdim=True).expand(self.target_ch, -1).clone()
            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Argument parsing ──────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chunk-based Swin C-VAE evaluation — SI-SDR, STFT loss, CDPAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to Swin C-VAE .ckpt file.")
    p.add_argument("--target-dir",   default=str(DATA_PATH),
                   help="Reference audio directory (FMA test split).")
    p.add_argument("--output-dir",   required=True,
                   help="Per-checkpoint output root; metrics/ sub-dir is created here.")
    p.add_argument("--device",       default=DEFAULT_DEVICE)
    p.add_argument("--num-workers",  type=int, default=4,
                   help="DataLoader prefetch workers.")
    p.add_argument("--max-files",    type=int, default=DEFAULT_MAX_FILES,
                   help="Limit number of files (0 = all).")
    p.add_argument("--fma-csv-path", default=FMA_METADATA)
    p.add_argument("--extensions",   default=",".join(DEFAULT_AUDIO_EXTENSIONS))
    p.add_argument("--skip-cdpam",   action="store_true")
    p.add_argument("--resume",       action="store_true",
                   help="Skip if metrics/spectral.csv already exists.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    device      = torch.device(args.device)
    target_dir  = Path(args.target_dir).expanduser().resolve()
    output_dir  = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"

    if args.resume and (metrics_dir / "spectral.csv").exists():
        info(f"[RESUME] {output_dir.name} already evaluated — skipping.")
        return

    if not target_dir.is_dir():
        err(f"Target directory not found: {target_dir}"); return

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect audio files ────────────────────────────────────
    audio_exts  = {(e if e.startswith(".") else f".{e}").lower()
                   for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files found in {target_dir}"); return
    ok(f"Files to evaluate: {len(audio_files)}", prefix="EVAL")

    # ── Load codec ─────────────────────────────────────────────
    codec = EuleroEncodeDecode(args.checkpoint, device=device)
    sr: int = codec.sample_rate    or 44100
    ch: int = codec.audio_channels or 2
    ok(f"Codec: sr={sr} Hz, channels={ch}", prefix="EVAL")

    expected_frames = (get_expected_frames(codec.autoencoder.encoder)
                       if hasattr(codec.autoencoder, "encoder") else None)
    stft_cfg      = getattr(codec.autoencoder, "_stft_config", None)
    hop           = stft_cfg.hop_length if (expected_frames and stft_cfg) else None
    chunk_samples = ((expected_frames - 1) * hop) if (expected_frames and hop) else None

    if chunk_samples is None:
        err("Could not determine chunk_samples from model. "
            "Check that the checkpoint is a Swin C-VAE model."); return
    info(f"Chunk: {chunk_samples} samples ({chunk_samples / sr:.2f}s)", prefix="EVAL")

    # ── DataLoader ─────────────────────────────────────────────
    dataset = _AudioDataset(audio_files, sr, ch)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    spectral_rows: list[dict] = []
    cdpam_rows:    list[dict] = []
    skipped = 0
    total   = len(loader)

    with torch.no_grad():
        for idx, (wav, stem) in enumerate(tqdm(loader, desc="Evaluating", unit="file",
                                               dynamic_ncols=True, mininterval=10.0)):
            if wav is None:
                skipped += 1; continue

            n_chunks = wav.shape[-1] // chunk_samples
            if n_chunks == 0:
                warn(f"File {stem} shorter than chunk ({chunk_samples} samples), skipping.")
                skipped += 1; continue

            sdrs, stfts, cdpams = [], [], []

            try:
                for i in range(n_chunks):
                    chunk     = wav[:, i * chunk_samples : (i + 1) * chunk_samples]  # [C, S]
                    chunk_gpu = chunk.unsqueeze(0).to(device)                         # [1, C, S]

                    lat      = codec.encode(chunk_gpu)
                    pred     = codec.decode(lat, target_length=chunk_samples)         # [1, C, S]
                    pred_cpu = pred[0].cpu()                                          # [C, S]

                    sdrs.append(si_sdr(chunk, pred_cpu))
                    stfts.append(stft_loss(chunk, pred_cpu))

                    if not args.skip_cdpam:
                        try:
                            cdpams.append(cdpam_score(chunk, pred_cpu, sr, device=device))
                        except Exception as e:
                            warn(f"CDPAM {stem} chunk {i}: {e}", prefix="CDPAM")

            except Exception as e:
                err(f"Inference error {stem}: {e}", prefix="EVAL")
                skipped += 1; continue

            spectral_rows.append({
                "file":      stem,
                "si_sdr":    float(np.mean(sdrs)),
                "stft_loss": float(np.mean(stfts)),
                "n_chunks":  n_chunks,
            })
            if cdpams:
                cdpam_rows.append({"file": stem, "cdpam": float(np.mean(cdpams))})

            if (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"[EVAL] {idx+1}/{total} ({(idx+1)/total*100:.1f}%)", flush=True)

    ok(f"Done — skipped: {skipped}", prefix="EVAL")

    if spectral_rows:
        write_csv(metrics_dir / "spectral.csv",
                  ["file", "si_sdr", "stft_loss", "n_chunks"], spectral_rows)
        ok(f"spectral.csv ({len(spectral_rows)} files)", prefix="EVAL")
    if cdpam_rows:
        write_csv(metrics_dir / "cdpam.csv", ["file", "cdpam"], cdpam_rows)
        ok(f"cdpam.csv ({len(cdpam_rows)} files)", prefix="EVAL")

    ok("Evaluation complete.", prefix="EVAL")


if __name__ == "__main__":
    main()
