"""
evaluation/utils.py
Pure utilities for the in-memory evaluation pipeline.
Contains: I/O helpers, alignment, cache helpers, codec batching.

Loss functions (si_sdr, stft_loss, cdpam_score) live in losses.py.
Embedding functions (embed_clap, embed_mert, cosine_sim) live in
compute_clap_score.py and compute_fad.py respectively.
"""

from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sage.utils.console import warn


# ── I/O helpers ───────────────────────────────────────────────

# ── Channel selector (Mid vs Side) ───────────────────────────────────────────
# Ogni embedder (MERT / PANN / CLAP) e CDPAM riducono lo stereo a un canale con
# la stessa riga: `wav.mean(0)`. Questo selettore la sostituisce SENZA toccare
# nulla del preprocessing a valle (resample, normalizzazione, quantizzazione),
# che resta quello definito dagli autori originali: cambia solo QUALE segnale
# entra. Default "mid" → comportamento bit-identico a prima.
#
#   mid  = (L+R)/2   il downmix mono storico: quello che FAD/CLAP/CDPAM vedono
#   side = (L-R)/2   stessa convenzione di scala del mid (NON /sqrt(2)), così
#                    le due modalità sono direttamente confrontabili
CHANNEL_MID  = "mid"
CHANNEL_SIDE = "side"
CHANNELS     = (CHANNEL_MID, CHANNEL_SIDE)


def ch_name(model_name: str, channel: str = CHANNEL_MID) -> str:
    """Cache/stats dir name for an embedder on a given channel.

    Side artefacts live under a DISTINCT name ("<model>-side") so a Side
    prediction can never be scored against Mid reference stats: that mistake is
    silent and inflates FAD by ~50x (see compute_fad.embed_mert_framewise).
    """
    return model_name if channel == CHANNEL_MID else f"{model_name}-{channel}"


def downmix(wav: torch.Tensor, channel: str = CHANNEL_MID) -> torch.Tensor:
    """Reduce a [C, T] waveform to the single channel an embedder consumes.

    Args:
        wav: waveform [C, T]. Mono input is returned as-is for "mid"; for
            "side" it is identically zero (a mono signal has no Side).
        channel: "mid" (default, = wav.mean(0)) or "side".

    Returns:
        1-D tensor [T]. Callers needing [1, T] add the axis themselves, exactly
        as they did around the `mean(0, keepdim=True)` they replace.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if channel == CHANNEL_MID or wav.shape[0] == 1:
        return wav.mean(0) if channel == CHANNEL_MID else torch.zeros_like(wav[0])
    return (wav[0] - wav[1]) / 2


def atomic_save_npy(path: Path, data: np.ndarray) -> None:
    """Save numpy array atomically (PID-unique tmp → os.replace).

    np.save always appends .npy if the path doesn't already end in .npy,
    so we keep .npy in the tmp name to avoid a double-extension mismatch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.npy"
    np.save(str(tmp), data)
    os.replace(tmp, path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@contextmanager
def silence_output():
    """Suppress stdout + logging (used when loading noisy models)."""
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        logger = logging.getLogger()
        old = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(old)


# ── FMA file collection ───────────────────────────────────────

_SKIP_DIRS = frozenset({"metrics", "embeddings", "cache", "convert"})


def collect_fma_files(root: Path, audio_exts: set[str], fma_csv_path, max_files: int) -> list[Path]:
    """Scan *root* recursively, filter to FMA test split if CSV given.

    Skips files inside metrics/, embeddings/, cache/ subdirectories.
    """
    audio_files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in audio_exts
        and not _SKIP_DIRS.intersection(p.relative_to(root).parts[:-1])
    )
    if fma_csv_path and Path(fma_csv_path).exists():
        import pandas as pd
        try:
            tracks = pd.read_csv(fma_csv_path, index_col=0, header=[0, 1])
            test_ids = {f"{tid:06d}" for tid in tracks[tracks[("set", "split")] == "test"].index}
            dedup: dict[str, Path] = {}
            for f in audio_files:
                if f.stem in test_ids:
                    if f.stem not in dedup or len(f.parts) < len(dedup[f.stem].parts):
                        dedup[f.stem] = f
            audio_files = sorted(dedup.values())
        except Exception as e:
            warn(f"FMA CSV parse failed: {e}")
    if max_files > 0:
        audio_files = audio_files[:max_files]
    return audio_files


def collect_moisesdb_files(root: Path, split: str, max_files: int) -> list[Path]:
    """Scan *root* (chunks_30s flat dir) for MoisesDB .wav files.

    Args:
        root:      Path to the chunks_30s directory.
        split:     'mixtures' (only *_mixture.wav) or 'stems' (all other .wav).
        max_files: Max files to return (0 = all).
    """
    if split == "mixtures":
        audio_files = sorted(p for p in root.glob("*_mixture.wav"))
    elif split == "stems":
        audio_files = sorted(p for p in root.glob("*.wav") if not p.name.endswith("_mixture.wav"))
    else:
        raise ValueError(f"Unknown moisesdb split: {split}")

    if max_files > 0:
        audio_files = audio_files[:max_files]
    return audio_files


# ── Cross-correlation alignment ───────────────────────────────

def batch_align(target: torch.Tensor, pred: torch.Tensor, sr: int, max_shift_seconds: float = 1.0):
    """GPU-accelerated batch alignment via FFT cross-correlation.

    Args:
        target: [B, C, T]
        pred:   [B, C, T]
    Returns:
        (aligned_target, aligned_pred, best_lags)
    """
    B, C, T = target.shape
    device = target.device

    t_m = target.mean(1) - target.mean(1).mean(-1, keepdim=True)
    p_m = pred.mean(1)   - pred.mean(1).mean(-1, keepdim=True)

    L = 2 * T - 1
    n_fft = 1
    while n_fft < L:
        n_fft *= 2

    corr = torch.fft.irfft(
        torch.fft.rfft(torch.flip(t_m, dims=[-1]), n=n_fft) *
        torch.fft.rfft(p_m, n=n_fft),
        n=n_fft
    )[:, :L]

    lags = torch.arange(-T + 1, T, device=device)
    max_shift = int(max_shift_seconds * sr)
    mask = (lags >= -max_shift) & (lags <= max_shift)
    best_lags = lags[mask][corr[:, mask].argmax(-1)]

    aligned_target = torch.zeros_like(target)
    aligned_pred   = torch.zeros_like(pred)
    for i in range(B):
        lag = int(best_lags[i].item())
        x, y = target[i], pred[i]
        if lag > 0:
            y, x = y[..., lag:], x[..., : y.shape[-1] - lag]
        elif lag < 0:
            a = abs(lag)
            x, y = x[..., a:], y[..., : x.shape[-1] - a]
        cur = x.shape[-1]
        aligned_target[i, :, :cur] = x
        aligned_pred[i, :, :cur]   = y

    return aligned_target, aligned_pred, best_lags


# ── Cache helpers ─────────────────────────────────────────────

def load_or_embed(ml, embed_fn, wav: torch.Tensor, src_sr: int,
                  device, cache_path: Optional[Path]) -> np.ndarray:
    """Return cached .npy embedding if present; else compute (and optionally cache) it."""
    if cache_path and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    emb = embed_fn(ml, wav, src_sr, device)
    if cache_path:
        atomic_save_npy(cache_path, emb)
    return emb.astype(np.float32)


def target_cache_path(shared_cache: Optional[Path], model_name: str, stem: str) -> Optional[Path]:
    if shared_cache is None:
        return None
    p = shared_cache / model_name / "target" / f"{stem}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Swin frame-size detection ─────────────────────────────────

def get_expected_frames(module: torch.nn.Module) -> Optional[int]:
    """Recursively find PatchEmbed img_size[1] — the Swin temporal context window."""
    if hasattr(module, "img_size") and isinstance(module.img_size, (tuple, list)) and len(module.img_size) == 2:
        return int(module.img_size[1])
    for child in module.children():
        v = get_expected_frames(child)
        if v is not None:
            return v
    return None


# ── Batched multi-file codec inference ────────────────────────

def infer_batch(codec, waveforms: list[torch.Tensor], chunk_samples: int) -> list[torch.Tensor]:
    """Encode+decode B waveforms simultaneously, chunked by chunk_samples.

    All chunks from all files are stacked into a single GPU call.

    Args:
        codec:         sage.SAGE instance (already on device).
        waveforms:     List of [C, T] tensors on GPU.
        chunk_samples: Model context window in samples.

    Returns:
        List of reconstructed [C, T] tensors (same lengths as input).
    """
    B = len(waveforms)
    all_chunks = []
    file_chunk_counts = []
    pad_lens = []

    for wav in waveforms:
        chunks = list(torch.split(wav, chunk_samples, dim=-1))
        pl = 0
        if chunks[-1].shape[-1] < chunk_samples:
            pl = chunk_samples - chunks[-1].shape[-1]
            chunks[-1] = F.pad(chunks[-1], (0, pl))
        all_chunks.extend(chunks)
        file_chunk_counts.append(len(chunks))
        pad_lens.append(pl)

    # Stack all chunks together and pass through the codec in one go
    t = torch.stack(all_chunks).to(codec.device) # [TotalChunks, C, chunk_samples]
    lat = codec.encode(t)
    out = codec.decode(lat, target_length=chunk_samples).cpu() # [TotalChunks, C, chunk_samples]

    # Reconstruct the original audio files
    results: list[torch.Tensor] = []
    offset = 0
    for b in range(B):
        count = file_chunk_counts[b]
        chunks_out = [out[i] for i in range(offset, offset + count)]
        offset += count

        pl = pad_lens[b]
        if pl > 0:
            chunks_out[-1] = chunks_out[-1][..., : chunk_samples - pl]
        
        results.append(torch.cat(chunks_out, dim=-1))

    return results


