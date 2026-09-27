# =============================================================================
# Shared helpers of the evaluation scripts: the mono downmix every embedder
# uses, atomic .npy / CSV writes, the evaluation-set file lists and the cache
# of target embeddings shared by all models (so their FADs are comparable).
# =============================================================================
from __future__ import annotations

import csv
import logging
import os
import random
import zlib
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from sage.utils.console import warn


# ── Channel fed to the embedders ─────────────────────────────────────────────
# Every embedder (MERT / PANN / CLAP) and CDPAM reduce stereo to one channel;
# the paper uses the mono downmix (L+R)/2. The metric functions keep a `channel`
# argument; "side" = (L-R)/2 is not exposed by the evaluator (the paper has no
# Side-channel FAD) but left for analysis.
CHANNEL_MID  = "mid"
CHANNEL_SIDE = "side"
CHANNELS     = (CHANNEL_MID, CHANNEL_SIDE)


def downmix(wav: torch.Tensor, channel: str = CHANNEL_MID) -> torch.Tensor:
    """Reduce a [C, T] waveform to the single channel an embedder consumes.

    Args:
        wav: waveform [C, T]. Mono input is returned as-is for "mid"; for
            "side" it is identically zero (a mono signal has no Side).
        channel: "mid" (default, = wav.mean(0)) or "side".

    Returns:
        1-D tensor [T]. Callers needing [1, T] add the axis themselves.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if channel == CHANNEL_MID or wav.shape[0] == 1:
        return wav.mean(0) if channel == CHANNEL_MID else torch.zeros_like(wav[0])
    return (wav[0] - wav[1]) / 2


# Embedder names: prediction/target embedding folders and reference-statistics folders.
MERT_NAME       = "MERT-v1-95M-4"          # == fadtk MERTModel(layer=4).name
CLAP_AUDIO_NAME = "clap-laion-audio"
CLAP_MUSIC_NAME = "clap-laion-music"
CLAP_GUD_NAME   = "clap-laion-audio-gud"   # whole-file CLAP → FAD-CLAP


def file_seed(seed: int, stem: str) -> int:
    """Seed of one file, from the run seed and the file name (independent of run layout)."""
    return seed + zlib.crc32(stem.encode())


def seed_everything(s: int) -> None:
    """Seed torch (CPU and every CUDA device), numpy and random."""
    torch.manual_seed(s)
    np.random.seed(s % 2**32)
    random.seed(s)


# ── I/O ──────────────────────────────────────────────────────────────────────

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


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append one row, writing the header if the file is new or empty."""
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


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


# ── Evaluation sets ──────────────────────────────────────────────────────────

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


def collect_clip_files(data_dir: Path, max_files: int) -> list[Path]:
    """The .wav clips at the top level of a clip set (MoisesDB, MusicCaps, Song Describer)."""
    files = sorted(data_dir.glob("*.wav"))
    return files[:max_files] if max_files > 0 else files


def first_item(batch: list):
    """collate_fn of the batch-size-1 loaders. A module-level function, not a lambda, so the
    loader also works with worker processes under the spawn start method (macOS)."""
    return batch[0]


class AudioDataset(Dataset):
    """Load, resample and fix the channel count; optionally cache the result as .npy."""

    def __init__(self, files: list[Path], target_sr: int, target_channels: int = 2,
                 cache_dir: Optional[Path] = None):
        self.files = files
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            cache_file = (self.cache_dir / f"wav_{p.stem}_ch{self.target_channels}_sr{self.target_sr}.npy"
                          if self.cache_dir is not None else None)
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


# ── Target-embedding cache (FMA protocol) ────────────────────────────────────

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
