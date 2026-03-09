# =============================================================================
# Audio STFT Dataloader and Dataset with support for custom file providers.
# Supports pluggable file providers via `custom_metadata_module` parameter.
# =============================================================================

import torch
import torchaudio
import os
from pathlib import Path
from typing import Callable, Optional, Sequence
from torch.utils.data import Dataset
import pytorch_lightning as pl

# ─────────────────────────────────────────────────────────────────────────
# Imports from phase 2 refactoring
# ─────────────────────────────────────────────────────────────────────────
from ar_spectra.utils.console import warn
from ar_spectra.utils.audio import is_silence, load_waveform, random_crop_or_pad
from ar_spectra.utils.file_scanning import fast_scandir
from ar_spectra.utils.metadata.providers import load_file_provider_fn
from ar_spectra.utils.audio_probe import pick_probe_fn

from config import (
    DEFAULT_SEED,
    DEFAULT_MAX_RETRIES_PER_SAMPLE,
    DEFAULT_MAX_PAD_RATIO
)

# ─────────────────────────────────────────────────────────────────────────
# Datasets and Epoch Setters
# ─────────────────────────────────────────────────────────────────────────

class DatasetEpochSetter(pl.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(trainer.current_epoch)

class OnTheFlySTFTDataset(Dataset):
    """
    Load audio files recursively, resample to target sample rate, optionally keep stereo,
    crop/pad a random fixed-length segment, and compute a complex STFT.
    """

    def __init__(
        self, 
        *,
        audio_dir: str | os.PathLike,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable = torch.hann_window,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        max_pad_ratio: float = DEFAULT_MAX_PAD_RATIO,
        extensions: Optional[Sequence[str]] = None,
        stereo: bool = True,
        cac: bool = False,
        seed: int = DEFAULT_SEED,
        dtype: torch.dtype = torch.complex64,
        skip_mismatched_sr: bool = False,
        skip_mismatched_channels: bool = False,
        length: Optional[int] = None,
        target_frames: Optional[int] = None,
        full_waveform: bool = False,
        return_paths: bool = False,
        custom_metadata_module: Optional[str] = None,
        custom_metadata_kwargs: Optional[dict] = None,
        max_retries_per_sample: int = DEFAULT_MAX_RETRIES_PER_SAMPLE,
        skip_failed_samples: bool = False,
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.extensions = list(extensions or AUDIO_EXTENSIONS)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.pad_mode = str(pad_mode)
        self.normalized = bool(normalized)
        self.dtype = dtype
        self.max_pad_ratio = float(max_pad_ratio)
        self.stereo = bool(stereo)
        self.cac = bool(cac)
        self.full_waveform = bool(full_waveform)
        self.return_paths = bool(return_paths)
        self.max_retries_per_sample = int(max_retries_per_sample)
        self.skip_failed_samples = bool(skip_failed_samples)
        
        self._file_provider = load_file_provider_fn(custom_metadata_module)
        self._file_provider_kwargs = custom_metadata_kwargs or {}
        
        self.skip_mismatched_sr = bool(skip_mismatched_sr)
        self.skip_mismatched_channels = bool(skip_mismatched_channels)

        self.audio_channels = 2 if self.stereo else 1
        self.spec_channels = 2 * self.audio_channels if self.cac else self.audio_channels
        
        if self.full_waveform:
            self.target_frames = None
            self.segment_samples = None
            self.min_acceptable_len = 0
        else:
            frames = target_frames if target_frames is not None else length
            if frames is None:
                raise ValueError("OnTheFlySTFTDataset requires 'target_frames'. Provide target_frames in dataset kwargs.")
            if frames < 2:
                raise ValueError("target_frames must be >= 2 to compute STFT segments.")
            self.target_frames = int(frames)
            self.segment_samples = (self.target_frames - 1) * self.hop_length
            self.min_acceptable_len = int(self.segment_samples * (1.0 - self.max_pad_ratio))

        self._probe_fn = pick_probe_fn()
        
        self.files = self._scan_and_filter_files()
        self._probe_fn = None
        
        if not self.files:
            raise RuntimeError(
                f"No usable files found in {self.audio_dir} with min length {self.min_acceptable_len} samples."
            )

        self._stft = None
        if not self.full_waveform:
            self._stft = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn=window_fn,
                power=None,
                center=self.center,
                pad_mode=self.pad_mode,
                normalized=self.normalized,
            )

        self._base_seed = int(seed)
        self._epoch = 0
        self._rng = torch.Generator()
        self._reset_rng()
        
        self._warned_channel_mismatch = False
        self._warned_sr_mismatch = False
        self._warned_failed_samples = False

    def _scan_and_filter_files(self) -> list[Path]:
        if self._file_provider is not None:
            file_paths = self._file_provider(
                str(self.audio_dir), 
                **self._file_provider_kwargs
            )
            files = [Path(p) for p in sorted(file_paths)]
        else:
            _, file_paths = fast_scandir(str(self.audio_dir), self.extensions)
            files = [Path(p) for p in sorted(file_paths)]
        
        if self._probe_fn is None:
            return files
        
        filtered = []
        skipped_sr = 0
        skipped_ch = 0
        skipped_len = 0
        
        for p in files:
            try:
                src_sr, num_frames, num_channels = self._probe_fn(p)
                
                if self.skip_mismatched_sr and src_sr != self.sample_rate:
                    skipped_sr += 1
                    continue
                
                if self.skip_mismatched_channels and num_channels != self.audio_channels:
                    skipped_ch += 1
                    continue
                
                if not self.full_waveform:
                    est_len = int(round(num_frames * (self.sample_rate / src_sr))) if src_sr > 0 else num_frames
                    if est_len < self.min_acceptable_len:
                        skipped_len += 1
                        continue
                
                filtered.append(p)
            except Exception:
                continue
        
        if skipped_sr > 0:
            warn(f"Skipped {skipped_sr} file(s) due to sample rate mismatch (expected {self.sample_rate} Hz)", prefix="DATA WARNING")
        if skipped_ch > 0:
            warn(f"Skipped {skipped_ch} file(s) due to channel mismatch (expected {self.audio_channels})", prefix="DATA WARNING")
        if skipped_len > 0:
            warn(f"Skipped {skipped_len} file(s) due to insufficient length", prefix="DATA WARNING")
        
        return filtered

    def _reset_rng(self):
        mixed = (self._base_seed & 0xFFFFFFFF) ^ ((self._epoch * 0x9E3779B1) & 0xFFFFFFFF)
        self._rng.manual_seed(mixed)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._reset_rng()
        self._warned_failed_samples = False

    def enable_return_paths(self):
        self.return_paths = True

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
        if complex_dtype == torch.complex64:
            return torch.float32
        if complex_dtype == torch.complex128:
            return torch.float64
        return torch.float32

    def __getitem__(self, index: int):
        n = len(self)
        last_error: Exception | None = None
        last_path: Path | None = None

        for attempt in range(self.max_retries_per_sample):
            idx = (index + attempt) % n
            path = self.files[idx]
            last_path = path

            try:
                wav, sr, ch_mismatch, sr_mismatch = load_waveform(
                    path, 
                    target_sample_rate=self.sample_rate, 
                    expected_channels=self.audio_channels
                )
                if ch_mismatch and not self._warned_channel_mismatch and self._epoch == 0:
                    self._warned_channel_mismatch = True
                if sr_mismatch and not self._warned_sr_mismatch and self._epoch == 0:
                    self._warned_sr_mismatch = True
            except Exception as e:
                last_error = e
                continue

            if self.full_waveform:
                wav = wav.contiguous()
                if self.return_paths:
                    return None, wav, str(path)
                return None, wav

            try:
                seg = random_crop_or_pad(wav, self.segment_samples, self.min_acceptable_len, self._rng)
            except Exception as e:
                last_error = e
                continue

            if is_silence(seg) and n > 1:
                last_error = RuntimeError("Silence segment")
                continue

            try:
                S = self._stft(seg)
            except Exception as e:
                last_error = e
                continue

            if not torch.is_complex(S):
                if S.dim() >= 4 and S.size(-1) == 2:
                    S = torch.view_as_complex(S.contiguous())
                else:
                    S = S.to(torch.complex64)
            S = S.to(self.dtype)

            if self.cac:
                S = torch.cat([S.real, S.imag], dim=0).contiguous()
                S = S.to(self._real_dtype_for(self.dtype))
            else:
                S = S.contiguous()

            seg = seg.contiguous()

            if self.return_paths:
                return S, seg, str(path)
            return S, seg

        if self.skip_failed_samples:
            if not self._warned_failed_samples:
                warn(
                    "Skipping failed sample after "
                    f"{self.max_retries_per_sample} attempts. Last error: {last_error} "
                    f"(path: {last_path})",
                    prefix="DATA"
                )
                self._warned_failed_samples = True
            return None

        raise RuntimeError(
            f"Failed to fetch item after {self.max_retries_per_sample} attempts. "
            f"Last error: {last_error} (path: {last_path})"
        )
