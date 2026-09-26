# =============================================================================
# Audio STFT Dataloader and Dataset with support for custom file providers.
# Supports pluggable file providers via `custom_metadata_module` parameter.
# =============================================================================

import torch
import torchaudio
import os
import math
import signal
from pathlib import Path
from typing import Callable, Optional, Sequence
from torch.utils.data import Dataset, ConcatDataset

from sage.utils.console import warn
from sage.utils.audio import is_silence, load_waveform, random_crop_or_pad, get_audio_info
from sage.training.data.file_scanning import fast_scandir
from sage.training.data.metadata.providers import load_file_provider_fn
from sage.training.data.audio_probe import pick_probe_fn

from config import (
    DEFAULT_SEED,
    DEFAULT_MAX_RETRIES_PER_SAMPLE,
    DEFAULT_MAX_PAD_RATIO,
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_AUDIO_LOAD_TIMEOUT,
    DEFAULT_PARTIAL_READ,
    DEFAULT_PARTIAL_READ_MARGIN,
)


class _AudioLoadTimeout(Exception):
    pass


class _file_load_timeout:
    """Context manager: raises _AudioLoadTimeout if the block takes longer than `seconds`.
    Uses SIGALRM — Unix only, safe in forked DataLoader worker processes.
    No-op on Windows (os.name == 'nt').
    """
    def __init__(self, seconds: int):
        self._seconds = seconds
        self._old_handler = None

    def __enter__(self):
        if os.name == "nt" or self._seconds <= 0:
            return self
        self._old_handler = signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self._seconds)
        return self

    def __exit__(self, *_):
        if os.name == "nt" or self._seconds <= 0:
            return False
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old_handler or signal.SIG_DFL)
        return False

    def _handle(self, signum, frame):
        raise _AudioLoadTimeout(f"Audio load timed out after {self._seconds}s")

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
        partial_read: bool = DEFAULT_PARTIAL_READ,
        partial_read_margin: int = DEFAULT_PARTIAL_READ_MARGIN,
        max_files: Optional[int] = None,
        filelist_cache: Optional[str | os.PathLike] = None,
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.extensions = list(extensions or DEFAULT_AUDIO_EXTENSIONS)
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
        self.partial_read = bool(partial_read)               # decode only a windowed slice of long files
        self.partial_read_margin = int(partial_read_margin)  # source-domain slack around the window
        # Optional deterministic cap on the file list, applied BEFORE probe-filtering.
        # Default None = no cap (production). Used by smoke/CI runs to bound build time.
        self.max_files = int(max_files) if max_files is not None else None
        # Optional .txt cache of the provider-filtered, sorted file list (multi-corpus only).
        # Default None = always scan+probe (single-corpus path unaffected). When set, the
        # first build writes the list here and later builds reload it, skipping the ~11-min probe.
        self.filelist_cache = Path(filelist_cache).expanduser() if filelist_cache else None
        
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
        # Shared epoch counter in shared memory: persistent DataLoader workers are forked
        # once, so a plain attribute mutated in the main process would never reach them.
        # A shared-memory tensor is inherited by the fork, so set_epoch() updates ARE visible
        # inside workers. Crop randomness is derived per-item from (base_seed, epoch, index)
        # → deterministic, decorrelated across files, and genuinely different every epoch.
        self._epoch_t = torch.zeros(1, dtype=torch.long).share_memory_()

        self._warned_channel_mismatch = False
        self._warned_sr_mismatch = False
        self._warned_failed_samples = False

    def _scan_and_filter_files(self) -> list[Path]:
        # Cache fast-path: a persisted .txt of the already-filtered, sorted list lets us
        # skip the ~11-min provider-scan + per-file probe. Disabled when max_files is set
        # (a capped smoke list must never be persisted as the full corpus). Falls back to a
        # full scan if the cache is missing or unreadable/empty.
        cache_active = self.filelist_cache is not None and self.max_files is None
        if cache_active and self.filelist_cache.exists():
            cached = self._load_filelist_cache()
            if cached:
                warn(f"Loaded {len(cached)} files from cache {self.filelist_cache} (probe skipped)", prefix="DATA")
                return cached
            warn(f"Filelist cache {self.filelist_cache} empty/unreadable — rebuilding", prefix="DATA WARNING")

        if self._file_provider is not None:
            file_paths = self._file_provider(
                str(self.audio_dir),
                **self._file_provider_kwargs
            )
            files = [Path(p) for p in sorted(file_paths)]
        else:
            _, file_paths = fast_scandir(str(self.audio_dir), self.extensions)
            files = [Path(p) for p in sorted(file_paths)]

        # Deterministic truncation (sorted order) before the expensive probe loop.
        if self.max_files is not None:
            files = files[: self.max_files]

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

        if cache_active:
            self._write_filelist_cache(filtered)

        return filtered

    def _load_filelist_cache(self) -> list[Path]:
        """Read the cached filelist (one absolute path per line; '#' header lines ignored).
        Returns [] on any read error so the caller can fall back to a full scan."""
        try:
            lines = self.filelist_cache.read_text().splitlines()
        except OSError as e:
            warn(f"Could not read filelist cache {self.filelist_cache}: {e}", prefix="DATA WARNING")
            return []
        return [Path(ln) for ln in lines if ln and not ln.startswith("#")]

    def _write_filelist_cache(self, files: list[Path]) -> None:
        """Persist the filtered, sorted filelist atomically (tmp + os.replace, so concurrent
        DDP ranks never observe a half-written file). Best-effort: a write failure only loses
        the speedup, never the run."""
        try:
            self.filelist_cache.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"# filelist cache — {len(files)} files\n"
                f"# audio_dir={self.audio_dir}\n"
                f"# sample_rate={self.sample_rate} min_acceptable_len={self.min_acceptable_len}\n"
                f"# provider_kwargs={self._file_provider_kwargs}\n"
                f"# DELETE this file to force a rebuild after changing the corpus.\n"
            )
            body = "\n".join(str(p) for p in files)
            tmp = self.filelist_cache.with_suffix(self.filelist_cache.suffix + f".tmp.{os.getpid()}")
            tmp.write_text(header + body + "\n")
            os.replace(tmp, self.filelist_cache)
            warn(f"Wrote filelist cache {self.filelist_cache} ({len(files)} files)", prefix="DATA")
        except OSError as e:
            warn(f"Could not write filelist cache {self.filelist_cache}: {e}", prefix="DATA WARNING")

    @staticmethod
    def _mix_seed(base: int, epoch: int, index: int) -> int:
        """SplitMix64-style hash → a per-item crop seed that is a pure function of
        (base_seed, epoch, index): deterministic, decorrelated across indices, and
        different every epoch. Independent of worker id and batch order."""
        mask = 0xFFFFFFFFFFFFFFFF
        h = (int(base) + 0x9E3779B97F4A7C15) & mask
        h = (h ^ ((int(epoch) + 1) * 0xBF58476D1CE4E5B9)) & mask
        h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & mask
        h = (h ^ (int(index) * 0x94D049BB133111EB)) & mask
        h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & mask
        h = (h ^ (h >> 31)) & mask
        return h

    @property
    def epoch(self) -> int:
        """Current epoch as seen by this process (read from the shared counter)."""
        return int(self._epoch_t[0])

    def set_epoch(self, epoch: int):
        self._epoch_t[0] = int(epoch)
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

    def _note_mismatch(self, ch_mismatch: bool, sr_mismatch: bool, epoch: int):
        """Record (once, at epoch 0) that some files needed channel/SR conversion."""
        if ch_mismatch and not self._warned_channel_mismatch and epoch == 0:
            self._warned_channel_mismatch = True
        if sr_mismatch and not self._warned_sr_mismatch and epoch == 0:
            self._warned_sr_mismatch = True

    def _try_partial_segment(self, path: Path, crop_rng: torch.Generator) -> Optional[torch.Tensor]:
        """Decode only a ~segment-long window of a (possibly multi-minute) file, then crop it.
        Returns None to signal "fall back to full load" on short files or any seek/decode failure."""
        try:
            num_frames, src_sr, _ = get_audio_info(path)
        except Exception:
            return None
        if num_frames <= 0 or src_sr <= 0:
            return None
        # Source-domain window that, after resampling to target SR, still covers one segment + margin.
        win_src = math.ceil(self.segment_samples * src_sr / self.sample_rate) + self.partial_read_margin
        if num_frames < win_src:
            return None  # too short for a windowed read → let full load (with padding) handle it
        max_off = num_frames - win_src
        offset = int(torch.randint(0, max_off + 1, (1,), generator=crop_rng).item())
        try:
            with _file_load_timeout(DEFAULT_AUDIO_LOAD_TIMEOUT):
                wav, _, ch_m, sr_m = load_waveform(
                    path,
                    target_sample_rate=self.sample_rate,
                    expected_channels=self.audio_channels,
                    frame_offset=offset,
                    num_frames=win_src,
                )
        except Exception:
            return None
        if wav.shape[-1] < self.segment_samples:
            return None  # resample edges shrank the window below target → fall back
        self._note_mismatch(ch_m, sr_m, self.epoch)
        return random_crop_or_pad(wav, self.segment_samples, self.min_acceptable_len, crop_rng)

    def _load_segment(self, path: Path, crop_rng: torch.Generator, epoch: int) -> torch.Tensor:
        """Return a (channels, segment_samples) crop. Windowed partial read when enabled,
        else (or on fallback) a full-file decode followed by a random crop/pad."""
        if self.partial_read:
            seg = self._try_partial_segment(path, crop_rng)
            if seg is not None:
                return seg
        with _file_load_timeout(DEFAULT_AUDIO_LOAD_TIMEOUT):
            wav, _, ch_m, sr_m = load_waveform(
                path,
                target_sample_rate=self.sample_rate,
                expected_channels=self.audio_channels,
            )
        self._note_mismatch(ch_m, sr_m, epoch)
        return random_crop_or_pad(wav, self.segment_samples, self.min_acceptable_len, crop_rng)

    def __getitem__(self, index: int):
        n = len(self)
        epoch = self.epoch
        last_error: Exception | None = None
        last_path: Path | None = None

        for attempt in range(self.max_retries_per_sample):
            idx = (index + attempt) % n
            path = self.files[idx]
            last_path = path

            # Per-item crop generator seeded by (base_seed, epoch, file index). Pure function
            # of those three → reproducible, varies per epoch, independent of worker/batch order.
            crop_rng = torch.Generator()
            crop_rng.manual_seed(self._mix_seed(self._base_seed, epoch, idx))

            if self.full_waveform:
                try:
                    with _file_load_timeout(DEFAULT_AUDIO_LOAD_TIMEOUT):
                        wav, _, ch_m, sr_m = load_waveform(
                            path,
                            target_sample_rate=self.sample_rate,
                            expected_channels=self.audio_channels,
                        )
                except Exception as e:
                    last_error = e
                    continue
                self._note_mismatch(ch_m, sr_m, epoch)
                wav = wav.contiguous()
                if self.return_paths:
                    return None, wav, str(path)
                return None, wav

            try:
                seg = self._load_segment(path, crop_rng, epoch)
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


class MultiCorpusDataset(ConcatDataset):
    """Concatenation of several ``OnTheFlySTFTDataset`` corpora (FMA-full + Jamendo +
    M4Singer) behind a single flat global index space.

    Reuses ``ConcatDataset`` for ``__len__``/``__getitem__`` routing (a global index
    is dispatched to the owning child via the cumulative sizes) and adds two things
    the multi-corpus path needs:

    * ``set_epoch`` is forwarded to every child so the per-item crop seeding keeps
      varying the 1.5 s window each epoch inside each corpus (validation, which never
      calls ``set_epoch``, stays at epoch 0 → fixed crops, unchanged for FMA-only val).
    * ``corpus_sizes`` exposes the post-filter length of each child, which the
      ``MultiCorpusRotatingSampler`` consumes to build its rotation (never hardcoded).
    """

    def __init__(self, datasets: Sequence[Dataset]):
        super().__init__(datasets)  # builds self.datasets + self.cumulative_sizes

    @property
    def corpus_sizes(self) -> list[int]:
        """Per-corpus item counts, in the order the corpora were passed."""
        return [len(d) for d in self.datasets]

    def set_epoch(self, epoch: int) -> None:
        """Propagate the epoch to every child that supports per-epoch crop variation."""
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)
