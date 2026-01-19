"""
Audio STFT Dataset with custom metadata support.

Supports pluggable metadata providers via `custom_metadata_module` parameter.
Example metadata module:

    # my_metadata.py
    def get_custom_metadata(info, audio):
        return {"prompt": info["relpath"]}

Then in your config:
    custom_metadata_module: "my_metadata"
"""

import torch
import torchaudio
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union, List
import torch.nn.functional as F
from mutagen.mp3 import MP3
from torch.utils.data import Dataset
import numpy as np
from rich.console import Console

from .metadata_providers import load_metadata_fn

console = Console()

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")


# --- File scanning utilities ---

AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus")


def fast_scandir(dir: str, ext: list) -> tuple[list[str], list[str]]:
    """
    Very fast glob alternative for recursively scanning directories.
    From https://stackoverflow.com/a/59803793/4259243
    """
    subfolders, files = [], []
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    
    try:
        for f in os.scandir(dir):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")
                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for subdir in list(subfolders):
        sf, f = fast_scandir(subdir, ext)
        subfolders.extend(sf)
        files.extend(f)
    
    return subfolders, files


def get_audio_filenames(
    paths: Union[str, List[str]],
    exts: list = None,
) -> list[str]:
    """Recursively get a list of audio filenames from directories."""
    if exts is None:
        exts = list(AUDIO_EXTENSIONS)
    
    filenames = []
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        # Check for filelist.txt at the root of the directory
        filelist_path = os.path.join(path, "filelist.txt")
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = [os.path.join(path, line.strip()) for line in f if line.strip()]
                filenames.extend(files)
            continue
        
        _, files = fast_scandir(path, exts)
        filenames.extend(files)
    
    return filenames


# --- Audio utilities ---

def get_dbmax(audio: torch.Tensor) -> float:
    """Finds the loudest value in the entire clip and puts that into dB (full scale)."""
    return 20 * torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()


def is_silence(audio: torch.Tensor, thresh: float = -62) -> bool:
    """Checks if entire clip is 'silence' below some dB threshold."""
    return get_dbmax(audio) < thresh


# --- Dataset ---

class OnTheFlySTFTDataset(Dataset):
    """
    Load audio files recursively, resample to target sample rate, optionally keep stereo,
    crop/pad a random fixed-length segment, and compute a complex STFT.

    Output shape per item (batch dim added by DataLoader):
      - If cac=True (complex-as-channels): (2*C, F, T), where C is 2 for stereo, 1 for mono.
      - If cac=False (complex tensor): (C, F, T) with complex dtype.

    Args:
        audio_dir: Directory containing audio files (scanned recursively).
        sample_rate: Target sample rate.
        n_fft, hop_length, win_length: STFT parameters.
        target_frames: Number of STFT time frames to output.
        stereo: If True, output stereo (2 channels); if False, mono.
        cac: If True, output complex-as-channels format.
        custom_metadata_module: Dotted path to module with get_custom_metadata(info, audio).
        skip_mismatched_sr: Skip files with different sample rate (no resampling).
        skip_mismatched_channels: Skip files with different channel count (no conversion).
        extensions: Allowed audio file extensions.
        max_pad_ratio: Maximum ratio of padding allowed for short files.
        full_waveform: If True, return full waveform without cropping/STFT.
        return_paths: If True, also return file paths.
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
        max_pad_ratio: float = 0.05,
        extensions: Optional[Sequence[str]] = None,
        stereo: bool = True,
        cac: bool = False,
        seed: int = 42,
        dtype: torch.dtype = torch.complex64,
        skip_mismatched_sr: bool = False,
        skip_mismatched_channels: bool = False,
        length: Optional[int] = None,
        target_frames: Optional[int] = None,
        full_waveform: bool = False,
        return_paths: bool = False,
        custom_metadata_module: Optional[str] = None,
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
        
        # Skip criteria
        self.skip_mismatched_sr = bool(skip_mismatched_sr)
        self.skip_mismatched_channels = bool(skip_mismatched_channels)

        # Channel info
        self.audio_channels = 2 if self.stereo else 1
        self.spec_channels = 2 * self.audio_channels if self.cac else self.audio_channels
        
        # Metadata provider
        self._get_metadata = load_metadata_fn(custom_metadata_module)
        
        # Segment length calculation
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

        # Probe function for quick file info
        self._probe_fn = self._pick_probe_fn()
        
        # Scan and filter files
        self.files = self._scan_and_filter_files()
        
        if not self.files:
            raise RuntimeError(
                f"No usable files found in {self.audio_dir} with min length {self.min_acceptable_len} samples."
            )

        # STFT transform
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

        # RNG for reproducible sampling
        self._base_seed = int(seed)
        self._epoch = 0
        self._rng = torch.Generator()
        self._reset_rng()
        
        # Warning flags (show once)
        self._warned_channel_mismatch = False
        self._warned_sr_mismatch = False

    def _scan_and_filter_files(self) -> list[Path]:
        """Scan audio directory and filter files based on criteria."""
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
                
                # Check sample rate
                if self.skip_mismatched_sr and src_sr != self.sample_rate:
                    skipped_sr += 1
                    continue
                
                # Check channels
                if self.skip_mismatched_channels and num_channels != self.audio_channels:
                    skipped_ch += 1
                    continue
                
                # Check length
                if not self.full_waveform:
                    est_len = int(round(num_frames * (self.sample_rate / src_sr))) if src_sr > 0 else num_frames
                    if est_len < self.min_acceptable_len:
                        skipped_len += 1
                        continue
                
                filtered.append(p)
            except Exception:
                continue
        
        # Log skip summary
        if skipped_sr > 0:
            warn(f"Skipped {skipped_sr} file(s) due to sample rate mismatch (expected {self.sample_rate} Hz)")
        if skipped_ch > 0:
            warn(f"Skipped {skipped_ch} file(s) due to channel mismatch (expected {self.audio_channels})")
        if skipped_len > 0:
            warn(f"Skipped {skipped_len} file(s) due to insufficient length")
        
        return filtered

    def _reset_rng(self):
        mixed = (self._base_seed & 0xFFFFFFFF) ^ ((self._epoch * 0x9E3779B1) & 0xFFFFFFFF)
        self._rng.manual_seed(mixed)

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to vary sampling reproducibly."""
        self._epoch = int(epoch)
        self._reset_rng()

    def enable_return_paths(self):
        """Enable returning source file paths alongside samples."""
        self.return_paths = True

    def __len__(self) -> int:
        return len(self.files)

    def _load_waveform(self, path: Path) -> tuple[torch.Tensor, int]:
        """Load and preprocess audio waveform."""
        wav, sr = torchaudio.load(str(path), normalize=True)
        wav = wav.to(torch.float32)
        
        # Adapt channels
        file_channels = wav.size(0)
        expected = self.audio_channels
        
        if file_channels != expected:
            if not self._warned_channel_mismatch and self._epoch == 0:
                want = "stereo" if self.stereo else "mono"
                warn(f"Some files have different channels than expected ({want}). Converting automatically.")
                self._warned_channel_mismatch = True
            
            if expected == 1 and file_channels > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif expected == 2 and file_channels == 1:
                wav = wav.repeat(2, 1)
        
        # Resample if needed
        if sr != self.sample_rate:
            if not self._warned_sr_mismatch and self._epoch == 0:
                warn(f"Some files have different sample rate. Resampling to {self.sample_rate} Hz.")
                self._warned_sr_mismatch = True
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate
        
        return wav, sr

    def _random_crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        """Random crop or pad waveform to segment_samples length."""
        cur_len = wav.shape[-1]
        
        if cur_len >= self.segment_samples:
            max_start = cur_len - self.segment_samples
            start = int(torch.randint(0, max_start + 1, (1,), generator=self._rng).item())
            return wav[..., start:start + self.segment_samples]
        elif cur_len >= self.min_acceptable_len:
            pad_needed = self.segment_samples - cur_len
            left = pad_needed // 2
            right = pad_needed - left
            return F.pad(wav, (left, right), mode="constant", value=0.0)
        else:
            raise RuntimeError("File too short")

    @staticmethod
    def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
        if complex_dtype == torch.complex64:
            return torch.float32
        if complex_dtype == torch.complex128:
            return torch.float64
        return torch.float32

    def _get_file_info(self, path: Path) -> dict:
        """Build info dict for metadata provider."""
        return {
            "path": str(path),
            "relpath": str(path.relative_to(self.audio_dir)),
        }

    def __getitem__(self, index: int):
        path = self.files[index]
        
        # Load waveform - skip on failure
        try:
            wav, _ = self._load_waveform(path)
        except Exception as e:
            warn(f"Failed to load {path}: {e}")
            new_idx = (index + 1) % len(self)
            if new_idx == index:
                raise RuntimeError(f"Cannot load any file. Last error: {e}")
            return self[new_idx]
        
        # Full waveform mode
        if self.full_waveform:
            wav = wav.contiguous()
            result = (None, wav)
            if self._get_metadata:
                info = self._get_file_info(path)
                result = (*result, self._get_metadata(info, wav))
            if self.return_paths:
                result = (*result, str(path))
            return result
        
        # Crop/pad segment - skip on failure
        try:
            seg = self._random_crop_or_pad(wav)
        except Exception:
            new_idx = (index + 1) % len(self)
            if new_idx == index:
                raise RuntimeError("Cannot process any file")
            return self[new_idx]
        
        # Skip silence
        if is_silence(seg) and len(self) > 1:
            new_idx = (index + 1) % len(self)
            if new_idx != index:
                return self[new_idx]
        
        # Compute STFT
        S = self._stft(seg)
        
        # Ensure complex dtype
        if not torch.is_complex(S):
            if S.dim() >= 4 and S.size(-1) == 2:
                S = torch.view_as_complex(S.contiguous())
            else:
                S = S.to(torch.complex64)
        S = S.to(self.dtype)
        
        # Format output
        if self.cac:
            S_ri = torch.stack((S.real, S.imag), dim=1)
            S = S_ri.flatten(0, 1).contiguous()
            S = S.to(self._real_dtype_for(self.dtype))
        else:
            S = S.contiguous()
        
        seg = seg.contiguous()
        result = (S, seg)
        
        if self._get_metadata:
            info = self._get_file_info(path)
            result = (*result, self._get_metadata(info, seg))
        
        if self.return_paths:
            result = (*result, str(path))
        
        return result

    def _pick_probe_fn(self):
        """Pick the best available probe function for file metadata."""
        def _probe_mp3(path: Path):
            meta = MP3(str(path))
            sr = int(meta.info.sample_rate)
            nf = int(meta.info.length * sr)
            nc = int(meta.info.channels)
            return sr, nf, nc

        if hasattr(torchaudio, "info"):
            def _probe(path):
                if path.suffix.lower() == ".mp3":
                    return _probe_mp3(path)
                i = torchaudio.info(str(path))
                sr = getattr(i, "sample_rate", None)
                nf = getattr(i, "num_frames", None)
                nc = getattr(i, "num_channels", getattr(i, "channels", None))
                if sr is None or nf is None or nc is None:
                    raise RuntimeError("Incomplete AudioMetaData")
                return int(sr), int(nf), int(nc)
            return _probe

        try:
            import soundfile as sf
            def _probe(path):
                with sf.SoundFile(str(path)) as f:
                    return int(f.samplerate), int(len(f)), int(f.channels)
            return _probe
        except Exception:
            pass

        return None
