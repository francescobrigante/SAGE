import torch, torchaudio
import os
from pathlib import Path
from typing import Callable, Optional, Sequence
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rich.console import Console
console = Console()  

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")

class OnTheFlySTFTDataset(Dataset):
    """
    Load audio files recursively, resample to target sample rate, optionally keep stereo,
    crop/pad a random fixed-length segment, and compute a complex STFT.

    Output shape per item (batch dim added by DataLoader):
      - If cac=True (complex-as-channels): (2*C, F, T), where C is 2 for stereo, 1 for mono.
      - If cac=False (complex tensor): (C, F, T) with complex dtype.

    Notes:
      - segment_samples = (target_frames - 1) * hop_length
      - Files estimated shorter than (1 - max_pad_ratio) * segment_samples after resampling are filtered out.
    """

    def __init__(
        self, 
        audio_dir: str | os.PathLike,
        *,
        target_frames: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable[[int, torch.dtype, torch.device, bool], torch.Tensor] = torch.hann_window,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        max_pad_ratio: float = 0.05,
        extensions: Optional[Sequence[str]] = None,
        stereo: bool = True,
        cac: bool = False,  # complex-as-channels
        seed: int = 42,
        dtype: torch.dtype = torch.complex64,
        skip_broken_files: bool = True,
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.extensions = tuple((ext.lower() for ext in (extensions or [".wav", ".flac", ".mp3", ".ogg", ".m4a"])))

        self.target_frames = int(target_frames)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.pad_mode = str(pad_mode)
        self.normalized = bool(normalized)
        self.dtype = dtype  # complex dtype used when cac=False
        self.max_pad_ratio = float(max_pad_ratio)
        self.stereo = bool(stereo)
        self.cac = bool(cac)
        self.skip_broken_files = bool(skip_broken_files)

        # Channels info
        self.audio_channels: int = 2 if self.stereo else 1
        self.spec_channels: int = (self.audio_channels * 2) if self.cac else self.audio_channels

        self.segment_samples = (self.target_frames - 1) * self.hop_length
        self.min_acceptable_len = int(self.segment_samples * (1.0 - self.max_pad_ratio))

        candidate_files = sorted(
            [p for p in self.audio_dir.rglob("*") if p.suffix.lower() in self.extensions]
        )

        # Prefer torchaudio.info; fallback a torchaudio.io.info; altrimenti nessun pre-filtraggio.
        self._probe_fn = self._pick_probe_fn()

        self.files: list[Path] = []
        mismatched_count = 0
        if self._probe_fn is None:
            # Nessun modo rapido per stimare la lunghezza: tieni tutti i file, filtra a runtime.
            self.files = candidate_files
        else:
            for p in candidate_files:
                try:
                    src_sr, num_frames, num_channels = self._probe_fn(p)
                    # Se vogliamo skippare file con SR o canali diversi, filtra qui:
                    if self.skip_broken_files and (src_sr != self.sample_rate or num_channels != self.audio_channels):
                        mismatched_count += 1
                        continue
                    # altrimenti includi il file se la sua lunghezza stimata è accettabile
                    est_len = int(round(num_frames * (self.sample_rate / src_sr))) if src_sr > 0 else num_frames
                    if est_len >= self.min_acceptable_len:
                        self.files.append(p)
                except Exception:
                    # Non scartare l’intero dataset per errori puntuali, continua
                    continue
        # Emit a single warning if abbiamo escluso file per mismatch (solo se ne abbiamo esclusi)
        if mismatched_count > 0:
            warn(f"Excluded {mismatched_count} file(s) because sample rate or channel count did not match the dataset config.")

        if not self.files:
            raise RuntimeError(
                f"No usable files found in {self.audio_dir} with min length {self.min_acceptable_len} samples."
            )

        self._stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=window_fn,
            power=None,  # complex STFT
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
        )

        self._base_seed = int(seed)
        self._epoch = 0
        self._rng = torch.Generator()
        self._reset_rng()
        # Flag for warning about channel mismatch
        self._warned_channel_mismatch: int = 0

    def _reset_rng(self):
        mixed = (self._base_seed & 0xFFFFFFFF) ^ ((self._epoch * 0x9E3779B1) & 0xFFFFFFFF)
        self._rng.manual_seed(mixed)

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to vary sampling reproducibly."""
        self._epoch = int(epoch)
        self._reset_rng()

    def __len__(self) -> int:
        return len(self.files)

    def _load_waveform(self, path: Path) -> tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(str(path), normalize=True)  # (C, N)

        # If the channels do not match the required configuration, warn once
        # (only in the first epoch) and adapt the waveform accordingly.
        file_channels = int(wav.size(0))
        expected = 2 if self.stereo else 1
        if file_channels != expected and not self.skip_broken_files:
            if (self._warned_channel_mismatch==0) and (self._epoch == 0):
                got = f"{file_channels} canale{'i' if file_channels>1 else ''}"
                want = "stereo" if self.stereo else "mono"
                other = "mono" if self.stereo else "stereo"
                
                warn(
                    f"Some tracks do not match the required mode: found {got} but dataset configured for {want}. "
                    f"Some files are in {other}. This warning is shown only once (first epoch)."
                )
                self._warned_channel_mismatch = 1

            
            # Adapt the waveform: stereo->mono (downmix) or mono->stereo (duplicate)
            if expected == 1 and file_channels > 1:
                wav = wav.mean(dim=0, keepdim=True)  # downmix to mono
            elif expected == 2 and file_channels == 1:
                wav = wav.repeat(2, 1)  # duplicate mono -> stereo

        wav = wav.to(torch.float32)
        if sr != self.sample_rate:
            warn(f"Resampling from {sr} Hz to {self.sample_rate} Hz for file: {path}")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate
        return wav, sr

    def _random_crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Input: wav (C, N). Returns a segment (C, segment_samples).
        - If N >= segment_samples: random crop without padding.
        - If min_acceptable_len <= N < segment_samples: symmetric zero-padding, allowed up to max_pad_ratio.
        - Else: raise.
        """
        cur_len = wav.shape[-1]
        if cur_len >= self.segment_samples:
            max_start = cur_len - self.segment_samples
            start = int(torch.randint(low=0, high=max_start + 1, size=(1,), generator=self._rng).item())
            return wav[..., start:start + self.segment_samples]
        elif cur_len >= self.min_acceptable_len:
            pad_needed = self.segment_samples - cur_len
            if pad_needed > int(self.segment_samples * self.max_pad_ratio):
                raise RuntimeError("Requested padding exceeds allowed threshold.")
            left = pad_needed // 2
            right = pad_needed - left
            return torch.nn.functional.pad(wav, (left, right), mode="constant", value=0.0)
        else:
            raise RuntimeError("File too short relative to threshold; it should have been filtered.")

    @staticmethod
    def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
        if complex_dtype == torch.complex64:
            return torch.float32
        if complex_dtype == torch.complex128:
            return torch.float64
        # Fallback
        return torch.float32
    
    def _match_channels(self, wav: torch.Tensor) -> torch.Tensor:
        """Adatta i canali al setting stereo/mono richiesto."""
        file_channels = int(wav.size(0))
        expected = 2 if self.stereo else 1
        if file_channels == expected:
            return wav
        if expected == 1 and file_channels > 1:
            return wav.mean(dim=0, keepdim=True)       # downmix
        if expected == 2 and file_channels == 1:
            return wav.repeat(2, 1)                    # mono -> stereo
        return wav  # fallback conservativo

    
    def _load_segment_or_full(self, path: Path) -> tuple[torch.Tensor, int]:
        """
        MP3: prova a decodificare solo il segmento via frame_offset/num_frames.
        Altrimenti, fallback a load completo + crop.
        Altri formati: usa load completo + crop.
        """
        suffix = path.suffix.lower()

        # se non abbiamo un probe affidabile, fallback semplice
        if self._probe_fn is None or suffix != ".mp3":
            wav, sr = self._load_waveform(path)    # load completo
            seg = self._random_crop_or_pad(wav)    # crop/pad
            return seg.to(torch.float32), sr

        # qui: MP3 con probe disponibile
        try:
            src_sr, total_frames, _ch = self._probe_fn(path)
        except Exception:
            # Ultimo fallback
            wav, sr = self._load_waveform(path)
            seg = self._random_crop_or_pad(wav)
            return seg.to(torch.float32), sr

        # se troppo corto rispetto alla tua soglia, lascia che venga filtrato prima
        if total_frames < self.segment_samples:
            raise RuntimeError("MP3 too short; should have been filtered earlier.")

        # offset casuale nello spazio dei frame della sorgente
        start = int(torch.randint(
            0, total_frames - self.segment_samples + 1,
            (1,), generator=self._rng
        ).item())

        wav, sr = torchaudio.load(
            str(path),
            frame_offset=start,
            num_frames=self.segment_samples,
            normalize=True
        )
        wav = self._match_channels(wav).to(torch.float32)

        # In teoria, con skip_broken_files=True SR e canali sono già allineati.
        # Manteniamo un resample difensivo solo se richiesto.
        if sr != self.sample_rate and not self.skip_broken_files:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate

        return wav, sr


    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        if path.suffix.lower() == ".mp3":
            # già croppato dal loader
            seg, _ = self._load_segment_or_full(path)  # (C, segment_samples)
        else:
            wav, _ = self._load_waveform(path)         # (C, N)
            seg = self._random_crop_or_pad(wav)        # (C, segment_samples)
        
        
        seg = seg.contiguous()
        orig_waveform = seg
        S = self._stft(seg)  # (C, F, T), complex if power=None

        if not torch.is_complex(S):
            # Safety: convert (C, F, T, 2) -> complex if transform returned separate real/imag
            if S.dim() >= 4 and S.size(-1) == 2:
                S = torch.view_as_complex(S.contiguous())
            else:
                S = S.to(torch.complex64)

        S = S.to(self.dtype)

        if self.cac:
            # (C,F,T) complex -> (C,2,F,T) -> (2C,F,T) float
            S_ri = torch.stack((S.real, S.imag), dim=1)
            S = S_ri.flatten(0, 1).contiguous()  # nuovo storage
            S = S.to(self._real_dtype_for(self.dtype))
        else:
            # mantieni complesso ma assicurati storage nuovo
            S = S.contiguous()

        orig_waveform = orig_waveform.contiguous()
        return S, orig_waveform

    def _pick_probe_fn(self):
        # torchaudio.info se esiste
        if hasattr(torchaudio, "info"):
            def _probe(path):
                i = torchaudio.info(str(path))
                # In torchaudio nuove versioni: AudioMetaData con sample_rate, num_frames, num_channels
                sr = getattr(i, "sample_rate", None)
                nf = getattr(i, "num_frames", None)
                nc = getattr(i, "num_channels", getattr(i, "channels", None))
                if sr is None or nf is None or nc is None:
                    raise RuntimeError("Incomplete AudioMetaData from torchaudio.info")
                return int(sr), int(nf), int(nc)
            return _probe

        # soundfile per wav/flac/ogg ecc.
        try:
            import soundfile as sf
            def _probe(path):
                with sf.SoundFile(str(path)) as f:
                    return int(f.samplerate), int(len(f)), int(f.channels)
            return _probe
        except Exception:
            pass

        # nessun probe veloce disponibile
        return None

