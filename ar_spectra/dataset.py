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

def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()


def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-62,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


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
        *,
        audio_dir: str | os.PathLike,
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
        cac: bool = False,
        seed: int = 42,
        dtype: torch.dtype = torch.complex64,
        skip_broken_files: Union[bool, Sequence[str]] = True,
        skip_criteria: Optional[Sequence[str]] = None,
        max_t_retries: int = 2,
        max_replacements: int = 8,
        length: Optional[int] = None,
        target_frames: Optional[int] = None,
        full_waveform: bool = False,
        return_paths: bool = False,
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.extensions = tuple((ext.lower() for ext in (extensions or [".wav", ".flac", ".mp3", ".ogg", ".m4a"])))

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
        self.max_t_retries = int(max_t_retries)
        self.max_replacements = int(max_replacements)
        self.full_waveform = bool(full_waveform)
        # When true, __getitem__ returns the source file path for downstream naming.
        self.return_paths = bool(return_paths)

        # --- Gestione granulare dello skip ---
        if isinstance(skip_broken_files, bool):
            if skip_broken_files:
                # Comportamento legacy: skippa per sr E canali
                if skip_criteria is not None:
                    self.skip_criteria = {crit.lower() for crit in skip_criteria}
                else:
                    self.skip_criteria = {"sample_rate", "channels"}  # default legacy
            else:
                self.skip_criteria = set()
        else: # È una sequenza di stringhe
            self.skip_criteria = {crit.lower() for crit in skip_broken_files}

        # channels info
        self.audio_channels = 2 if self.stereo else 1
        self.spec_channels = 2 * self.audio_channels if self.cac else self.audio_channels
        
        if self.full_waveform:
            self.target_frames = None
            self.segment_samples = None
            self.min_acceptable_len = 0
        else:
            # Resolve target frames length (support legacy `length` and preferred `target_frames`)
            frames = None
            if target_frames is not None:
                frames = int(target_frames)
            elif length is not None:
                frames = int(length)
            else:
                raise ValueError("OnTheFlySTFTDataset requires 'target_frames' (frames). Provide target_frames in dataset kwargs.")
            if frames < 2:
                raise ValueError("target_frames must be >= 2 to compute STFT segments.")
            self.target_frames = frames

            # number of samples in time domain for the requested spectrogram frames:
            # segment_samples = (frames - 1) * hop_length
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

                    # Controllo granulare per lo skip
                    if "sample_rate" in self.skip_criteria and src_sr != self.sample_rate:
                        mismatched_count += 1
                        continue
                    
                    if "channels" in self.skip_criteria and num_channels != self.audio_channels:
                        mismatched_count += 1
                        continue
                    if not self.full_waveform:
                        # altrimenti includi il file se la sua lunghezza stimata è accettabile
                        est_len = int(round(num_frames * (self.sample_rate / src_sr))) if src_sr > 0 else num_frames
                        if est_len >= self.min_acceptable_len:
                            self.files.append(p)
                    else:
                        self.files.append(p)
                except Exception:
                    # Non scartare l’intero dataset per errori puntuali, continua
                    continue
        # Emit a single warning if abbiamo escluso file per mismatch (solo se ne abbiamo esclusi)
        if mismatched_count > 0:
            warn(f"Excluded {mismatched_count} file(s) due to mismatch with skip_criteria: {list(self.skip_criteria)}")

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
        # NEW: flag per warning singolo sul fallback mp3
        self._warned_torchaudio_mp3_partial_fail: int = 0

    def enable_return_paths(self) -> None:
        """Enable returning source file paths alongside samples."""
        self.return_paths = True

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
        
        # Se i canali non corrispondono e non stiamo skippando per 'channels',
        # avvisa una volta (solo alla prima epoca) e adatta la forma d'onda.
        file_channels = int(wav.size(0))
        expected = 2 if self.stereo else 1
        if file_channels != expected and "channels" not in self.skip_criteria:
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
        # Resample solo se il sample rate è diverso E non stiamo skippando per 'sample_rate'
        if sr != self.sample_rate and "sample_rate" not in self.skip_criteria:
            if self._epoch == 0: # Avvisa solo alla prima epoca per evitare spam
                warn(f"Resampling from {sr} Hz to {self.sample_rate} Hz for file: {path}")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate
        return wav, sr

    def _random_crop_or_pad(self, wav: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Input: wav (C, N). Returns a segment (C, segment_samples).
        - If N >= segment_samples: random crop without padding.
        - If min_acceptable_len <= N < segment_samples: symmetric zero-padding, allowed up to max_pad_ratio.
        - Else: raise.
        Accepts optional generator to allow reproducible alternate crops.
        """
        gen = generator or self._rng
        cur_len = wav.shape[-1]
        if cur_len >= self.segment_samples:
            max_start = cur_len - self.segment_samples
            start = int(torch.randint(low=0, high=max_start + 1, size=(1,), generator=gen).item())
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

    
    def _load_segment_or_full(self, path: Path, generator: Optional[torch.Generator] = None) -> tuple[torch.Tensor, int]:
        """
        Prova a decodificare solo il segmento richiesto via frame_offset/num_frames
        per tutti i formati supportati (WAV, FLAC, MP3, ecc.).
        Fallback a load completo + crop se il partial load fallisce.

        Accepts optional generator to make the chosen start reproducible/controllable.
        """
        gen = generator or self._rng

        # se non abbiamo un probe affidabile, fallback semplice (load completo)
        if self._probe_fn is None:
            wav, sr = self._load_waveform(path)
            seg = self._random_crop_or_pad(wav, generator=gen)
            return seg.to(torch.float32), sr

        # Prova partial load per TUTTI i formati
        try:
            src_sr, total_frames, _ch = self._probe_fn(path)
        except Exception:
            wav, sr = self._load_waveform(path)
            seg = self._random_crop_or_pad(wav, generator=gen)
            return seg.to(torch.float32), sr

        # Calcola quanti sample leggere nel sample rate SORGENTE
        # per ottenere segment_samples dopo eventuale resampling
        if src_sr != self.sample_rate and "sample_rate" not in self.skip_criteria:
            # Dobbiamo leggere più sample dal file sorgente
            src_segment_samples = int(np.ceil(self.segment_samples * (src_sr / self.sample_rate)))
        else:
            src_segment_samples = self.segment_samples

        if total_frames < src_segment_samples:
            # File troppo corto, fallback a load completo con padding
            wav, sr = self._load_waveform(path)
            seg = self._random_crop_or_pad(wav, generator=gen)
            return seg.to(torch.float32), sr

        start = int(torch.randint(
            0, total_frames - src_segment_samples + 1,
            (1,), generator=gen
        ).item())

        try:
            wav, sr = torchaudio.load(
                str(path),
                frame_offset=start,
                num_frames=src_segment_samples,
                normalize=True
            )
        except Exception as e:
            # Fallback: load completo + crop
            try:
                wav, sr = self._load_waveform(path)
                seg = self._random_crop_or_pad(wav, generator=gen)
                return seg.to(torch.float32), sr
            except Exception as e2:
                raise RuntimeError(f"Partial load failed and full load fallback failed: {e2}") from e

        wav = self._match_channels(wav).to(torch.float32)
        
        # Resample se necessario
        if sr != self.sample_rate and "sample_rate" not in self.skip_criteria:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate
        
        # Dopo resampling, potremmo avere qualche sample in più o in meno
        # Crop/pad per ottenere esattamente segment_samples
        cur_len = wav.shape[-1]
        if cur_len > self.segment_samples:
            wav = wav[..., :self.segment_samples]
        elif cur_len < self.segment_samples:
            pad_needed = self.segment_samples - cur_len
            wav = F.pad(wav, (0, pad_needed), mode="constant", value=0.0)
        
        return wav, sr

    def __getitem__(self, index: int) -> torch.Tensor:
        # Minimal control: a few retries on the same file if T mismatches, otherwise skip to a different index
        MAX_T_RETRIES = self.max_t_retries       # re-crop attempts on the same file when T != target_frames
        MAX_REPLACEMENTS = self.max_replacements   # max consecutive skips to different indices before failing

        replacements = 0
        while replacements <= MAX_REPLACEMENTS:
            path = self.files[index]

            if self.full_waveform:
                try:
                    wav, _ = self._load_waveform(path)
                except Exception:
                    index = int(torch.randint(0, len(self), (1,), generator=self._rng).item())
                    replacements += 1
                    continue

                wav = wav.contiguous()
                if self.return_paths:
                    return None, wav, str(path)
                return None, wav

            # Segment extraction: use partial loading for all formats
            try:
                seg, _ = self._load_segment_or_full(path)
            except Exception:
                # Problematic file: pick a different index
                index = int(torch.randint(0, len(self), (1,), generator=self._rng).item())
                replacements += 1
                continue

            seg = seg.contiguous()

            # If the segment is silence, try a few alternative crops from the SAME file
            # using different generator seeds. If still silent, pick another file.
            if is_silence(seg):
                if len(self) > 1:
                    max_tries = 1
                    found = False
                    for _ in range(max_tries):
                        seed_val = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
                        gen = torch.Generator(); gen.manual_seed(seed_val)
                        try:
                            candidate, _sr = self._load_segment_or_full(path, generator=gen)
                        except RuntimeError:
                            continue
                        if not is_silence(candidate):
                            seg = candidate
                            found = True
                            break
                    if not found:
                        new_idx = int(torch.randint(0, len(self), (1,), generator=self._rng).item())
                        if new_idx == index and len(self) > 1:
                            new_idx = (new_idx + 1) % len(self)
                        index = new_idx
                        replacements += 1
                        continue
                # If dataset has size 1, keep seg as-is (no alternative)

            orig_waveform = seg
            S = self._stft(seg)  # (C, F, T), complex if power=None

            # Ensure complex dtype
            if not torch.is_complex(S):
                if S.dim() >= 4 and S.size(-1) == 2:
                    S = torch.view_as_complex(S.contiguous())
                else:
                    S = S.to(torch.complex64)
            S = S.to(self.dtype)

            # Time-length guard: if T != target_frames, retry a few re-crops on the SAME file, else skip file
            target_T = self.target_frames
            if S.shape[-1] != target_T:
                t_retries = 0
                success = False
                while t_retries < MAX_T_RETRIES:
                    seed_val = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())
                    gen = torch.Generator(); gen.manual_seed(seed_val)
                    try:
                        seg2, _ = self._load_segment_or_full(path, generator=gen)
                    except Exception:
                        t_retries += 1
                        continue

                    S2 = self._stft(seg2)
                    if not torch.is_complex(S2):
                        if S2.dim() >= 4 and S2.size(-1) == 2:
                            S2 = torch.view_as_complex(S2.contiguous())
                        else:
                            S2 = S2.to(torch.complex64)
                    S2 = S2.to(self.dtype)

                    if S2.shape[-1] == target_T:
                        S = S2.contiguous()
                        orig_waveform = seg2.contiguous()
                        success = True
                        break
                    t_retries += 1

                if not success:
                    # Skip this file and move to another index
                    index = int(torch.randint(0, len(self), (1,), generator=self._rng).item())
                    replacements += 1
                    continue

            # Final formatting: complex-as-channels or complex tensor
            if self.cac:
                S_ri = torch.stack((S.real, S.imag), dim=1)  # (C,2,F,T)
                S = S_ri.flatten(0, 1).contiguous()         # (2C,F,T)
                S = S.to(self._real_dtype_for(self.dtype))
            else:
                S = S.contiguous()

            orig_waveform = orig_waveform.contiguous()
            if self.return_paths:
                return S, orig_waveform, str(path)
            return S, orig_waveform

        raise RuntimeError("Too many consecutive invalid samples in the dataset. Check dataset or STFT parameters.")


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
