# ================================================================================
# Audio Probe Utilities
# 
# Contains backend-specific functions for fast, efficient reading of
# audio metadata (duration, sample rate, channels) without loading the full file.
# ================================================================================

from pathlib import Path
from typing import Union
import torchaudio
from mutagen.mp3 import MP3

def probe_mp3(path: Path):
    """Probes MP3 metadata specifically using mutagen (often faster/more reliable)."""
    meta = MP3(str(path))
    sr = int(meta.info.sample_rate)
    nf = int(meta.info.length * sr)
    nc = int(meta.info.channels)
    return sr, nf, nc

def probe_torchaudio(path: Union[str, Path]):
    """Default metadata probe using torchaudio.info()."""
    p = Path(path)
    if p.suffix.lower() == ".mp3":
        return probe_mp3(p)
    i = torchaudio.info(str(p))
    sr = getattr(i, "sample_rate", None)
    nf = getattr(i, "num_frames", None)
    nc = getattr(i, "num_channels", getattr(i, "channels", None))
    if sr is None or nf is None or nc is None:
        raise RuntimeError("Incomplete AudioMetaData")
    return int(sr), int(nf), int(nc)

def probe_soundfile(path: Union[str, Path]):
    """Fallback metadata probe using soundfile."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        return int(f.samplerate), int(len(f)), int(f.channels)

def pick_probe_fn():
    """Pick the best available probe function for file metadata."""
    if hasattr(torchaudio, "info"):
        return probe_torchaudio
    try:
        import soundfile as sf
        return probe_soundfile
    except Exception:
        return None
