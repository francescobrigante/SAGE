# ===============================================================================
# Audio Tensor Utilities
#
#   contains tensor manipulations applied to audio (trimming, folding channels)
# =================================================================================

import torch
from einops import rearrange
import torch.nn.functional as F
from config import DEFAULT_SILENCE_THRESHOLD
import torchaudio

def trim_to_shortest(a: torch.Tensor, b: torch.Tensor):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:, :, :b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:, :, :a.shape[-1]]
    return a, b

def fold_channels_into_batch(x: torch.Tensor) -> torch.Tensor:
    """Fold channel dimension into batch dimension."""
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(spec_bc: torch.Tensor, channels: int) -> torch.Tensor:
    """Restores the channel dimension that was folded into the batch dimension."""
    Bc, F, T = spec_bc.shape
    B = Bc // channels
    return spec_bc.reshape(B, channels, F, T)

def get_dbmax(audio: torch.Tensor) -> float:
    """Finds the loudest value in the entire clip and puts that into dB (full scale)."""
    return 20 * torch.log10(torch.flatten(audio.abs()).max()).cpu().item()

def is_silence(audio: torch.Tensor, thresh: float = DEFAULT_SILENCE_THRESHOLD) -> bool:
    """Checks if entire clip is 'silence' below some dB threshold."""
    return get_dbmax(audio) < thresh

def load_waveform(path: str, target_sample_rate: int, expected_channels: int) -> tuple[torch.Tensor, int, bool, bool]:
    """
    Loads an audio waveform from disk and applies standard conversions (stereo/mono, resampling).

    Args:
        path: Path to the audio file.
        target_sample_rate: Expected sample rate for resampling.
        expected_channels: 1 for mono, 2 for stereo.
    
    Returns:
        wav: The processed waveform tensor [channels, time].
        sr: The sample rate of the returned waveform.
        channel_mismatched: True if original channels differed from expected_channels.
        sr_mismatched: True if original sample rate differed from target_sample_rate.
    """
    wav, sr = torchaudio.load(str(path), normalize=True)
    wav = wav.to(torch.float32)
    
    file_channels = wav.size(0)
    channel_mismatched = (file_channels != expected_channels)
    sr_mismatched = (sr != target_sample_rate)
    
    if channel_mismatched:
        if expected_channels == 1 and file_channels > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif expected_channels == 2 and file_channels == 1:
            wav = wav.repeat(2, 1)
            
    if sr_mismatched:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        sr = target_sample_rate
        
    return wav, sr, channel_mismatched, sr_mismatched

def random_crop_or_pad(wav: torch.Tensor, segment_samples: int, min_acceptable_len: int, rng: torch.Generator) -> torch.Tensor:
    """
    Randomly crops a segment of exactly `segment_samples` from the waveform.
    If the waveform is shorter but above `min_acceptable_len`, it pads it with silence (zeros) symmetrically.
    """
    cur_len = wav.shape[-1]
    
    if cur_len >= segment_samples:
        max_start = cur_len - segment_samples
        start = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
        return wav[..., start:start + segment_samples]
    elif cur_len >= min_acceptable_len:
        pad_needed = segment_samples - cur_len
        left = pad_needed // 2
        right = pad_needed - left
        return F.pad(wav, (left, right), mode="constant", value=0.0)
    else:
        raise RuntimeError(f"File too short: {cur_len} samples < {min_acceptable_len}")
