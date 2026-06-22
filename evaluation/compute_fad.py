import sys
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ar_spectra.utils.console import ok, warn

from fadtk.fad import get_cache_embedding_path, calc_frechet_distance
from fadtk.model_loader import CLAPLaionModel, MERTModel

def compute_incremental_stats(files: list[Path]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute mean and covariance incrementally from cached embedding .npy files to avoid OOM."""
    n = 0
    sum_x = None
    sum_sq_x = None
    
    for f in files:
        if not f.exists(): continue
        x = np.atleast_2d(np.load(f)).astype(np.float64)
        if len(x) == 0: continue
        
        if sum_x is None:
            d = x.shape[1]
            sum_x = np.zeros(d, dtype=np.float64)
            sum_sq_x = np.zeros((d, d), dtype=np.float64)
            
        sum_x += x.sum(axis=0)
        sum_sq_x += x.T @ x
        n += len(x)
        
    if n < 2:
        return None, None
        
    mu = sum_x / n
    cov = (sum_sq_x / (n - 1)) - np.outer(mu, mu) * (n / (n - 1))
    return mu, cov


def compute_stats(fad, files: list[Path], cache_dir: Path = None, subfolder: str = None):
    """Compute mean and covariance from cached embedding .npy files."""
    model_name = fad.ml.name
    paths = []
    
    for f in files:
        if not cache_dir:
            cp = get_cache_embedding_path(model_name, f)
        else:
            if subfolder:
                cp = cache_dir / model_name / subfolder / f"{f.stem}.npy"
            else:
                cp = cache_dir / model_name / f"{f.stem}.npy"
        paths.append(cp)
    
    return compute_incremental_stats(paths)


def embed_mert(model, wav: torch.Tensor, src_sr: int, device) -> np.ndarray:
    """In-memory MERT embedding from a [C, T] waveform tensor.

    Returns (N_chunks, D) float16 array.
    Imported by evaluate_sao.py / evaluate_swin.py for the in-memory pipeline.
    """
    msr = 24000
    if src_sr != msr:
        wav = torchaudio.functional.resample(wav.cpu(), src_sr, msr)
    wav_m = wav.mean(0)  # (T,)

    cl, hl = 5 * msr, msr
    chunks = [
        F.pad(wav_m[s: s + cl], (0, max(0, cl - wav_m[s: s + cl].shape[0]))).cpu().numpy()
        for s in range(0, wav_m.shape[0], hl)
    ] or [np.zeros(cl, dtype=np.float32)]

    inputs = model.processor(chunks, sampling_rate=msr, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.model(**inputs, output_hidden_states=True)
    layer = getattr(model, "layer", 12)
    return out.hidden_states[layer].mean(1).cpu().numpy().astype(np.float16)


_FADTK_RESAMPLERS: dict[int, "torchaudio.transforms.Resample"] = {}


def _fadtk_resampler(src_sr: int, dst_sr: int = 24000):
    """torchaudio Resample with the EXACT params fadtk uses in load_audio()
    (Kaiser window). Cached per source sample rate."""
    if src_sr not in _FADTK_RESAMPLERS:
        _FADTK_RESAMPLERS[src_sr] = torchaudio.transforms.Resample(
            src_sr, dst_sr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    return _FADTK_RESAMPLERS[src_sr]


def embed_mert_framewise(model, wav: torch.Tensor, src_sr: int, device) -> np.ndarray:
    """Per-frame MERT embedding (N_frames, D), matching fadtk's canonical
    pipeline (mono mean + Kaiser resample to 24 kHz + MERTModel._get_embedding,
    NO temporal pooling).

    Use this (NOT embed_mert) whenever predictions are compared against
    reference statistics produced by the canonical fadtk pipeline, e.g. the
    pre-computed chunks_mix_original stats, which store per-frame embeddings
    (≈749 frames / 10 s clip). embed_mert mean-pools over 5 s chunks (≈10
    vectors / clip), which lives in a DIFFERENT space and inflates FAD ~50×.
    """
    msr = 24000
    x = wav.cpu().mean(0, keepdim=True)               # mono (1, T) — fadtk does mean over ch
    if src_sr != msr:
        x = _fadtk_resampler(src_sr, msr)(x)
    wav_m = x.squeeze(0).contiguous().numpy().astype(np.float32)  # (T,)
    emb = model.get_embedding(wav_m)                  # fadtk: (N_frames, 768) float16
    return np.atleast_2d(emb).astype(np.float16)


def compute_fad_from_embeddings(
    model,
    target_files:  list[Path],
    pred_emb_files: list[Path],
    shared_cache:  Path,
    csv_path:      Path,
    model_label:   str,
) -> Optional[float]:
    """Compute FAD from cached target .npy embeddings + pred .npy embeddings incrementally."""
    import csv as _csv
    model_name: str = model.name
    
    t_paths = [
        shared_cache / model_name / "target" / f"{f.stem}.npy"
        for f in target_files
    ]
    t_mu, t_cov = compute_incremental_stats(t_paths)
    
    if t_mu is None:
        warn(f"No cached target embeddings for {model_name} — FAD skipped.")
        return None

    p_mu, p_cov = compute_incremental_stats(pred_emb_files)
    if p_mu is None:
        warn(f"No pred embeddings for {model_name} — FAD skipped.")
        return None

    score = calc_frechet_distance(t_mu, t_cov, p_mu, p_cov)
    ok(f"FAD ({model_label}): {score:.6f}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "score"])
        w.writeheader()
        w.writerow({"model": model_label, "score": score})
    return score

