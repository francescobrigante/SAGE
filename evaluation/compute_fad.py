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

def compute_stats(fad, files: list[Path], cache_dir: Path = None, subfolder: str = None):
    """Compute mean and covariance from cached embedding .npy files."""
    model_name = fad.ml.name
    embeddings = []
    
    for f in files:
        if not cache_dir:
            cp = get_cache_embedding_path(model_name, f)
        else:
            if subfolder:
                cp = cache_dir / model_name / subfolder / f"{f.stem}.npy"
            else:
                cp = cache_dir / model_name / f"{f.stem}.npy"
        
        if cp.exists():
            # (T, D) -> pool to (D,)
            emb = np.atleast_2d(np.load(cp)).astype(np.float32)
            embeddings.append(emb)
    
    if not embeddings:
        return None, None
        
    all_embs = np.concatenate(embeddings, axis=0)
    mu = np.mean(all_embs, axis=0)
    cov = np.cov(all_embs, rowvar=False)
    return mu, cov


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


def compute_fad_from_embeddings(
    model,
    target_files:  list[Path],
    pred_emb_list: list[np.ndarray],
    shared_cache:  Path,
    csv_path:      Path,
    model_label:   str,
) -> Optional[float]:
    """Compute FAD from cached target .npy embeddings + in-memory pred embeddings.

    Used by evaluate_sao.py / evaluate_swin.py — no WAV files are written to disk.
    """
    import csv as _csv
    model_name: str = model.name
    t_embs = [
        np.load(shared_cache / model_name / "target" / f"{f.stem}.npy").astype(np.float32)
        for f in target_files
        if (shared_cache / model_name / "target" / f"{f.stem}.npy").exists()
    ]
    if not t_embs:
        warn(f"No cached target embeddings for {model_name} — FAD skipped.")
        return None

    all_t = np.concatenate(t_embs, axis=0)
    all_p = np.concatenate(pred_emb_list, axis=0).astype(np.float32)
    score = calc_frechet_distance(
        all_t.mean(0), np.cov(all_t, rowvar=False),
        all_p.mean(0), np.cov(all_p, rowvar=False),
    )
    ok(f"FAD ({model_label}): {score:.6f}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "score"])
        w.writeheader()
        w.writerow({"model": model_label, "score": score})
    return score

