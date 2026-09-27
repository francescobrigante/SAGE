# =============================================================================
# Fréchet audio distances: framewise MERT (layer 4, fadtk pipeline), whole-file
# CLAP ("GUD", see metrics/clap.py) and PANN Cnn14; incremental mean/covariance
# over per-file embeddings, and the FAD of a prediction set against targets.
# =============================================================================
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from evaluation.common import downmix, CHANNEL_MID
from sage.utils.console import ok, warn

from fadtk.fad import calc_frechet_distance

# PANN (Cnn14_16k) FAD embedder: shared name used as cache subdir + FAD label everywhere.
PANN_NAME = "pann-cnn14-16k"
# Cnn14 has 5 time-pooling stages after its mel front-end (hop 160), so it needs
# ≳5k samples @16k or the last conv collapses to size 0. Real clips are ≥10 s; this
# guard only protects against pathologically short inputs (pad to 1 s @16k).
_PANN_MIN_SAMPLES = 16000
# Cnn14_16k_mAP=0.438.pth (paths.pann, set by the entry points; https://zenodo.org/record/3987831)
PANN_CHECKPOINT: Optional[str] = None

def compute_incremental_stats(files: list[Path]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute mean and covariance incrementally from cached embedding .npy files to avoid OOM."""
    n = 0
    sum_x = None
    sum_sq_x = None
    
    for f in files:
        if not f.exists(): continue
        x = np.atleast_2d(np.load(f)).astype(np.float64)
        if len(x) == 0: continue
        # Un solo vettore non-finito avvelena mu/cov in silenzio (il Side di un
        # file quasi-mono e' ~0 e la normalizzazione per-input puo' dividere per
        # ~0). Scarta le righe non finite invece di propagarle.
        finite = np.isfinite(x).all(axis=1)
        if not finite.all():
            warn(f"{(~finite).sum()}/{len(x)} embedding non finiti scartati: {f.name}")
            x = x[finite]
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


def embed_mert_framewise(model, wav: torch.Tensor, src_sr: int, device,
                         channel: str = CHANNEL_MID) -> np.ndarray:
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
    x = downmix(wav.cpu(), channel).unsqueeze(0)      # (1, T) — fadtk does mean over ch
    if src_sr != msr:
        x = _fadtk_resampler(src_sr, msr)(x)
    wav_m = x.squeeze(0).contiguous().numpy().astype(np.float32)  # (T,)
    emb = model.get_embedding(wav_m)                  # fadtk: (N_frames, 768) float16
    return np.atleast_2d(emb).astype(np.float16)


_pann_model = None


def get_pann_model(device):
    """PANN Cnn14_16k whole-file embedder — singleton, mirrors get_gud_model.

    Uses the Cnn14_16k bundled in the installed ``frechet_audio_distance`` package
    (2048-dim, 16 kHz). The checkpoint (``Cnn14_16k_mAP=0.438.pth``) is read from
    PANN_CHECKPOINT (paths.pann): it is not downloaded, compute nodes are often offline.
    """
    global _pann_model
    if _pann_model is None:
        from frechet_audio_distance.models.pann import Cnn14_16k
        m = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160,
                      mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        if not PANN_CHECKPOINT or not Path(PANN_CHECKPOINT).is_file():
            raise FileNotFoundError(f"PANN checkpoint not found: {PANN_CHECKPOINT} (set paths.pann, env PANN_CKPT or "
                                    "SAGE_MODELS; download: https://zenodo.org/record/3987831)")
        ckpt_path = Path(PANN_CHECKPOINT)
        # weights_only=False: the Cnn14 ckpt pickles numpy arrays; torch>=2.6 defaults
        # weights_only=True and rejects them. The file is a trusted Zenodo download.
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        m.load_state_dict(checkpoint["model"])
        _pann_model = m.to(device).eval()
    return _pann_model


def embed_pann(wav: torch.Tensor, src_sr: int, device,
               channel: str = CHANNEL_MID) -> np.ndarray:
    """In-memory whole-file PANN (Cnn14_16k) embedding from a [C, T] waveform.

    Mono-mix + Kaiser resample to 16 kHz + Cnn14 forward, mirroring the
    frechet_audio_distance 'pann' pipeline. Returns (1, 2048) float16.
    """
    import resampy

    model = get_pann_model(device)
    wav_mono = downmix(wav, channel).cpu().numpy()                        # (T,)
    if src_sr != 16000:
        wav_mono = resampy.resample(wav_mono, src_sr, 16000)             # (T',)
    if wav_mono.shape[0] < _PANN_MIN_SAMPLES:                            # pad pathologically short clips
        wav_mono = np.pad(wav_mono, (0, _PANN_MIN_SAMPLES - wav_mono.shape[0]))
    tensor = torch.from_numpy(wav_mono).float().unsqueeze(0).to(device)  # (1, T')
    with torch.no_grad():
        out = model(tensor, None)
        embd = out["embedding"].data                                     # (1, 2048)
    return embd.cpu().numpy().astype(np.float16)                         # (1, 2048)


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

