import sys
import torch
import numpy as np
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torchaudio


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat numpy arrays."""
    a = a.flatten()
    b = b.flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_clap(model, wav: torch.Tensor, src_sr: int, device) -> np.ndarray:
    """In-memory CLAP embedding from a [C, T] waveform tensor.

    Returns (N_chunks, D) float16 array.
    Imported by evaluate_sao.py / evaluate_swin.py for the in-memory pipeline.
    """
    from laion_clap.training.data import int16_to_float32, float32_to_int16

    model_sr: int = model.sr
    if src_sr != model_sr:
        wav = torchaudio.functional.resample(wav.cpu(), src_sr, model_sr)
    wav_np = int16_to_float32(float32_to_int16(wav.mean(0).numpy().reshape(1, -1)))

    cs = 10 * model_sr   # 10-s chunk
    hs = model_sr        # 1-s hop
    T  = wav_np.shape[1]
    chunks = [np.pad(wav_np[0, s: s + cs], (0, max(0, cs - (T - s))))
              for s in range(0, T, hs)] or [np.zeros(cs, dtype=np.float32)]

    tensor = torch.from_numpy(np.stack(chunks)).float().to(device)
    with torch.no_grad():
        embs = model.model.get_audio_embedding_from_data(x=tensor, use_tensor=True)
    return embs.cpu().numpy().astype(np.float16)

