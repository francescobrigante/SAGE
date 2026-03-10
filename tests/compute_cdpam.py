#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import torch

# Add project root to path so we can import ar_spectra and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE
from ar_spectra.utils.console import ok, warn, err, info
import numpy as np

_SKIP_DIRS = frozenset({"metrics", "convert", "embeddings"})

# ── MONKEY-PATCH: cdpam compatibility fixes ───────────────────────────────────
# PyTorch 2.6+ requires weights_only to be explicit in torch.load.
# cdpam calls torch.load without it, so we intercept and supply the default.
_real_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _real_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import cdpam  # import after patch so cdpam.cdpam picks up the patched torch.load

# NumPy 1.24+ removed np.float as an alias for float.
# cdpam.__init__ calls inputData.astype(float) which is fine, but internally
# it also uses np.float in some versions — patch it back if missing.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
# ── END MONKEY-PATCH ──────────────────────────────────────────────────────────
from tqdm import tqdm
import csv
from ar_spectra.utils.reproducibility import configure_reproducibility

configure_reproducibility(seed=42, deterministic=False, strict_deterministic=False)

def ensure_batch(x):
    """
    Converte una waveform 1D (L,) in (1, L).
    Lascia intatta (1, L).
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Atteso np.ndarray, ottenuto {type(x)}")
    if x.ndim == 1:
        return x[None, :]
    elif x.ndim == 2 and x.shape[0] == 1:
        return x
    else:
        raise ValueError(f"Waveform shape inattesa {x.shape}; atteso (L,) o (1,L).")

def crop_to_min_length(a, b):
    L = min(a.shape[-1], b.shape[-1])
    return a[..., :L], b[..., :L]

def chunk_iterator(a, b, chunk_size):
    """
    Genera segmenti (1, chunk_size).
    """
    L = a.shape[-1]
    usable = (L // chunk_size) * chunk_size
    for start in range(0, usable, chunk_size):
        end = start + chunk_size
        yield a[..., start:end], b[..., start:end]

def compute_score(model, wav_ref, wav_out, chunk_size=None):
    """
    Restituisce uno scalare GPU (shape []).
    """
    wav_ref, wav_out = crop_to_min_length(wav_ref, wav_out)

    if chunk_size and wav_ref.shape[-1] > chunk_size:
        scores = []
        for r_seg, o_seg in chunk_iterator(wav_ref, wav_out, chunk_size):
            with torch.no_grad():
                s = model.forward(r_seg, o_seg)   # tipicamente (1,1) o (1,)
            # Normalizzo subito a scalare per evitare shape (1,1)
            s = s.flatten().mean()
            scores.append(s)
        return torch.stack(scores).mean()   # già scalare
    else:
        with torch.no_grad():
            s = model.forward(wav_ref, wav_out)
        return s.flatten().mean()  # scalare

def match_pairs(target_dir, preds_dir, allowed_ext=None):
    target_dir = Path(target_dir)
    preds_dir = Path(preds_dir)

    target_map = {}
    for f in target_dir.rglob("*"):
        if f.is_file():
            stem = f.stem
            ext = f.suffix.lower().lstrip(".")
            if allowed_ext and ext not in allowed_ext:
                continue
            if _SKIP_DIRS.intersection(f.relative_to(target_dir).parts[:-1]):
                continue
            target_map.setdefault(stem, []).append(f)

    pairs = []
    for g in preds_dir.rglob("*"):
        if g.is_file():
            stem = g.stem
            ext = g.suffix.lower().lstrip(".")
            if allowed_ext and ext not in allowed_ext:
                continue
            if _SKIP_DIRS.intersection(g.relative_to(preds_dir).parts[:-1]):
                continue
            if stem in target_map:
                pairs.append((target_map[stem][0], g))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Confronto CDPAM tra due directory (preds vs target).")
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--extensions", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--csv_out", type=str, default="")
    args = parser.parse_args()

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        warn("CUDA non disponibile, uso CPU.")
        device = "cpu"
    else:
        device = args.device

    # Modello (usa pesi di default)
    loss_fn = cdpam.CDPAM(dev=device)

    allowed_ext = None
    if args.extensions.strip():
        allowed_ext = {e.strip().lower().lstrip(".") for e in args.extensions.split(",") if e.strip()}

    pairs = match_pairs(args.target_dir, args.preds_dir, allowed_ext=allowed_ext)
    if not pairs:
        warn("Nessuna coppia trovata.")
        return

    # score_sum come scalare GPU
    score_sum = torch.zeros((), device=loss_fn.device)
    count = 0
    per_file = []
    chunk_size = args.chunk_size if args.chunk_size > 0 else None

    pbar = tqdm(pairs, desc="Calcolo CDPAM", unit="pair")

    try:
     for tpath, ppath in pbar:
        try:
            # Carica audio (usa loader del package)
            wav_target = cdpam.load_audio(str(tpath))
            wav_pred = cdpam.load_audio(str(ppath))

            wav_target = ensure_batch(wav_target)  # (1,L)
            wav_pred   = ensure_batch(wav_pred)    # (1,L)

            score_tensor = compute_score(loss_fn, wav_target, wav_pred, chunk_size=chunk_size)
            # Assicuriamoci sia scalare shape []
            if score_tensor.ndim != 0:
                score_tensor = score_tensor.flatten().mean()
        except Exception as e:
            err(f"Errore su coppia {tpath.name}/{ppath.name}: {e}")
            continue

        score_sum = score_sum + score_tensor  # ora non è in-place += (evita alcuni edge case broadcast)
        count += 1
        mean_tensor = score_sum / count
        pbar.set_postfix({"mean_score": f"{mean_tensor.item():.4f}"})

        per_file.append({
            "target_file": tpath.name,
            "pred_file": ppath.name,
            "score": score_tensor.item()
        })
    except KeyboardInterrupt:
        pbar.close()
        warn("[Interrupted] Printing partial results...")

    if count == 0:
        warn("Nessun punteggio calcolato.")
        return

    final_mean = (score_sum / count).item()
    ok(f"Score medio CDPAM: {final_mean:.6f} su {count} coppie.")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["target_file", "pred_file", "score"])
            writer.writeheader()
            writer.writerows(per_file)
        info(f"Salvato CSV: {out_path}")

if __name__ == "__main__":
    main()