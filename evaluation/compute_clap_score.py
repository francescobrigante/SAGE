import os
import sys
import argparse
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import new evaluation utilities
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from eval_dataloader import PairedEvalDataset, batch_embed_files, load_pooled_embedding, silence_output

from config import DATA_PATH, RUNS_DIR
from ar_spectra.utils.console import ok, warn, err, info

import fadtk
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import get_cache_embedding_path
import torchaudio


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat numpy arrays."""
    a = a.flatten()
    b = b.flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_clap(ml, wav: torch.Tensor, src_sr: int, device) -> np.ndarray:
    """In-memory CLAP embedding from a [C, T] waveform tensor.

    Returns (N_chunks, D) float16 array.
    Imported by evaluate.py for the in-memory pipeline.
    """
    from laion_clap.training.data import int16_to_float32, float32_to_int16

    model_sr: int = ml.sr
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
        embs = ml.model.get_audio_embedding_from_data(x=tensor, use_tensor=True)
    return embs.cpu().numpy().astype(np.float16)

def main():
    parser = argparse.ArgumentParser(description="Batch-optimized CLAP evaluation.")
    parser.add_argument("--target-dir", default=str(DATA_PATH))
    parser.add_argument("--preds-dir", default=str(RUNS_DIR / "inference"))
    parser.add_argument("--model", default="both", choices=["music", "audio", "both"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, help="Per-checkpoint cache (predictions).")
    parser.add_argument("--shared-cache-dir", type=Path, default=None,
                        help="Shared cache for target embeddings (reused across checkpoints).")
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--extensions", default="wav,mp3,flac")
    args = parser.parse_args()

    # 0. Load Data
    from config import FMA_METADATA
    exts = ["." + e.strip().lstrip(".") for e in args.extensions.split(",")]
    dataset = PairedEvalDataset(
        target_dir=args.target_dir, 
        preds_dir=args.preds_dir, 
        extensions=exts, 
        max_files=args.max_files,
        fma_csv_path=FMA_METADATA
    )
    info(f"Found {len(dataset)} matched pairs.")

    # pred_subfolder is unique per checkpoint run; target cache may be shared across checkpoints
    ckpt_name = Path(args.preds_dir).name
    pred_subfolder = f"preds_{ckpt_name}"
    target_cache = args.shared_cache_dir if args.shared_cache_dir else args.cache_dir

    flavours = ["music", "audio"] if args.model == "both" else [args.model]
    scores_by_flavour = {}

    for flavour in flavours:
        info(f"Processing CLAP-{flavour}...")
        with silence_output():
            ml = CLAPLaionModel(flavour)
            fad = fadtk.FrechetAudioDistance(ml)
        ok(f"CLAP-{flavour} model initialized successfully.")

        model_name = fad.ml.name

        # 1. Collect unique files for target and prediction
        target_files = sorted(list({p[0] for p in dataset.pairs}))
        pred_files = sorted(list({p[1] for p in dataset.pairs}))

        # 2. Batch Embed — targets go to shared cache, predictions to per-checkpoint cache
        if args.cache_dir or target_cache:
            info(f"Embedding target files for CLAP-{flavour} → shared cache...")
            batch_embed_files(fad, target_files, batch_size=args.batch_size, cache_dir=target_cache, subfolder="target")
            info(f"Embedding prediction files for CLAP-{flavour} → '{pred_subfolder}'...")
            batch_embed_files(fad, pred_files, batch_size=args.batch_size, cache_dir=args.cache_dir, subfolder=pred_subfolder)
        else:
            batch_embed_files(fad, target_files + pred_files, batch_size=args.batch_size)

        # 3. Calculate similarity scores
        results = []
        all_scores = []

        for tpath, ppath in tqdm(dataset.pairs, desc=f"CLAP-{flavour} Scoring"):
            try:
                t_base = target_cache or args.cache_dir
                p_base = args.cache_dir or target_cache
                if t_base:
                    cp_t = t_base / model_name / "target" / f"{tpath.stem}.npy"
                    cp_p = (p_base / model_name / pred_subfolder / f"{ppath.stem}.npy"
                            if args.cache_dir else get_cache_embedding_path(model_name, ppath))
                else:
                    cp_t = get_cache_embedding_path(model_name, tpath)
                    cp_p = get_cache_embedding_path(model_name, ppath)

                s = cosine_sim(load_pooled_embedding(cp_t), load_pooled_embedding(cp_p))
                all_scores.append(s)
                results.append({"target_file": tpath.name, "pred_file": ppath.name, f"clap_{flavour}": s})
            except Exception as e:
                err(f"Error on {tpath.stem}: {e}")

        scores_by_flavour[flavour] = results
        if all_scores:
            final_mean = np.mean(all_scores)
            ok(f"CLAP-{flavour} Final Score: {final_mean:.6f}")
        else:
            warn(f"No scores calculated for CLAP-{flavour}.")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            if args.model == "both":
                fieldnames = ["target_file", "pred_file", "clap_music", "clap_audio"]
            else:
                fieldnames = ["target_file", "pred_file", f"clap_{args.model}"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if args.model == "both" and "music" in scores_by_flavour and "audio" in scores_by_flavour:
                # Merge results by target/pred file pairs
                merged = []
                # Assume pairs are in the same order
                for r_m, r_a in zip(scores_by_flavour["music"], scores_by_flavour["audio"]):
                    row = r_m.copy()
                    row["clap_audio"] = r_a["clap_audio"]
                    writer.writerow(row)
            else:
                writer.writerows(scores_by_flavour.get(args.model, []))
        info(f"Saved CSV: {out_path}")

if __name__ == "__main__":
    main()
