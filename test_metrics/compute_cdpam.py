#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script per valutare un intero dataset con CDPAM.

Esempi d'uso:
1) Directory parallele (matching per stem, estensioni diverse):
    python eval_cdpam.py \
        --ref_dir data/refs_mp3 \
        --pred_dir data/preds_wav \
        --pattern "*.mp3" \
        --pred_ext wav \
        --out results.csv

2) Mapping da file CSV (colonne: ref,pred):
    python eval_cdpam.py --pairs_csv pairs.csv --out results.csv

3) JSON output + progress bar:
    python eval_cdpam.py --ref_dir data/refs --pred_dir data/preds --json results.json

Opzioni utili:
    --sr 16000           Forza resampling
    --limit 100          Valuta solo le prime 100 coppie
    --workers 4          Caricamento I/O dimostrativo (modello seriale)
    --pred_ext wav,flac  Specifica le estensioni delle preds
"""
import argparse
import os
import sys
import glob
import json
import csv
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None

import torch

# Tentativo di import del modello CDPAM (adatta se necessario)
try:
    from cdpam import CDPAM
except ImportError:
    CDPAM = None
    # L'utente dovrà installare il pacchetto corretto.


def match_by_stem_multi_ext(ref_files,
                            pred_dir,
                            pred_patterns=("*.wav","*.mp3","*.flac","*.ogg")):
    """
    Crea coppie abbinando per stem (nome senza estensione). Se ci sono più file
    con stesso stem nelle preds prende la prima occorrenza trovata (puoi cambiare politica).
    """
    pred_index = {}
    for pat in pred_patterns:
        for p in glob.glob(os.path.join(pred_dir, pat)):
            stem = Path(p).stem.lower()
            pred_index.setdefault(stem, p)  # mantiene la prima
    pairs = []
    for r in ref_files:
        stem = Path(r).stem.lower()
        if stem in pred_index:
            pairs.append((r, pred_index[stem]))
    return pairs

@dataclass
class PairResult:
    ref_path: str
    pred_path: str
    score: Optional[float]
    duration_sec: Optional[float]
    error: Optional[str]

def list_audio_files(directory: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, pattern)))

def load_audio(path: str, target_sr: Optional[int]=None) -> Tuple[np.ndarray, int]:
    """
    Carica audio. Usa soundfile se possibile, altrimenti fallback a librosa (necessario per MP3).
    Converte in mono (media canali). Resample se target_sr specificato.
    """
    try:
        data, sr = sf.read(path)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if target_sr and sr != target_sr:
            if not librosa:
                raise RuntimeError("Resampling richiesto ma librosa non installato.")
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data, sr
    except Exception:
        if not librosa:
            raise
        data, sr = librosa.load(path, sr=target_sr, mono=True)
        data = data.astype(np.float32, copy=False)
        return data, sr

def compute_cdpam_score(model,
                        ref_audio: np.ndarray, ref_sr: int,
                        pred_audio: np.ndarray, pred_sr: int,
                        device: str) -> float:
    """
    Adatta se l'API del modello è differente. Qui si assume che model(ref_t, pred_t) funzioni.
    """
    min_len = min(len(ref_audio), len(pred_audio))
    ref_audio = ref_audio[:min_len]
    pred_audio = pred_audio[:min_len]

    ref_t = torch.from_numpy(ref_audio).unsqueeze(0).to(device)
    pred_t = torch.from_numpy(pred_audio).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            score = model(ref_t, pred_t)
        except TypeError:
            try:
                score = model.forward(ref_t, pred_t)
            except Exception:
                # fallback se il modello richiede sr esplicite
                score = model(ref_t, ref_sr, pred_t, pred_sr)

    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().item()
    elif isinstance(score, (list, tuple)):
        score = float(score[0])
    return float(score)

def evaluate_pairs(pairs: List[Tuple[str, str]],
                   model,
                   device: str,
                   target_sr: Optional[int]=None,
                   limit: Optional[int]=None,
                   workers: int=1,
                   show_progress: bool=True) -> List[PairResult]:
    if limit:
        pairs = pairs[:limit]

    results: List[PairResult] = []

    def progress(i, total):
        if not show_progress:
            return
        bar_len = 30
        filled = int(bar_len * (i+1) / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {i+1}/{total}", end='', flush=True)

    if workers <= 1:
        for i, (ref_path, pred_path) in enumerate(pairs):
            try:
                ref_audio, ref_sr = load_audio(ref_path, target_sr)
                pred_audio, pred_sr = load_audio(pred_path, target_sr)
                score = compute_cdpam_score(model, ref_audio, ref_sr, pred_audio, pred_sr, device)
                duration = len(ref_audio) / ref_sr
                results.append(PairResult(ref_path, pred_path, score, duration, None))
            except Exception as e:
                results.append(PairResult(ref_path, pred_path, None, None, f"{type(e).__name__}: {e}"))
            progress(i, len(pairs))
    else:
        from multiprocessing import Pool

        def _worker(args):
            ref_path, pred_path, target_sr = args
            try:
                ref_audio, ref_sr = load_audio(ref_path, target_sr)
                pred_audio, pred_sr = load_audio(pred_path, target_sr)
                return PairResult(ref_path, pred_path, None, None,
                                  "Calcolo modello non parallelizzato (usa --workers 1)")
            except Exception as e:
                return PairResult(ref_path, pred_path, None, None, f"{type(e).__name__}: {e}")

        print("ATTENZIONE: --workers > 1 parallelizza solo I/O, non il modello.")
        with Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap(_worker, [(r, p, target_sr) for r, p in pairs])):
                results.append(res)
                progress(i, len(pairs))

    print()
    return results

def summarize(results: List[PairResult]) -> Dict[str, float]:
    scores = [r.score for r in results if r.score is not None]
    if not scores:
        return {"mean": math.nan, "std": math.nan, "count": 0}
    arr = np.array(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(len(arr))
    }

def save_results(results: List[PairResult],
                 summary: Dict[str, float],
                 out_csv: Optional[str]=None,
                 out_json: Optional[str]=None):
    rows = []
    for r in results:
        rows.append({
            "ref_path": r.ref_path,
            "pred_path": r.pred_path,
            "score": r.score,
            "duration_sec": r.duration_sec,
            "error": r.error
        })
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    if out_json:
        payload = {"summary": summary, "results": rows}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

def read_pairs_csv(path: str) -> List[Tuple[str,str]]:
    pairs = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row.get("ref")
            pred = row.get("pred")
            if ref and pred and os.path.isfile(ref) and os.path.isfile(pred):
                pairs.append((ref, pred))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Valutazione dataset con CDPAM")
    g_input = parser.add_argument_group("Input dataset")
    g_input.add_argument("--ref_dir", type=str,
                         default="/home/cerovaz/repos/data/jamendo_full/test_trimmed",
                         help="Directory con audio di riferimento")
    g_input.add_argument("--pred_dir", type=str,
                         default="/home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/real_dataset_outputs_24epoch",
                         help="Directory con audio predetti")
    g_input.add_argument("--pattern", type=str, default="*.wav",
                         help="Pattern dei file di riferimento (es: *.mp3 se i ref sono MP3)")
    g_input.add_argument("--pairs_csv", type=str,
                         help="CSV con colonne ref,pred (salta matching automatico)")
    g_input.add_argument("--pred_ext", type=str,
                         default="wav,mp3,flac,ogg",
                         help="Estensioni (senza punto) dei file predetti separate da virgola (default: wav,mp3,flac,ogg)")

    g_proc = parser.add_argument_group("Processing")
    g_proc.add_argument("--sr", type=int, default=None, help="Forza resampling a questa frequenza")
    g_proc.add_argument("--limit", type=int, default=None, help="Limita il numero di coppie")
    g_proc.add_argument("--workers", type=int, default=1, help="Numero processi I/O (modello seriale)")
    g_proc.add_argument("--device", type=str, default="cuda", help="Device (cpu|cuda)")

    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--out", type=str, help="Scrivi risultati per-coppia in CSV")
    g_out.add_argument("--json", type=str, help="Scrivi risultati + riepilogo in JSON")
    g_out.add_argument("--no-progress", action="store_true", help="Disabilita progress bar")

    args = parser.parse_args()

    if not args.pairs_csv and (not args.ref_dir or not args.pred_dir):
        print("Errore: specifica --pairs_csv oppure --ref_dir e --pred_dir.")
        sys.exit(1)

    if args.pairs_csv:
        if not os.path.isfile(args.pairs_csv):
            print(f"File pairs_csv non trovato: {args.pairs_csv}")
            sys.exit(1)
        pairs = read_pairs_csv(args.pairs_csv)
        if not pairs:
            print("Nessuna coppia valida trovata nel CSV.")
            sys.exit(1)
    else:
        if not os.path.isdir(args.ref_dir):
            print(f"ref_dir non valida: {args.ref_dir}")
            sys.exit(1)
        if not os.path.isdir(args.pred_dir):
            print(f"pred_dir non valida: {args.pred_dir}")
            sys.exit(1)
        ref_files = list_audio_files(args.ref_dir, args.pattern)
        if not ref_files:
            print("Nessun file di riferimento trovato.")
            sys.exit(1)
        pred_patterns = tuple(f"*.{ext.strip()}" for ext in args.pred_ext.split(",") if ext.strip())
        pairs = match_by_stem_multi_ext(ref_files, args.pred_dir, pred_patterns=pred_patterns)
        if not pairs:
            print("Nessuna coppia abbinata (verifica estensioni o nomi).")
            sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if CDPAM is None:
        print("Il modulo cdpam non è importabile. Installa il pacchetto appropriato.")
        print("Esempio: pip install git+https://github.com/pranaymanocha/PerceptualAudio")
        sys.exit(1)

    print(f"Carico modello CDPAM su device {device}...")
    try:
        model = CDPAM()
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
    except Exception as e:
        print("Errore nell'istanziamento del modello CDPAM:", e)
        traceback.print_exc()
        sys.exit(1)

    print(f"Coppie da valutare: {len(pairs)}")
    start = time.time()
    results = evaluate_pairs(
        pairs,
        model=model,
        device=device,
        target_sr=args.sr,
        limit=args.limit,
        workers=args.workers,
        show_progress=not args.no_progress
    )
    elapsed = time.time() - start
    summary = summarize(results)

    print("\n=== RISULTATI ===")
    print(f"Tempo totale: {elapsed:.2f}s")
    print(f"Numero coppie valide: {summary['count']}")
    print(f"Media CDPAM: {summary['mean']:.6f}")
    print(f"Deviazione standard: {summary['std']:.6f}")

    if args.out or args.json:
        save_results(results, summary, args.out, args.json)
        print("File salvati.")

    valid = [r for r in results if r.score is not None]
    print("\nEsempio (prime 5):")
    for r in valid[:5]:
        print(f"{os.path.basename(r.ref_path)} | score={r.score:.6f} | dur={r.duration_sec:.2f}s")

if __name__ == "__main__":
    main()