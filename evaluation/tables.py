#!/usr/bin/env python3
# =============================================================================
# The paper's tables from the outputs of the evaluation scripts, as markdown:
#
#   python -m evaluation.tables \
#       --recon "FMA test=results/recon_fma" --recon "MoisesDB mixtures=results/recon_moisesdb_mix" \
#       --maeb results/maeb/SAGE_FTe992 --maeb results/maeb/same-s
#
#   --recon LABEL=DIR   one evaluation set; DIR/<model>/metrics/ as written by
#                       evaluation.reconstruction → Table 2, plus the stereo-image
#                       metrics when ms_metrics.csv exists (compute_ms_metrics=true)
#   --maeb DIR          the MAEB output of one encoder (evaluation.maeb) → Table 9
#                       and the block averages of Table 4; a directory named
#                       "clap" is the oracle row, shown but never bolded
# Per-file metrics are averaged over files; FADs are corpus-level scalars.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Table 2 columns: (label, source CSV, column, higher is better)
RECON_COLUMNS = [
    ("FAD MERT", "fad_mert.csv", "score", False),
    ("FAD CLAP", "fad_gudgud.csv", "score", False),
    ("FAD PANN", "fad_pann.csv", "score", False),
    ("CLAP music", "clap_music.csv", "cosine", True),
    ("CLAP audio", "clap_audio.csv", "cosine", True),
    ("SDR", "spectral.csv", "sdr", True),
    ("dSTFT", "spectral.csv", "stft_loss", False),
]
# Stereo-image columns (ms_metrics.csv). Table 7 of the paper compares phase-1
# checkpoints with and without L_SD; its "Side level (dB)" column is not one of these.
STEREO_COLUMNS = [("width bias", "width_bias"), ("d_width", "d_width"),
                  ("SI-SDR side", "sisdr_s"), ("SI-SDR mid", "sisdr_m")]
# The nineteen probing tasks, in three blocks (Table 4 averages each block).
MAEB_BLOCKS = {
    "FMA": ["FMAGenreClassification", "FMAGenreClustering", "FMAArtistClustering",
            "FMAArtistA2ARetrieval", "FMAGenreAudioReranking", "FMAArtistPairClassification"],
    "MoisesDB": ["MoisesDBGenreClassification", "MoisesDBGenreClustering", "MoisesDBArtistClustering",
                 "MoisesDBArtistA2ARetrieval", "MoisesDBGenreAudioReranking",
                 "MoisesDBArtistPairClassification", "MoisesDBInstrumentClassification"],
    "Upstream MAEB": ["GTZANGenre", "GTZANGenreClustering", "MusicGenreClustering",
                      "GTZANAudioReranking", "NSynth", "JamAltArtistA2ARetrieval"],
}


def _column(path: Path, col: str) -> np.ndarray:
    """Finite values of one CSV column (NaN rows, e.g. mono references, are dropped)."""
    with open(path, newline="") as f:
        vals = [float(r[col]) for r in csv.DictReader(f) if r.get(col) not in (None, "")]
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _value(metrics: Path, csv_name: str, col: str) -> float:
    path = metrics / csv_name
    if not path.exists():
        return float("nan")
    arr = _column(path, col)
    return float(arr.mean()) if arr.size else float("nan")


def recon_table(root: Path) -> pd.DataFrame:
    rows = {}
    for metrics in sorted(root.glob("*/metrics")):
        rows[metrics.parent.name] = {label: _value(metrics, f, c) for label, f, c, _ in RECON_COLUMNS}
        rows[metrics.parent.name]["files"] = len(_column(metrics / "spectral.csv", "sdr")) \
            if (metrics / "spectral.csv").exists() else 0
    return pd.DataFrame.from_dict(rows, orient="index")


def stereo_table(root: Path) -> pd.DataFrame:
    rows = {}
    for ms in sorted(root.glob("*/metrics/ms_metrics.csv")):
        with open(ms, newline="") as f:
            data = list(csv.DictReader(f))
        stereo = [r for r in data if r.get("ref_mono", "0") in ("0", "")]      # a mono reference has no Side
        rows[ms.parent.parent.name] = {
            label: float(np.nanmean([float(r[c]) for r in stereo])) if stereo else float("nan")
            for label, c in STEREO_COLUMNS}
    return pd.DataFrame.from_dict(rows, orient="index")


def _main_score(path: Path):
    d = json.loads(path.read_text())
    for sv in d.get("scores", {}).values():
        for s in (sv if isinstance(sv, list) else [sv]):
            if isinstance(s, dict) and "main_score" in s:
                return float(s["main_score"])
    return None


def maeb_scores(root: Path) -> dict[str, float]:
    """Task → main score, from the per-task JSONs mteb writes under <model name>/<revision>/."""
    models = {p.parent.parent.name for p in root.glob("*/*/*.json")}
    if len(models) > 1:
        print(f"warning: {root} holds several models {sorted(models)}; their tasks are merged", file=sys.stderr)
    scores = {}
    for jp in root.glob("*/*/*.json"):
        if jp.stem not in ("summary", "model_meta"):
            score = _main_score(jp)
            if score is not None:
                scores[jp.stem] = score
    return scores


def _markdown(df: pd.DataFrame, higher: dict[str, bool] | None = None, exclude: tuple = ()) -> str:
    """Markdown table, best value per column in bold (rows in `exclude` are never bolded)."""
    out = df.copy().astype(object)
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        ranked = vals[[i not in exclude for i in df.index]]
        best = None
        if higher is not None and col in higher and ranked.notna().any():
            best = ranked.max() if higher[col] else ranked.min()
        out[col] = [("—" if pd.isna(v) else f"**{v:.3f}**" if v == best and i not in exclude else
                     f"{v:.0f}" if col == "files" else f"{v:.3f}") for i, v in zip(df.index, vals)]
    header = ["", *map(str, out.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join([str(idx), *map(str, row)]) + " |" for idx, row in zip(out.index, out.values)]
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Print the paper's tables from evaluation outputs.")
    p.add_argument("--recon", action="append", default=[], metavar="LABEL=DIR")
    p.add_argument("--maeb", action="append", default=[], type=Path, metavar="DIR")
    args = p.parse_args(argv)

    higher = {label: h for label, _, _, h in RECON_COLUMNS}
    for spec in args.recon:
        label, _, directory = spec.partition("=")
        root = Path(directory or label)
        print(f"\n### Reconstruction — {label} (Table 2)\n\n{_markdown(recon_table(root), higher)}")
        stereo = stereo_table(root)
        if not stereo.empty:
            print(f"\n### Stereo image — {label}\n\n"
                  f"{_markdown(stereo, {'d_width': False, 'SI-SDR side': True, 'SI-SDR mid': True})}")

    if args.maeb:
        scores = {d.name: maeb_scores(d) for d in args.maeb}
        tasks = [t for block in MAEB_BLOCKS.values() for t in block]
        per_task = pd.DataFrame({m: [s.get(t, np.nan) for t in tasks] for m, s in scores.items()}, index=tasks)
        blocks = pd.DataFrame({m: [np.mean([s[t] for t in ts]) if all(t in s for t in ts) else np.nan
                                   for ts in MAEB_BLOCKS.values()] for m, s in scores.items()},
                              index=list(MAEB_BLOCKS))
        oracle = ("clap",)
        print(f"\n### Semantic probing, nineteen tasks (Table 9)\n\n"
              f"{_markdown(per_task.T, {t: True for t in tasks}, exclude=oracle)}")
        print(f"\n### Average per block (Table 4)\n\n{_markdown(blocks.T, {b: True for b in MAEB_BLOCKS}, exclude=oracle)}")


if __name__ == "__main__":
    main()
