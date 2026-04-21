#!/usr/bin/env python3
# ===============================================================================
# eval_sweep.py
# Evaluate every .ckpt under checkpoints/ on FMA-small test split.
# Runs the full metric pipeline per checkpoint via compute_all.py, then
# aggregates all mean scores into runs/eval/summary.csv (one row per run).
#
# Usage:
#   uv run python eval_sweep.py
#   uv run python eval_sweep.py --skip-cdpam --skip-fad-gudgud --max-files 50
#   uv run python eval_sweep.py --checkpoints-dir checkpoints/FMA_autoencoder_KL
# ===============================================================================
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PATH, RUNS_DIR, DEFAULT_DEVICE
from ar_spectra.utils.console import ok, warn, err, info

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _csv_means(path: Path, columns: list[str]) -> dict[str, float | None]:
    """Return mean of each column in a per-file CSV. Missing columns → None."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[str, float | None] = {}
    for col in columns:
        vals = [float(r[col]) for r in rows if col in r and r[col].strip()]
        result[col] = (sum(vals) / len(vals)) if vals else None
    return result


def _parse_fad(stdout: str) -> dict[str, float]:
    """Extract FAD scalar values from captured (plain-text) stdout."""
    metrics: dict[str, float] = {}
    # gudgud96: "FAD gudgud (clap-audio): 8.901"
    for m in re.finditer(r"FAD gudgud \(([\w-]+)\):\s*([\d.eE+\-]+)", stdout):
        try:
            metrics[f"fad_gudgud_{m.group(1)}"] = float(m.group(2))
        except ValueError:
            pass
    # fadtk:   "FAD (mert): 12.345"
    for m in re.finditer(r"FAD \(([\w-]+)\):\s*([\d.eE+\-]+)", stdout):
        try:
            metrics[f"fad_fadtk_{m.group(1)}"] = float(m.group(2))
        except ValueError:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def _run_one(
    ckpt_path: Path,
    target_dir: str,
    eval_root: Path,
    pipeline_flags: list[str],
    device: str,
) -> dict:
    """Run the full pipeline for one checkpoint and return a metrics dict."""
    stem = ckpt_path.stem
    run_dir = eval_root / stem
    metrics_dir = run_dir / "metrics"
    ckpt_cache = run_dir / "cache"
    ckpt_cache.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "compute_all.py"),
        "--checkpoint", str(ckpt_path),
        "--target-dir", target_dir,
        "--output-dir", str(run_dir),
        "--csv-dir", str(metrics_dir),
        "--infer-device", device,
        "--cache-dir", str(ckpt_cache),
    ] + pipeline_flags

    import pty

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "1"}

    info(f"Running pipeline for: {stem}")
    
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(cmd, stdout=slave_fd, stderr=subprocess.STDOUT, close_fds=True, env=env)
    os.close(slave_fd)

    output_bytes = bytearray()
    try:
        while True:
            chunk = os.read(master_fd, 4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output_bytes.extend(chunk)
    except OSError:
        pass # EIO exception when slave FD closes

    proc.wait()
    os.close(master_fd)
    
    stdout_clean = _strip_ansi(output_bytes.decode("utf-8", errors="replace"))

    if proc.returncode != 0:
        err(f"[{stem}] Pipeline exited with code {proc.returncode} — partial metrics may be missing.")

    return _collect_metrics(metrics_dir, stem, ckpt_path, proc.returncode, stdout_clean)


def _collect_metrics(metrics_dir: Path, stem: str, ckpt_path: Path, exit_code: int = 0, stdout_clean: str = "") -> dict:
    """Collect all computed metrics from CSV files and stdout into a single record."""
    record: dict = {
        "checkpoint": stem,
        "checkpoint_path": str(ckpt_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exit_code": exit_code,
    }

    # Spectral (si_sdr, stft_loss)
    spec = _csv_means(metrics_dir / "spectral.csv", ["si_sdr", "stft_loss"])
    record["si_sdr_mean"] = spec.get("si_sdr")
    record["stft_loss_mean"] = spec.get("stft_loss")
    # n_files from spectral CSV (most files processed)
    if (metrics_dir / "spectral.csv").exists():
        try:
            with open(metrics_dir / "spectral.csv") as f:
                record["n_files"] = sum(1 for _ in csv.DictReader(f))
        except Exception:
            pass

    # CLAP cosine
    clap = _csv_means(metrics_dir / "clap_score.csv", ["clap_music", "clap_audio"])
    record["clap_music_mean"] = clap.get("clap_music")
    record["clap_audio_mean"] = clap.get("clap_audio")

    # CDPAM
    cdpam = _csv_means(metrics_dir / "cdpam.csv", ["cdpam"])
    record["cdpam_mean"] = cdpam.get("cdpam")

    # FAD scores: prioritize CSV files, fallback to stdout parsing
    fad_files = {
        "fad_mert.csv": "fad_fadtk_mert",
        "fad_gudgud.csv": "fad_gudgud_clap-audio"
    }
    for fname, key in fad_files.items():
        fpath = metrics_dir / fname
        if fpath.exists():
            try:
                with open(fpath, "r") as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    record[key] = float(row["score"])
            except Exception:
                pass

    # Fallback to stdout if CSVs weren't found or failed
    if stdout_clean:
        fad_stdout = _parse_fad(stdout_clean)
        for k, v in fad_stdout.items():
            if k not in record or record[k] is None or (isinstance(record[k], float) and np.isnan(record[k])):
                record[k] = v

    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(records: list[dict], out_path: Path) -> None:
    """Upsert rows into summary.csv — safe for concurrent parallel jobs (fcntl lock)."""
    if not records:
        return
    import fcntl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Read existing rows
            existing: dict[str, dict] = {}
            existing_keys: list[str] = []
            if out_path.exists():
                with open(out_path, newline="") as f:
                    reader = csv.DictReader(f)
                    existing_keys = list(reader.fieldnames or [])
                    for row in reader:
                        existing[row["checkpoint"]] = dict(row)

            # Upsert: overwrite matching checkpoint rows, append new ones
            for rec in records:
                existing[rec["checkpoint"]] = rec

            # Union all keys
            all_keys: list[str] = list(existing_keys)
            seen: set[str] = set(all_keys)
            for rec in existing.values():
                for k in rec:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing.values())
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    ok(f"Summary → {out_path}  ({len(existing)} checkpoint(s))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep all .ckpt files under checkpoints/ and evaluate on FMA-small test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints-dir", type=Path, default=PROJECT_ROOT / "checkpoints",
                   help="Directory to search recursively for .ckpt files.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a single .ckpt file. Overrides --checkpoints-dir.")
    p.add_argument("--target-dir", type=str, default=str(DATA_PATH),
                   help="Reference audio directory (FMA-small test split).")
    p.add_argument("--eval-root", type=Path, default=RUNS_DIR / "eval",
                   help="Root for per-checkpoint output dirs.")
    p.add_argument("--summary-csv", type=Path, default=None,
                   help="Output path for aggregated summary CSV (default: --eval-root/summary.csv).")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                   help="Inference device.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Max audio files per checkpoint (0 = all).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for all evaluation steps.")
    p.add_argument("--metrics-env", type=str, default=None,
                   help="Path to a separate virtualenv for FAD/CDPAM if dependencies conflict.")
    p.add_argument("--resume", action="store_true",
                   help="Skip checkpoints that already have metric CSVs in --eval-root.")
    # Pass-through flags
    p.add_argument("--skip-cdpam", action="store_true")
    p.add_argument("--skip-fad", action="store_true")
    p.add_argument("--skip-clap", action="store_true")
    p.add_argument("--skip-fad-gudgud", action="store_true")
    p.add_argument("--clap-model", default="both", choices=["music", "audio", "both"])
    p.add_argument("--fad-model", default="mert",
                   choices=["vggish", "clap-laion", "clap-laion-audio", "mert"])
    p.add_argument("--fad-gudgud-model", default="clap-audio",
                   choices=["vggish", "pann", "clap-music", "clap-audio", "encodec"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary_csv = args.summary_csv or (args.eval_root / "summary.csv")

    # Discover checkpoints
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            err(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = sorted(args.checkpoints_dir.rglob("*.ckpt"))
    if not ckpt_paths:
        err(f"No .ckpt files found under {args.checkpoints_dir}")
        sys.exit(1)

    info(f"Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        try:
            info(f"  {p.relative_to(PROJECT_ROOT)}")
        except ValueError:
            info(f"  {p}")

    if not args.target_dir or not Path(args.target_dir).is_dir():
        err(f"Target directory not found: {args.target_dir!r} — set DATA_PATH in .env or pass --target-dir.")
        sys.exit(1)

    # Build pass-through flags for compute_all.py
    flags: list[str] = [
        "--clap-model", args.clap_model,
        "--fad-model", args.fad_model,
        "--fad-gudgud-model", args.fad_gudgud_model
    ]
    if args.metrics_env:    flags += ["--metrics-env", args.metrics_env]
    if args.skip_cdpam:     flags.append("--skip-cdpam")
    if args.skip_fad:       flags.append("--skip-fad")
    if args.skip_clap:      flags.append("--skip-clap")
    if args.skip_fad_gudgud: flags.append("--skip-fad-gudgud")
    if args.batch_size:     flags += ["--batch-size", str(args.batch_size)]
    if args.max_files > 0:  flags += ["--max-files", str(args.max_files)]

    records: list[dict] = []
    for i, ckpt in enumerate(tqdm(ckpt_paths, desc="Checkpoints", unit="ckpt"), 1):
        info(f"\n{'=' * 60}")
        info(f"Checkpoint {i}/{len(ckpt_paths)}: {ckpt.stem}")
        info(f"{'=' * 60}")

        # --resume: skip if metrics already exist for this checkpoint
        if args.resume and (args.eval_root / ckpt.stem / "metrics" / "spectral.csv").exists():
            warn(f"[{ckpt.stem}] Already evaluated (--resume), collecting existing metrics.")
            rec = _collect_metrics(args.eval_root / ckpt.stem / "metrics", ckpt.stem, ckpt)
            records.append(rec)
            continue

        rec = _run_one(ckpt, args.target_dir, args.eval_root, flags, args.device)
        records.append(rec)
        ok(f"Checkpoint {i}/{len(ckpt_paths)} done: {ckpt.stem}")

        # Incremental save — never lose completed results on Ctrl+C
        _write_summary(records, summary_csv)


if __name__ == "__main__":
    main()
