"""Resuming a training run where it stopped, as the paper's SLURM scripts did.

A run is identified by its name (``trainer.wandb.name``): its folders live under
``runs/<name>/<date>/`` and its W&B run id in ``.run_ids/<name>``. Relaunching the
same name (SLURM requeue, crash, manual resubmission) continues the newest checkpoint
in its own folder and logs to the same W&B run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_resume_checkpoint(run_root: Path) -> Optional[Path]:
    """Newest checkpoint under ``runs/<name>/*/``: Lightning's save on a SLURM requeue
    (``hpc_ckpt_<n>.ckpt`` in the run folder) or the end-of-epoch ``checkpoints/last.ckpt``."""
    if not run_root.is_dir():
        return None
    candidates = [*run_root.glob("*/hpc_ckpt_*.ckpt"), *run_root.glob("*/checkpoints/last.ckpt")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def run_dir_of(ckpt: Path) -> Path:
    """The run folder (``runs/<name>/<date>/``) a checkpoint was written in."""
    return ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent


def resume_run_dir(run_root: Path, ckpt_path: Optional[str]) -> Optional[Path]:
    """The folder to continue in: the one of ``ckpt_path`` when it belongs to this run name,
    else None (a new dated folder; e.g. a checkpoint of another run used as a starting point)."""
    if not ckpt_path:
        return None
    ckpt = Path(ckpt_path).expanduser().resolve()
    return run_dir_of(ckpt) if ckpt.is_relative_to(run_root.resolve()) else None


def read_run_id(run_ids_dir: Path, run_name: str) -> Optional[str]:
    """The W&B run id stored for a run name by its first launch, if any."""
    try:
        run_id = (run_ids_dir / run_name).read_text().strip()
    except OSError:
        return None
    return run_id or None
