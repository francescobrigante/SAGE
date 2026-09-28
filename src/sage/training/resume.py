"""Resuming a training run where it stopped, as the paper's SLURM scripts did.

A run is identified by its name (``trainer.wandb.name``): its folders live under
``runs/<name>/<date>/`` and its W&B run id in ``.run_ids/<name>``. A SLURM requeue always
continues the newest checkpoint of its run name; relaunching a *chosen* name (a recipe's or
``trainer.wandb.name=``) does too, in the run's own folder and W&B run. A name derived from
the config (no ``trainer.wandb.name``) is not resumed: it only reflects the loss weights, so
two different experiments can share it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_resume_checkpoint(run_root: Path) -> Optional[Path]:
    """Newest checkpoint under ``runs/<name>/*/``: Lightning's save on a SLURM requeue
    (``hpc_ckpt_<n>.ckpt`` in the run folder) or the end-of-epoch ``checkpoints/last.ckpt``."""
    if not run_root.is_dir():
        return None
    candidates = [*run_root.glob("*/hpc_ckpt_*.ckpt"), *run_root.glob("*/checkpoints/last.ckpt")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def is_slurm_requeue() -> bool:
    """True in a job SLURM has requeued (Lightning requeues on the time-limit signal)."""
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT") or 0) > 0
    except ValueError:
        return False


def should_auto_resume(found: Path, *, requeued: bool, chosen_name: bool, init_from: Optional[str]) -> bool:
    """Whether a launch continues ``found``, the newest checkpoint of its run name.

    A requeue always does (same job, same command). Otherwise only a chosen run name is
    resumed, and never silently over a new ``+init_from``: that launch stops and says how to
    either continue the run or start a new one."""
    if requeued:
        return True
    if not chosen_name:
        return False
    if init_from:
        raise SystemExit(
            f"This run name already has a checkpoint ({found}), and +init_from={init_from} asks for a "
            "new start. To continue the run, drop +init_from; to start a new run from init_from, set "
            "another trainer.wandb.name or auto_resume=false.")
    return True


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
