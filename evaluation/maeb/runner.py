#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/runner.py
# Run a list of MAEB tasks through MTEB one at a time, with ETA / resume /
# per-task num_proc patching, then write summary.json. Model-agnostic.
# =============================================================================
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm as _tqdm

from .compatibility import patch_num_proc

log = logging.getLogger(__name__)


def _configure_classification_solver() -> None:
    """Raise the LogisticRegression iteration cap (MTEB default 100 under-converges)."""
    from mteb.abstasks.classification import AbsTaskClassification
    from sklearn.linear_model import LogisticRegression
    AbsTaskClassification.evaluator_model = LogisticRegression(max_iter=1000)


def _extract_main_score(scores: dict) -> float | None:
    """Pull the primary metric out of an MTEB result's per-split scores."""
    for split_scores in scores.values():
        if isinstance(split_scores, list):
            for s in split_scores:
                if "main_score" in s:
                    return s["main_score"]
        elif isinstance(split_scores, dict) and "main_score" in split_scores:
            return split_scores["main_score"]
    return None


def run_maeb(
    model: Any,
    tasks: list[Any],
    output_dir: Path,
    *,
    encode_kwargs: dict[str, Any],
    overwrite: bool = False,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Evaluate ``tasks`` with ``model``, saving per-task results + summary.json.

    Args:
        model:         MTEB-compatible encoder (AbsEncoder subclass) with mteb_model_meta.
        tasks:         Resolved MTEB task objects (already audio-only checked).
        output_dir:    Where MTEB writes per-task JSON and summary.json.
        encode_kwargs: Passed to MTEB.run (e.g. {"batch_size": 16}).
        overwrite:     Re-run tasks even if results exist on disk.
        summary_extra: Extra key/values stored at the top of summary.json.

    Returns:
        Mapping task_name -> main_score (or None).
    """
    import mteb
    import torch

    _configure_classification_solver()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = len(tasks)
    log.info(
        "Starting MTEB evaluation → %s (%d tasks). "
        "Results saved per task — interrupt and re-run to resume (--overwrite to restart).",
        output_dir, n_tasks,
    )
    results: list[Any] = []
    start_wall = time.perf_counter()

    bar = _tqdm(
        total=n_tasks,
        desc="MAEB [starting]",
        unit="task",
        file=sys.stdout,
        ascii=True,
        ncols=100,
        mininterval=0.0,
        miniters=1,
        leave=True,
    )

    failed: list[str] = []

    for idx, task in enumerate(tasks):
        task_start = time.perf_counter()
        name = task.metadata.name
        bar.set_description(f"MAEB [{name[:35]}]")
        log.info("\n%s Task %d/%d: %s %s", "=" * 60, idx + 1, n_tasks, name, "=" * 60)
        patch_num_proc(task)
        evaluation = mteb.MTEB(tasks=[task])
        try:
            task_results = evaluation.run(
                model,
                output_folder=str(output_dir),
                encode_kwargs=encode_kwargs,
                overwrite_results=overwrite,
            )
            results.extend(task_results or [])
        except Exception:
            log.exception("Task %s FAILED — skipping, will not appear in summary.", name)
            failed.append(name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        done = idx + 1
        elapsed_total = time.perf_counter() - start_wall
        task_min = (time.perf_counter() - task_start) / 60
        if done < n_tasks:
            remaining_min = (elapsed_total / done) * (n_tasks - done) / 60
            bar.set_postfix(elapsed=f"{elapsed_total/60:.0f}m", eta=f"{remaining_min:.0f}m")
            log.info(
                "Task %d/%d done in %.1f min. Elapsed: %.1f min. ~%.0f min remaining.",
                done, n_tasks, task_min, elapsed_total / 60, remaining_min,
            )
        else:
            bar.set_postfix(elapsed=f"{elapsed_total/60:.0f}m", eta="0m")
        bar.update(1)

    bar.close()
    if failed:
        log.warning("%d task(s) failed and were skipped: %s", len(failed), failed)
    log.info("All %d tasks completed in %.1f min.",
             n_tasks, (time.perf_counter() - start_wall) / 60)

    # Summary
    log.info("\n=== MAEB Results Summary ===")
    summary: dict[str, float | None] = {}
    for res in results:
        score = _extract_main_score(res.scores)
        summary[res.task_name] = score
        if score is not None:
            log.info("  %-55s: %.4f", res.task_name, score)
        else:
            log.info("  %s: N/A", res.task_name)

    # Write summary inside the model's MTEB subdir (safe for parallel runs with
    # different models writing to the same output_dir root).
    meta = getattr(model, "mteb_model_meta", None)
    if meta is not None:
        model_subdir = meta.name.replace("/", "__")
        revision = getattr(meta, "revision", None) or "local"
        summary_path = output_dir / model_subdir / revision / "summary.json"
    else:
        summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {**(summary_extra or {}), "results": summary}
    if failed:
        payload["failed_tasks"] = failed
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Summary saved to: %s", summary_path)
    return summary
