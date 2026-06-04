#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/tasks.py
# Task resolution for the FMA-local MAEB suite: the 6 FMA music tasks plus one
# upstream music pair task (NMSQA). Audio-only filtering + name resolution.
# =============================================================================
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# FMA-local music-semantic suite (defined in fma_tasks.py) + one upstream music
# pair task. These run entirely on the FMA large test set (NMSQA from the hub).
FMA_SUITE = [
    "FMAGenreClassification",
    "FMAGenreClustering",
    "FMAArtistClustering",
    "FMAArtistA2ARetrieval",
    "FMAGenreAudioReranking",
    "FMAArtistPairClassification",
]
FMA_SUITE_EXTRA = ["NMSQAPairClassification"]   # upstream mteb music-similarity task

# The complete set of task names these CLIs may run.
ALLOWED_TASKS = FMA_SUITE + FMA_SUITE_EXTRA


def is_audio_only_task(task: Any) -> bool:
    """True if the task declares modalities and every one of them is 'audio'."""
    return bool(task.metadata.modalities) and all(
        modality == "audio" for modality in task.metadata.modalities
    )


def ensure_audio_only_tasks(
    tasks: list[Any], *, source: str, encoder_label: str = "This encoder"
) -> list[Any]:
    """Return tasks unchanged, or raise if any is not audio-only."""
    unsupported = [t for t in tasks if not is_audio_only_task(t)]
    if unsupported:
        preview = ", ".join(
            f"{t.metadata.name} ({'/'.join(t.metadata.modalities)})"
            for t in unsupported[:8]
        )
        if len(unsupported) > 8:
            preview += f", ... (+{len(unsupported) - 8} more)"
        raise ValueError(
            f"{encoder_label} only supports audio-only tasks. "
            f"Selection from {source} contains unsupported tasks: {preview}"
        )
    return tasks


def get_tasks_by_name(
    names: list[str], *, encoder_label: str = "This encoder", max_files: int = 0
) -> list[Any]:
    """Resolve task names into objects, restricted to the allowed FMA suite.

    FMA-local names are instantiated from ``fma_tasks.FMA_TASK_REGISTRY`` (with
    the ``max_files`` cap); ``NMSQAPairClassification`` is loaded from the hub.
    Any name outside :data:`ALLOWED_TASKS` is rejected.

    Args:
        names:         Task names to resolve (must be in ALLOWED_TASKS).
        encoder_label: Label used in the audio-only validation error.
        max_files:     Per-task sample cap for FMA-local tasks (0 = all).

    Returns:
        Audio-only-validated list of task objects.

    Raises:
        ValueError: if any requested name is not in ALLOWED_TASKS.
    """
    from .fma_tasks import FMA_TASK_REGISTRY, get_fma_tasks

    unknown = [n for n in names if n not in ALLOWED_TASKS]
    if unknown:
        raise ValueError(
            f"Unsupported task(s) {unknown}. This harness only runs the FMA suite: "
            f"{ALLOWED_TASKS}."
        )
    fma_names = [n for n in names if n in FMA_TASK_REGISTRY]
    other_names = [n for n in names if n not in FMA_TASK_REGISTRY]   # only NMSQA
    tasks = get_fma_tasks(fma_names, max_files=max_files)
    if other_names:
        import mteb
        tasks += list(mteb.get_tasks(tasks=other_names))   # max_files n/a for hub tasks
    return ensure_audio_only_tasks(tasks, source="task selection", encoder_label=encoder_label)
