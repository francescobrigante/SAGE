#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/tasks.py
# Task resolution for the MAEB suite: the 6 FMA music tasks (default), opt-in
# upstream MTEB music tasks (EXTRA_TASKS), and opt-in MoisesDB tasks.
# Audio-only filtering + name resolution.
# =============================================================================
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# FMA-local music-semantic suite (defined in fma_tasks.py). Default per ogni eval.
FMA_SUITE = [
    "FMAGenreClassification",
    "FMAGenreClustering",
    "FMAArtistClustering",
    "FMAArtistA2ARetrieval",
    "FMAGenreAudioReranking",
    "FMAArtistPairClassification",
]
# Task MTEB musicali upstream (hub, già in cache su $FAST). NON nel default: si
# attivano via flag — `--with-extra` le aggiunge a FMA, `--extra-only` solo queste.
# (NMSQA rimosso: pair task di speech, saturo ~0.49, fuori dominio musicale.)
EXTRA_TASKS = [
    "GTZANGenre",
    "MusicGenreClustering",
]

# MoisesDB-local tasks (chunks_30s). Opt-in via --with-moisesdb / --moisesdb-only.
from .moisesdb_tasks import MOISESDB_SUITE  # noqa: E402

# The complete set of task names these CLIs may run.
ALLOWED_TASKS = FMA_SUITE + EXTRA_TASKS + MOISESDB_SUITE


def select_task_names(
    subset: list[str] | None,
    *,
    with_extra: bool = False,
    extra_only: bool = False,
    with_moisesdb: bool = False,
    moisesdb_only: bool = False,
) -> list[str]:
    """Pick which task names to run, given the opt-in flags.

    Precedence:
    1. An explicit ``subset`` (``--tasks``) always wins.
    2. ``moisesdb_only`` → only MoisesDB tasks.
    3. ``extra_only`` → only EXTRA_TASKS.
    4. Otherwise: FMA suite (default), optionally + extra, optionally + moisesdb.
    """
    if subset:
        return list(subset)
    if moisesdb_only:
        return list(MOISESDB_SUITE)
    if extra_only:
        return list(EXTRA_TASKS)
    result = list(FMA_SUITE)
    if with_extra:
        result += EXTRA_TASKS
    if with_moisesdb:
        result += MOISESDB_SUITE
    return result


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
    """Resolve task names into objects.

    FMA-local names are instantiated from ``fma_tasks.FMA_TASK_REGISTRY``,
    MoisesDB names from ``moisesdb_tasks.MOISESDB_TASK_REGISTRY``, and
    upstream hub tasks (GTZAN etc.) are fetched via ``mteb.get_tasks``.

    Args:
        names:         Task names to resolve (must be in ALLOWED_TASKS).
        encoder_label: Label used in the audio-only validation error.
        max_files:     Per-task sample cap for local tasks (0 = all).

    Returns:
        Audio-only-validated list of task objects.

    Raises:
        ValueError: if any requested name is not in ALLOWED_TASKS.
    """
    from .fma_tasks import FMA_TASK_REGISTRY, get_fma_tasks
    from .moisesdb_tasks import MOISESDB_TASK_REGISTRY, get_moisesdb_tasks

    unknown = [n for n in names if n not in ALLOWED_TASKS]
    if unknown:
        raise ValueError(
            f"Unsupported task(s) {unknown}. Allowed: {ALLOWED_TASKS}."
        )
    fma_names = [n for n in names if n in FMA_TASK_REGISTRY]
    moisesdb_names = [n for n in names if n in MOISESDB_TASK_REGISTRY]
    # Hub tasks: everything not FMA-local and not MoisesDB-local
    other_names = [n for n in names
                   if n not in FMA_TASK_REGISTRY and n not in MOISESDB_TASK_REGISTRY]

    tasks = get_fma_tasks(fma_names, max_files=max_files)
    tasks += get_moisesdb_tasks(moisesdb_names, max_files=max_files)
    if other_names:
        import mteb
        tasks += list(mteb.get_tasks(tasks=other_names))   # max_files n/a for hub tasks
    return ensure_audio_only_tasks(tasks, source="task selection", encoder_label=encoder_label)
