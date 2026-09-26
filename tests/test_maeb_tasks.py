# =============================================================================
# tests/test_maeb_tasks.py
# Unit tests for evaluation/maeb/tasks.py — audio-only filtering, task selection
# and the 19-task suite of the paper. Uses lightweight fake tasks (no mteb import needed).
# =============================================================================
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from maeb.tasks import (  # noqa: E402
    ALLOWED_TASKS,
    FMA_SUITE,
    MAEB_ORIGINAL_MUSIC,
    MOISESDB_SUITE,
    ensure_audio_only_tasks,
    get_tasks_by_name,
    is_audio_only_task,
    select_task_names,
)


def _task(name, modalities):
    return SimpleNamespace(metadata=SimpleNamespace(name=name, modalities=modalities))


def test_is_audio_only_true():
    assert is_audio_only_task(_task("A", ["audio"]))


def test_is_audio_only_false_for_crossmodal():
    assert not is_audio_only_task(_task("B", ["audio", "text"]))


def test_is_audio_only_false_for_empty_modalities():
    assert not is_audio_only_task(_task("C", []))


def test_ensure_audio_only_passes_through():
    tasks = [_task("A", ["audio"]), _task("B", ["audio"])]
    assert ensure_audio_only_tasks(tasks, source="x") == tasks


def test_ensure_audio_only_raises_on_crossmodal():
    tasks = [_task("A", ["audio"]), _task("B", ["audio", "text"])]
    with pytest.raises(ValueError, match="audio-only"):
        ensure_audio_only_tasks(tasks, source="x", encoder_label="SwinEncoder")


def test_select_default_is_the_fma_suite():
    assert select_task_names(None) == FMA_SUITE


def test_select_explicit_subset_wins():
    assert select_task_names(["GTZANGenre"], moisesdb_only=True) == ["GTZANGenre"]


def test_select_flags():
    assert select_task_names(None, maeb_original_music_only=True) == MAEB_ORIGINAL_MUSIC
    assert select_task_names(None, moisesdb_only=True) == MOISESDB_SUITE
    assert select_task_names(None, with_moisesdb=True) == FMA_SUITE + MOISESDB_SUITE


def test_all_19_paper_tasks_are_allowed():
    assert ALLOWED_TASKS == FMA_SUITE + MAEB_ORIGINAL_MUSIC + MOISESDB_SUITE
    assert len(set(ALLOWED_TASKS)) == 19


def test_unknown_task_name_raises():
    with pytest.raises(ValueError, match="Unsupported task"):
        get_tasks_by_name(["NotAMAEBTask"])
