#!/usr/bin/env python3
# =============================================================================
# tests/test_moisesdb_tasks.py
# Unit tests for evaluation/maeb/moisesdb_tasks.py — data loading from
# chunks_30s/, mixture/stem selection, instrument class whitelist, and task
# construction.
#
# Needs the MoisesDB dataset. Run with:  pytest tests/test_moisesdb_tasks.py -m weights -v
# =============================================================================
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

from evaluation.maeb.moisesdb_tasks import (
    _load_chunks_df,
    _load_track_metadata,
    _get_mixtures,
    _get_stems,
    _genre_filtered_mixtures,
    _GENRE_MIN_TRACKS,
    _INSTRUMENT_CLASSES,
    MOISESDB_TASK_REGISTRY,
    MOISESDB_SUITE,
    get_moisesdb_tasks,
)
from evaluation.maeb import moisesdb_tasks
from evaluation.maeb.moisesdb_tasks import _chunks_30s_root, _moisesdb_metadata_root

# The entry point sets these from configs/paths; here the same environment variables.
moisesdb_tasks.MOISESDB_ROOT = os.environ.get("MOISESDB_ROOT")
moisesdb_tasks.MOISESDB_CHUNKS = os.environ.get("MOISESDB_CHUNKS_ROOT")


def _moisesdb_available() -> bool:
    try:
        _chunks_30s_root(), _moisesdb_metadata_root()
    except FileNotFoundError:
        return False
    return True


# Needs the MoisesDB dataset (MOISESDB_ROOT + MOISESDB_CHUNKS_ROOT): opt-in with `-m weights`.
pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(not _moisesdb_available(), reason="MoisesDB not found (set MOISESDB_ROOT and MOISESDB_CHUNKS_ROOT)"),
]


# ===========================================================================
# Data loading tests — chunks_30s/ flat directory
# ===========================================================================

class TestLoadChunks:
    """Tests for _load_chunks_df() — flat directory scanner."""

    @pytest.fixture(autouse=True)
    def _load(self):
        """Load the DataFrame once for all tests in this class."""
        self.df = _load_chunks_df()

    def test_total_file_count(self):
        """Should have 1611 total files (240 mixtures + 1371 stems)."""
        assert len(self.df) == 1611, f"Expected 1611 files, got {len(self.df)}"

    def test_track_count(self):
        """Should have 240 unique tracks."""
        n_tracks = self.df["track_id"].nunique()
        assert n_tracks == 240, f"Expected 240 tracks, got {n_tracks}"

    def test_mixture_count(self):
        """Should have exactly 240 mixtures (one per track)."""
        n_mix = int(self.df["is_mixture"].sum())
        assert n_mix == 240, f"Expected 240 mixtures, got {n_mix}"

    def test_stem_count(self):
        """Should have 1371 stems (non-mixture files)."""
        n_stems = int((~self.df["is_mixture"]).sum())
        assert n_stems == 1371, f"Expected 1371 stems, got {n_stems}"

    def test_required_columns(self):
        """DataFrame must have the expected columns."""
        expected = {"track_id", "artist", "genre", "song", "stem_name", "is_mixture", "_path"}
        assert expected.issubset(set(self.df.columns)), \
            f"Missing columns: {expected - set(self.df.columns)}"

    def test_all_paths_exist(self):
        """All _path values should point to existing files."""
        missing = [p for p in self.df["_path"] if not Path(p).exists()]
        assert len(missing) == 0, f"{len(missing)} paths missing, e.g.: {missing[:3]}"

    def test_uuid_format(self):
        """All track_ids should be 36-char UUIDs."""
        bad = [tid for tid in self.df["track_id"].unique() if len(tid) != 36]
        assert len(bad) == 0, f"Non-UUID track_ids: {bad[:5]}"

    def test_no_unknown_metadata(self):
        """Most tracks should have known (non-'unknown') artist and genre."""
        n_unknown_genre = int((self.df.groupby("track_id")["genre"].first() == "unknown").sum())
        assert n_unknown_genre == 0, f"{n_unknown_genre} tracks have unknown genre"


# ===========================================================================
# Track metadata tests
# ===========================================================================

class TestTrackMetadata:
    """Tests for _load_track_metadata() — data.json parsing."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.meta = _load_track_metadata()

    def test_track_count(self):
        """Should have 240 tracks in metadata."""
        assert len(self.meta) == 240

    def test_keys_are_uuids(self):
        """All keys should be 36-char UUIDs."""
        bad = [k for k in self.meta if len(k) != 36]
        assert len(bad) == 0

    def test_metadata_fields(self):
        """Each entry should have artist, genre, song."""
        for tid, meta in self.meta.items():
            assert "artist" in meta, f"Track {tid} missing artist"
            assert "genre" in meta, f"Track {tid} missing genre"
            assert "song" in meta, f"Track {tid} missing song"


# ===========================================================================
# Mixture / Stem selection tests
# ===========================================================================

class TestMixtureVsStem:
    """Tests for _get_mixtures() and _get_stems()."""

    def test_mixtures_count(self):
        """_get_mixtures() should return exactly 240 rows."""
        mix = _get_mixtures()
        assert len(mix) == 240

    def test_mixtures_one_per_track(self):
        """Each track should appear exactly once in mixtures."""
        mix = _get_mixtures()
        assert mix["track_id"].is_unique

    def test_mixtures_all_are_mixtures(self):
        """All rows in _get_mixtures() should have stem_name == 'mixture'."""
        mix = _get_mixtures()
        assert (mix["stem_name"] == "mixture").all()

    def test_stems_count(self):
        """_get_stems() should return 1371 rows."""
        stems = _get_stems()
        assert len(stems) == 1371

    def test_stems_no_mixtures(self):
        """No row in _get_stems() should have stem_name == 'mixture'."""
        stems = _get_stems()
        assert not stems["is_mixture"].any()

    def test_mixtures_plus_stems_equals_total(self):
        """Mixtures + stems should account for all files."""
        df = _load_chunks_df()
        assert len(_get_mixtures(df)) + len(_get_stems(df)) == len(df)


# ===========================================================================
# Instrument class whitelist tests
# ===========================================================================

class TestInstrumentFilter:
    """Tests for the _INSTRUMENT_CLASSES whitelist."""

    def test_whitelist_has_7_classes(self):
        """Should include exactly 7 instrument classes."""
        assert len(_INSTRUMENT_CLASSES) == 7

    def test_whitelist_contents(self):
        """Should include the expected classes."""
        expected = {"vocals", "drums", "bass", "guitar", "piano", "percussion", "other_keys"}
        assert _INSTRUMENT_CLASSES == expected

    def test_excluded_classes_not_in_whitelist(self):
        """other, other_plucked, bowed_strings, wind must be excluded."""
        excluded = {"other", "other_plucked", "bowed_strings", "wind"}
        assert excluded.isdisjoint(_INSTRUMENT_CLASSES)

    def test_filtered_stems_count(self):
        """After filtering, should keep stems only in the 7 whitelist classes."""
        stems = _get_stems()
        filtered = stems[stems["stem_name"].isin(_INSTRUMENT_CLASSES)]
        actual_classes = set(filtered["stem_name"].unique())
        assert actual_classes == _INSTRUMENT_CLASSES, \
            f"Expected {_INSTRUMENT_CLASSES}, got {actual_classes}"

    def test_filtered_stems_no_excluded(self):
        """Filtered stems should not contain any excluded class."""
        stems = _get_stems()
        filtered = stems[stems["stem_name"].isin(_INSTRUMENT_CLASSES)]
        excluded = {"other", "other_plucked", "bowed_strings", "wind", "mixture"}
        remaining = set(filtered["stem_name"].unique())
        assert remaining.isdisjoint(excluded)


# ===========================================================================
# Genre filtering tests
# ===========================================================================

class TestGenreFilter:
    """Tests for genre filtering thresholds."""

    def test_genre_threshold(self):
        """After filtering, should keep exactly 6 genres."""
        filtered = _genre_filtered_mixtures()
        n_genres = filtered["genre"].nunique()
        assert n_genres == 6, f"Expected 6 genres after threshold, got {n_genres}"

    def test_genre_threshold_excludes_small(self):
        """jazz and bossa_nova (1 track each) must be excluded."""
        filtered = _genre_filtered_mixtures()
        genres = set(filtered["genre"].unique())
        assert "jazz" not in genres, "jazz should be excluded (1 track)"
        assert "bossa_nova" not in genres, "bossa_nova should be excluded (1 track)"

    def test_genre_threshold_min_tracks(self):
        """All surviving genres should have ≥ _GENRE_MIN_TRACKS tracks."""
        filtered = _genre_filtered_mixtures()
        counts = filtered.groupby("genre").size()
        assert (counts >= _GENRE_MIN_TRACKS).all()

    def test_genre_distribution(self):
        """rock should be the most common genre (≥100 tracks)."""
        mix = _get_mixtures()
        genre_counts = mix["genre"].value_counts()
        assert genre_counts.iloc[0] >= 100, "rock should have ≥100 tracks"
        assert genre_counts.index[0] == "rock", \
            f"Most common genre should be 'rock', got {genre_counts.index[0]}"


# ===========================================================================
# Task construction tests (verify datasets are non-empty and well-formed)
# ===========================================================================

class TestTaskConstruction:
    """Tests that each task can be instantiated and produces valid datasets."""

    def test_registry_has_7_tasks(self):
        """Registry should contain exactly 7 tasks."""
        assert len(MOISESDB_TASK_REGISTRY) == 7

    def test_suite_matches_registry(self):
        """MOISESDB_SUITE should list all registry keys."""
        assert set(MOISESDB_SUITE) == set(MOISESDB_TASK_REGISTRY.keys())

    def test_get_moisesdb_tasks_all(self):
        """get_moisesdb_tasks() with no names should return all 7."""
        tasks = get_moisesdb_tasks(max_files=50)
        assert len(tasks) == 7

    @pytest.mark.parametrize("task_name", [
        "MoisesDBGenreClassification",
        "MoisesDBGenreClustering",
        "MoisesDBArtistClustering",
    ])
    def test_dataset_task_load_data(self, task_name):
        """Dataset-based tasks should produce non-empty self.dataset after load_data()."""
        task = MOISESDB_TASK_REGISTRY[task_name](max_files=50)
        task.load_data()
        assert task.data_loaded
        assert task.dataset is not None
        # Should have at least one split with rows
        for split_name, split_ds in task.dataset.items():
            assert len(split_ds) > 0, f"{task_name}/{split_name} is empty"

    def test_retrieval_task_load_data(self):
        """Retrieval task should produce non-empty corpus/queries/relevant_docs."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBArtistA2ARetrieval"](max_files=50)
        task.load_data()
        assert task.data_loaded
        assert len(task.corpus["test"]) > 0
        assert len(task.queries["test"]) > 0
        assert len(task.relevant_docs["test"]) > 0

    def test_reranking_task_load_data(self):
        """Reranking task should produce non-empty corpus/queries/relevant_docs/top_ranked."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBGenreAudioReranking"](max_files=50)
        task.load_data()
        assert task.data_loaded
        assert len(task.corpus["test"]) > 0
        assert len(task.queries["test"]) > 0
        assert len(task.relevant_docs["test"]) > 0
        assert len(task.top_ranked["test"]) > 0

    def test_pair_task_load_data(self):
        """Pair classification task should produce balanced pos/neg pairs."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBArtistPairClassification"](max_files=50)
        task.load_data()
        assert task.data_loaded
        ds = task.dataset["test"]
        assert len(ds) > 0
        labels = ds["label"]
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        assert n_pos > 0, "No positive pairs"
        assert n_neg > 0, "No negative pairs"

    def test_instrument_task_load_data(self):
        """Instrument classification should have ≥10 samples and store group track IDs."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBInstrumentClassification"](max_files=100)
        task.load_data()
        assert task.data_loaded
        ds = task.dataset["train"]
        assert len(ds) >= 10
        # GroupKFold needs _group_track_ids aligned with dataset
        assert hasattr(task, "_group_track_ids")
        assert len(task._group_track_ids) == len(ds)

    def test_instrument_task_uses_stems_not_mixtures(self):
        """Instrument classification should only use stem files (no mixture.wav)."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBInstrumentClassification"](max_files=100)
        task.load_data()
        ds = task.dataset["train"]
        for item in ds["audio"]:
            path = item["path"]
            assert "_mixture.wav" not in path, f"Mixture file found in instrument task: {path}"


# ===========================================================================
# Anti-leakage tests
# ===========================================================================

class TestAntiLeakage:
    """Verify anti-leakage invariants."""

    def test_genre_cls_no_track_leakage(self):
        """Genre classification uses mixtures: one per track, no duplicates."""
        mix = _get_mixtures()
        assert mix["track_id"].is_unique

    def test_instrument_cls_group_ids_match_tracks(self):
        """Instrument classification: group IDs should be track UUIDs from the dataset."""
        task = MOISESDB_TASK_REGISTRY["MoisesDBInstrumentClassification"](max_files=100)
        task.load_data()
        # Each group ID should be a valid track UUID
        df = _load_chunks_df()
        valid_tracks = set(df["track_id"])
        for gid in task._group_track_ids:
            assert gid in valid_tracks, f"Invalid group track_id: {gid}"


# ===========================================================================
# Caching tests (lru_cache + data immutability)
# ===========================================================================

class TestCaching:
    """Tests for lru_cache correctness — repeated calls return the same object."""

    def test_load_chunks_df_cached(self):
        """Two calls to _load_chunks_df() should return the same object (identity)."""
        df1 = _load_chunks_df()
        df2 = _load_chunks_df()
        assert df1 is df2, "lru_cache should return the exact same object"

    def test_load_track_metadata_cached(self):
        """Two calls to _load_track_metadata() should return the same object."""
        m1 = _load_track_metadata()
        m2 = _load_track_metadata()
        assert m1 is m2, "lru_cache should return the exact same object"

    def test_deterministic_task_data(self):
        """Two constructions of the same task should produce identical paths."""
        task1 = MOISESDB_TASK_REGISTRY["MoisesDBGenreClassification"](max_files=50)
        task1.load_data()
        paths1 = list(task1.dataset["train"]["audio"])

        task2 = MOISESDB_TASK_REGISTRY["MoisesDBGenreClassification"](max_files=50)
        task2.load_data()
        paths2 = list(task2.dataset["train"]["audio"])

        assert paths1 == paths2, "Task data should be deterministic across constructions"
