#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/fma_tasks.py
# MAEB(audio-only) task definitions backed by the LOCAL FMA large test set.
# Music-semantic tasks (genre / artist) that load from tracks.csv + mp3 paths
# on $FAST, bypassing the HuggingFace hub via a custom load_data override.
# =============================================================================
"""FMA-local MAEB tasks.

Six tasks evaluated entirely on the FMA *large test set* (11 263 clips, 30 s,
44.1 kHz stereo, already on ``$FAST``):

* ``FMAGenreClassification``    — 16-class genre (K-fold cross-validation)
* ``FMAGenreClustering``        — 16-class genre clustering
* ``FMAArtistClustering``       — artist-id clustering (734 artists)
* ``FMAArtistA2ARetrieval``     — same-artist audio-to-audio retrieval
* ``FMAGenreAudioReranking``    — per-query candidate reranking by genre
* ``FMAArtistPairClassification`` — same-artist pair classification

Paths come from ``config.py`` (``FMA_METADATA`` → tracks.csv) with the audio
directory derived as ``<fma>/fma_large`` (override via env ``FMA_LARGE_DIR``).
A per-task ``max_files`` cap mirrors ``evaluate_swin_varT.collect_fma_files``:
each task uses ``min(task_samples, max_files)`` (``0`` = all), sampled
deterministically while preserving label / qrel structure.
"""
from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Hierarchical FMA subsets: 'large' is the outermost tier (small ⊂ medium ⊂ large).
_SUBSETS = ("small", "medium", "large")          # full test set (all tiers)
_ARTIST_MIN_TRACKS_RETRIEVAL = 5                 # query pool: artists with ≥5 clips
_ARTIST_MIN_TRACKS_CLUSTER = 2                   # artist clusters need ≥2 members
_N_PAIRS = 4000                                  # pair-classification: 2k pos + 2k neg
_RERANK_N_POS, _RERANK_N_NEG = 3, 10             # candidates/query (like GTZANAudioReranking)
_SEED = 94                                       # config.DEFAULT_SEED
_SR = 44100                                      # FMA native sample rate


# ---------------------------------------------------------------------------
# Path resolution + CSV loading (single source of truth)
# ---------------------------------------------------------------------------

# Repo root (…/C-VAE) holds config.py. Inject it here so `import config` works
# regardless of which encoder is loaded (SAO does not put it on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fma_paths() -> tuple[Path, Path]:
    """Resolve (tracks.csv, fma_large audio dir) without hardcoding.

    Returns:
        (metadata_csv, audio_dir): both verified to exist.

    Raises:
        FileNotFoundError: if either path is missing.
    """
    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import config

    csv = Path(os.getenv("FMA_METADATA") or config.FMA_METADATA)
    audio = Path(os.getenv("FMA_LARGE_DIR") or (csv.parent.parent / "fma_large"))
    if not csv.exists():
        raise FileNotFoundError(f"FMA tracks.csv not found: {csv}")
    if not audio.is_dir():
        raise FileNotFoundError(f"fma_large audio dir not found: {audio}")
    return csv, audio


@lru_cache(maxsize=1)
def _load_test_df():
    """Load tracks.csv → test-split DataFrame with existing-file paths.

    Returns:
        pandas.DataFrame indexed by track_id, restricted to the test split
        (all subset tiers) whose mp3 file actually exists on disk. Adds a
        ``_path`` column with the absolute mp3 path.
    """
    import pandas as pd

    csv, audio = _fma_paths()
    df = pd.read_csv(csv, index_col=0, header=[0, 1])
    mask = (df[("set", "split")] == "test") & df[("set", "subset")].isin(_SUBSETS)
    df = df[mask].copy()
    # FMA naming: track 2 → 000/000002.mp3
    df["_path"] = [str(audio / f"{t:06d}"[:3] / f"{t:06d}.mp3") for t in df.index]
    df["_exists"] = [Path(p).exists() for p in df["_path"]]
    n_missing = int((~df["_exists"]).sum())
    if n_missing:
        log.warning("FMA test set: %d files missing on disk — skipped.", n_missing)
    return df[df["_exists"]]


def _audio_dataset(paths: list[str], labels: list, label_name: str = "label"):
    """Build a datasets.Dataset with a lazy-decoding 'audio' column + label.

    Args:
        paths:      Absolute mp3 file paths.
        labels:     One label per path (int or str).
        label_name: Name of the label column.

    Returns:
        datasets.Dataset with columns ['audio', <label_name>]; audio decoded
        lazily to {array, sampling_rate} on access.
    """
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio": list(paths), label_name: list(labels)})
    # decode=False: yield {path, bytes} so a corrupt mp3 does NOT crash the
    # DataLoader; the encoder decodes each clip robustly and skips failures.
    return ds.cast_column("audio", Audio(sampling_rate=_SR, decode=False))


def _stratified_cap_indices(labels: list, k: int, seed: int) -> list[int]:
    """Pick ``k`` indices stratified by label (deterministic).

    Keeps class proportions; guarantees ≥1 per present label while the budget
    allows. Returns all indices unchanged when ``k >= len(labels)``.

    Args:
        labels: Per-sample labels.
        k:      Target number of indices to keep.
        seed:   RNG seed.

    Returns:
        Sorted list of selected positional indices.
    """
    n = len(labels)
    if k >= n:
        return list(range(n))
    rng = random.Random(seed)
    by_label: dict[Any, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    for idxs in by_label.values():
        rng.shuffle(idxs)
    # Proportional allocation per label (at least 1 each while budget remains).
    keep: list[int] = []
    quotas = {lab: max(1, round(k * len(idxs) / n)) for lab, idxs in by_label.items()}
    for lab, idxs in by_label.items():
        keep.extend(idxs[: quotas[lab]])
    # Trim/pad to exactly k deterministically.
    rng.shuffle(keep)
    if len(keep) > k:
        keep = keep[:k]
    return sorted(keep)


# ===========================================================================
# Base classes
# ===========================================================================

class _FMABase:
    """Common base: stores the ``max_files`` cap and exposes ``_cap``.

    Mixed in *before* the concrete ``AbsTask*`` so its ``__init__`` runs first
    and forwards ``seed`` down the MRO.
    """

    def __init__(self, max_files: int = 0, seed: int = _SEED, **kwargs: Any) -> None:
        self.max_files = max_files          # 0 = use all samples
        super().__init__(seed=seed, **kwargs)

    def _cap(self, n_available: int) -> int:
        """min(n_available, max_files); max_files<=0 means 'all'."""
        return n_available if self.max_files <= 0 else min(n_available, self.max_files)


class _FMADatasetTask(_FMABase):
    """Base for tasks whose data lives in ``self.dataset`` (cls/clustering/pair).

    Subclasses implement ``_build_dataset() -> DatasetDict``.
    """

    def _build_dataset(self):  # noqa: D401 - implemented by subclasses
        raise NotImplementedError

    def load_data(self, **kwargs: Any) -> None:
        """Build ``self.dataset`` in-memory from FMA; no HuggingFace hub I/O."""
        if getattr(self, "data_loaded", False):
            return
        self.dataset = self._build_dataset()
        self.dataset_transform()
        self.data_loaded = True


# ===========================================================================
# Task metadata helper
# ===========================================================================

def _fma_metadata(TaskMetadata, *, name, description, task_type, category,
                  eval_splits, main_score, task_subtypes):
    """Build a TaskMetadata with FMA-local defaults (path is cosmetic)."""
    return TaskMetadata(
        name=name,
        description=description,
        reference="https://github.com/mdeff/fma",
        dataset={"path": "FMA-local", "revision": "1.0"},   # unused: load_data overridden
        type=task_type,
        category=category,
        eval_splits=eval_splits,
        eval_langs=["zxx-Zxxx"],          # music, language-agnostic
        main_score=main_score,
        date=("2017-01-01", "2017-12-31"),
        domains=["Music"],
        task_subtypes=task_subtypes,
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fma_dataset,
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  title = {{FMA}: A Dataset for Music Analysis},
  year = {2017},
}
""",
    )


# ===========================================================================
# 1. Genre classification (K-fold cross-validation)
# ===========================================================================

def _import_abstasks():
    """Lazy import of mteb abstasks + TaskMetadata (mteb is heavy)."""
    from mteb.abstasks import AbsTaskClassification, AbsTaskClustering
    from mteb.abstasks.pair_classification import AbsTaskPairClassification
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.task_metadata import TaskMetadata
    return (AbsTaskClassification, AbsTaskClustering,
            AbsTaskPairClassification, AbsTaskRetrieval, TaskMetadata)


(_AbsClassification, _AbsClustering,
 _AbsPairClassification, _AbsRetrieval, _TaskMetadata) = _import_abstasks()


class FMAGenreClassification(_FMADatasetTask, _AbsClassification):
    """16-class genre classification on FMA test, via 5-fold cross-validation."""

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAGenreClassification",
        description="Music genre classification on the FMA large test set "
                    "(top-level genre, 16 classes).",
        task_type="AudioClassification",
        category="a2c",
        eval_splits=["train"],            # CV runs on a single split
        main_score="accuracy",
        task_subtypes=["Music Genre Classification"],
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    is_cross_validation: bool = True

    def _build_dataset(self):
        from datasets import DatasetDict

        df = _load_test_df()
        sub = df[df[("track", "genre_top")].notna()]
        labels = sub[("track", "genre_top")].astype("category").cat.codes.tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("FMAGenreClassification: %d clips, %d genres.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# 2. Genre clustering
# ===========================================================================

class FMAGenreClustering(_FMADatasetTask, _AbsClustering):
    """Genre clustering on FMA test (top-level genre, 16 classes)."""

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAGenreClustering",
        description="Music genre clustering on the FMA large test set "
                    "(top-level genre, 16 classes).",
        task_type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        main_score="v_measure",
        task_subtypes=["Music Clustering"],
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    max_fraction_of_documents_to_embed = None

    def _build_dataset(self):
        from datasets import DatasetDict

        df = _load_test_df()
        sub = df[df[("track", "genre_top")].notna()]
        labels = sub[("track", "genre_top")].astype("category").cat.codes.tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("FMAGenreClustering: %d clips, %d genres.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# 3. Artist clustering
# ===========================================================================

class FMAArtistClustering(_FMADatasetTask, _AbsClustering):
    """Artist-id clustering on FMA test (artists with ≥2 clips)."""

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAArtistClustering",
        description="Artist clustering on the FMA large test set "
                    "(artists with at least 2 tracks).",
        task_type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        main_score="v_measure",
        task_subtypes=["Music Clustering"],
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    max_fraction_of_documents_to_embed = None

    def _build_dataset(self):
        from datasets import DatasetDict

        df = _load_test_df()
        counts = df.groupby((("artist", "id"))).size()
        keep_artists = set(counts[counts >= _ARTIST_MIN_TRACKS_CLUSTER].index)
        sub = df[df[("artist", "id")].isin(keep_artists)]
        labels = sub[("artist", "id")].astype(int).tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("FMAArtistClustering: %d clips, %d artists.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# Retrieval / reranking helpers
# ===========================================================================

def _id_audio_dataset(ids: list[str], paths: list[str]):
    """Build a datasets.Dataset with 'id' + lazy-decoding 'audio' columns."""
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"id": list(ids), "audio": list(paths)})
    return ds.cast_column("audio", Audio(sampling_rate=_SR, decode=False))


def _select_groups(df, group_col, cap: int, seed: int):
    """Keep whole label-groups until the row budget ``cap`` is reached.

    Preserves group integrity (all rows of a chosen group are kept) so qrels /
    candidate lists stay consistent. ``cap<=0`` keeps everything.

    Args:
        df:        DataFrame to subsample.
        group_col: Column whose groups are selected atomically.
        cap:       Max total rows to keep (0 = all).
        seed:      RNG seed for deterministic group ordering.

    Returns:
        Filtered DataFrame.
    """
    if cap <= 0 or cap >= len(df):
        return df
    rng = random.Random(seed)
    groups = list(df.groupby(group_col).groups.items())   # [(gid, index), ...]
    rng.shuffle(groups)
    keep_idx, total = [], 0
    for _, idx in groups:
        keep_idx.extend(list(idx))
        total += len(idx)
        if total >= cap:
            break
    return df.loc[keep_idx]


# ===========================================================================
# 4. Artist audio-to-audio retrieval
# ===========================================================================

class FMAArtistA2ARetrieval(_FMABase, _AbsRetrieval):
    """Same-artist audio-to-audio retrieval on FMA test (artists with ≥5 clips).

    Query and corpus share the same clip set; for each query the relevant
    documents are all clips by the same artist. Mirrors JamAltArtistA2ARetrieval
    but monolingual (zxx) and backed by local FMA files.
    """

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAArtistA2ARetrieval",
        description="Given an FMA audio clip (query), retrieve all clips by the "
                    "same artist from the FMA large test set (artists with ≥5 tracks).",
        task_type="Any2AnyRetrieval",
        category="a2a",
        eval_splits=["test"],
        main_score="ndcg_at_10",
        task_subtypes=["Music Genre Classification"],   # upstream JamAlt uses this for artist A2A
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return

        df = _load_test_df()
        counts = df.groupby((("artist", "id"))).size()
        keep_artists = set(counts[counts >= _ARTIST_MIN_TRACKS_RETRIEVAL].index)
        sub = df[df[("artist", "id")].isin(keep_artists)]
        sub = _select_groups(sub, ("artist", "id"), self._cap(len(sub)), self.seed)

        tids = list(sub.index)
        paths = sub["_path"].tolist()
        artist_of = {t: int(a) for t, a in zip(tids, sub[("artist", "id")])}

        # qrels: each query → every corpus clip of the same artist
        by_artist: dict[int, list[int]] = {}
        for t in tids:
            by_artist.setdefault(artist_of[t], []).append(t)
        relevant = {
            f"q{t}": {f"c{t2}": 1 for t2 in by_artist[artist_of[t]]}
            for t in tids
        }

        split = "test"
        self.corpus = {split: _id_audio_dataset([f"c{t}" for t in tids], paths)}
        self.queries = {split: _id_audio_dataset([f"q{t}" for t in tids], paths)}
        self.relevant_docs = {split: relevant}
        log.info("FMAArtistA2ARetrieval: %d clips, %d artists.",
                 len(tids), len(by_artist))
        self.data_loaded = True


# ===========================================================================
# 5. Genre audio reranking (per-query candidate lists)
# ===========================================================================

class FMAGenreAudioReranking(_FMABase, _AbsRetrieval):
    """Genre-based audio reranking on FMA test.

    Each query gets a fixed candidate list of ``_RERANK_N_POS`` same-genre
    positives and ``_RERANK_N_NEG`` different-genre negatives; the model must
    rank positives above negatives (map). Mirrors GTZANAudioReranking.
    """

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAGenreAudioReranking",
        description="Given an FMA audio clip (query), rerank a candidate list so "
                    f"that {_RERANK_N_POS} same-genre clips rank above "
                    f"{_RERANK_N_NEG} different-genre clips (FMA large test set).",
        task_type="AudioReranking",
        category="a2a",
        eval_splits=["test"],
        main_score="map_at_1000",
        task_subtypes=["Music Genre Reranking"],
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return

        df = _load_test_df()
        sub = df[df[("track", "genre_top")].notna()]
        genre_of = {t: g for t, g in zip(sub.index, sub[("track", "genre_top")])}
        path_of = {t: p for t, p in zip(sub.index, sub["_path"])}

        by_genre: dict[str, list[int]] = {}
        for t, g in genre_of.items():
            by_genre.setdefault(g, []).append(t)
        # genres with enough positives + at least one other genre present
        valid_genres = {g for g, ts in by_genre.items() if len(ts) > _RERANK_N_POS}

        rng = random.Random(self.seed)
        query_tids = [t for t in sub.index if genre_of[t] in valid_genres]
        rng.shuffle(query_tids)
        query_tids = query_tids[: self._cap(len(query_tids))]

        all_other = list(sub.index)
        relevant: dict[str, dict[str, int]] = {}
        top_ranked: dict[str, list[str]] = {}
        corpus_tids: set[int] = set()

        for q in query_tids:
            g = genre_of[q]
            pos_pool = [t for t in by_genre[g] if t != q]
            neg_pool = [t for t in all_other if genre_of[t] != g]
            if len(pos_pool) < _RERANK_N_POS or len(neg_pool) < _RERANK_N_NEG:
                continue
            pos = rng.sample(pos_pool, _RERANK_N_POS)
            neg = rng.sample(neg_pool, _RERANK_N_NEG)
            cand = pos + neg
            rng.shuffle(cand)
            qid = f"q{q}"
            relevant[qid] = {f"c{t}": 1 for t in pos}
            top_ranked[qid] = [f"c{t}" for t in cand]
            corpus_tids.update(cand)

        query_tids = [int(qid[1:]) for qid in relevant]   # drop skipped queries
        split = "test"
        self.corpus = {split: _id_audio_dataset(
            [f"c{t}" for t in corpus_tids], [path_of[t] for t in corpus_tids])}
        self.queries = {split: _id_audio_dataset(
            [f"q{t}" for t in query_tids], [path_of[t] for t in query_tids])}
        self.relevant_docs = {split: relevant}
        self.top_ranked = {split: top_ranked}
        log.info("FMAGenreAudioReranking: %d queries, %d corpus clips.",
                 len(query_tids), len(corpus_tids))
        self.data_loaded = True


# ===========================================================================
# 6. Artist pair classification
# ===========================================================================

def _pair_audio_dataset(p1: list[str], p2: list[str], labels: list[int]):
    """Build a datasets.Dataset with two lazy audio columns + binary label."""
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio1": list(p1), "audio2": list(p2), "label": list(labels)})
    return (ds.cast_column("audio1", Audio(sampling_rate=_SR, decode=False))
              .cast_column("audio2", Audio(sampling_rate=_SR, decode=False)))


class FMAArtistPairClassification(_FMADatasetTask, _AbsPairClassification):
    """Same-artist pair classification on FMA test.

    Balanced positive (same artist, different clips) and negative (different
    artists) pairs; the model scores pair similarity (max average precision).
    Replaces a genre-based pair task — artist identity is a far stronger,
    fully-covered, less-skewed similarity signal than coarse top-level genre.
    """

    metadata = _fma_metadata(
        _TaskMetadata,
        name="FMAArtistPairClassification",
        description="Classify FMA clip pairs as same-artist or different-artist "
                    "(balanced) on the FMA large test set.",
        task_type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        main_score="max_ap",
        task_subtypes=["Duplicate Detection"],
    )

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"

    def _build_dataset(self):
        from datasets import DatasetDict

        df = _load_test_df()
        counts = df.groupby((("artist", "id"))).size()
        multi = set(counts[counts >= 2].index)              # artists usable for positives
        sub = df[df[("artist", "id")].isin(multi)]
        by_artist: dict[int, list[int]] = {}
        for t, a in zip(sub.index, sub[("artist", "id")]):
            by_artist.setdefault(int(a), []).append(t)
        path_of = {t: p for t, p in zip(df.index, df["_path"])}

        n_pairs = self._cap(_N_PAIRS)
        n_pos = n_neg = n_pairs // 2
        rng = random.Random(self.seed)
        artists = list(by_artist)

        # Positive pairs: two distinct clips from the same artist.
        p1, p2, labels = [], [], []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(labels) < n_pos and attempts < n_pos * 50:
            attempts += 1
            a = rng.choice(artists)
            if len(by_artist[a]) < 2:
                continue
            x, y = rng.sample(by_artist[a], 2)
            key = (min(x, y), max(x, y))
            if key in seen:
                continue
            seen.add(key)
            p1.append(path_of[x]); p2.append(path_of[y]); labels.append(1)

        # Negative pairs: clips from two different artists.
        attempts = 0
        while len(labels) < n_pos + n_neg and attempts < n_neg * 50:
            attempts += 1
            a, b = rng.sample(artists, 2)
            x = rng.choice(by_artist[a]); y = rng.choice(by_artist[b])
            key = (min(x, y), max(x, y))
            if key in seen:
                continue
            seen.add(key)
            p1.append(path_of[x]); p2.append(path_of[y]); labels.append(0)

        # Deterministic shuffle so positives/negatives interleave.
        order = list(range(len(labels)))
        rng.shuffle(order)
        p1 = [p1[i] for i in order]
        p2 = [p2[i] for i in order]
        labels = [labels[i] for i in order]
        log.info("FMAArtistPairClassification: %d pairs (%d pos, %d neg).",
                 len(labels), sum(labels), len(labels) - sum(labels))
        return DatasetDict({"test": _pair_audio_dataset(p1, p2, labels)})


# ===========================================================================
# Registry + public API
# ===========================================================================

FMA_TASK_REGISTRY: dict[str, type] = {
    "FMAGenreClassification": FMAGenreClassification,
    "FMAGenreClustering": FMAGenreClustering,
    "FMAArtistClustering": FMAArtistClustering,
    "FMAArtistA2ARetrieval": FMAArtistA2ARetrieval,
    "FMAGenreAudioReranking": FMAGenreAudioReranking,
    "FMAArtistPairClassification": FMAArtistPairClassification,
}


def get_fma_tasks(names: list[str] | None = None, *, max_files: int = 0) -> list:
    """Instantiate the requested FMA tasks (or all), propagating ``max_files``.

    Args:
        names:     Subset of FMA task names; ``None`` = all registered tasks.
        max_files: Per-task sample cap (0 = all).

    Returns:
        List of instantiated task objects ready for ``run_maeb``.
    """
    selected = names if names is not None else list(FMA_TASK_REGISTRY)
    return [FMA_TASK_REGISTRY[n](max_files=max_files)
            for n in selected if n in FMA_TASK_REGISTRY]
