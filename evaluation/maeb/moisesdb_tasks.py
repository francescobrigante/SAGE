#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/moisesdb_tasks.py
# MAEB(audio-only) task definitions backed by the LOCAL MoisesDB chunks_30s
# dataset. Music-semantic tasks (genre / artist / instrument) that load from
# the flat ``chunks_30s/`` directory + ``data.json`` metadata.
# =============================================================================
"""MoisesDB-local MAEB tasks (chunks_30s layout).

Seven tasks evaluated on the MoisesDB v0.1 dataset (240 multi-track songs,
WAV 44.1 kHz stereo, pre-processed 30-second chunks in
``chunks_30s/``):

* ``MoisesDBGenreClassification``      — genre classification (≥5 tracks/genre → 6 classes, CV)
* ``MoisesDBGenreClustering``          — genre clustering
* ``MoisesDBArtistClustering``         — artist-id clustering
* ``MoisesDBArtistA2ARetrieval``       — same-artist audio-to-audio retrieval
* ``MoisesDBGenreAudioReranking``      — per-query candidate reranking by genre
* ``MoisesDBArtistPairClassification`` — same-artist pair classification
* ``MoisesDBInstrumentClassification`` — stem instrument classification (7 classes, grouped CV)

Audio layout
~~~~~~~~~~~~
Flat directory: ``{track_uuid}_{stem_name}.wav`` + ``{track_uuid}_mixture.wav``
→ 1611 total files (240 mixtures + 1371 stems).

Tasks 1-6 (genre/artist): use ``_mixture.wav`` files — one per track (240 clips).
Task 7 (instrument): use stems only, filtered to the ``_INSTRUMENT_CLASSES``
whitelist of real, populous instrument classes. GroupKFold by track_id prevents
intra-track leakage across CV folds.
"""
from __future__ import annotations

import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Dataset thresholds (calibrated for 240 tracks / ~1371 stems)
_GENRE_MIN_TRACKS = 5                   # genres with fewer tracks are excluded
_ARTIST_MIN_TRACKS_RETRIEVAL = 3        # artists with ≥3 tracks for retrieval queries
_ARTIST_MIN_TRACKS_CLUSTER = 2          # artists with ≥2 tracks for clustering
_N_PAIRS = 1000                         # pair classification: 500 pos + 500 neg
_RERANK_N_POS, _RERANK_N_NEG = 2, 5    # reranking candidates per query
_SEED = 94                              # sage.constants.DEFAULT_SEED
_SR = 44100                             # MoisesDB native sample rate

# Instrument classification: only real, populous classes (≥45 samples).
# Excludes: other (39, catch-all), other_plucked (7, too few + ambiguous),
# bowed_strings (45, too sparse for 5-fold CV), wind (26, too few).
_INSTRUMENT_CLASSES = frozenset({
    "vocals",         # 239 samples
    "drums",          # 238
    "bass",           # 236
    "guitar",         # 222
    "piano",          # 110
    "percussion",     # 99
    "other_keys",     # 110 (organ, synth — explicitly approved by user)
})


# ---------------------------------------------------------------------------
# Path resolution + data loading
# ---------------------------------------------------------------------------

# Set by evaluation.maeb from configs/paths (paths.moisesdb_root, paths.moisesdb_chunks).
MOISESDB_ROOT: str | None = None
MOISESDB_CHUNKS: str | None = None


def _moisesdb_metadata_root() -> Path:
    """Resolve the MoisesDB metadata root (original ``moisesdb_v0.1/`` layout).

    Needed to read ``data.json`` per track for artist/genre/song metadata.
    Audio is loaded from ``chunks_30s/`` instead.
    """
    if MOISESDB_ROOT and Path(MOISESDB_ROOT).is_dir():
        return Path(MOISESDB_ROOT)
    raise FileNotFoundError(f"MoisesDB metadata root not found: {MOISESDB_ROOT} "
                            "(set paths.moisesdb_root, env MOISESDB_ROOT, to moisesdb_v0.1/)")


def _chunks_30s_root() -> Path:
    """The MoisesDB chunks_30s audio directory (MOISESDB_CHUNKS)."""
    if MOISESDB_CHUNKS and Path(MOISESDB_CHUNKS).is_dir():
        return Path(MOISESDB_CHUNKS)
    raise FileNotFoundError(f"MoisesDB chunks_30s dir not found: {MOISESDB_CHUNKS} "
                            "(set paths.moisesdb_chunks, env MOISESDB_CHUNKS_ROOT)")


@lru_cache(maxsize=1)
def _load_track_metadata() -> dict[str, dict[str, str]]:
    """Load per-track metadata (artist, genre, song) from data.json files.

    Returns:
        Dict mapping track_id (UUID) → {artist, genre, song}.
    """
    root = _moisesdb_metadata_root()
    metadata: dict[str, dict[str, str]] = {}
    for track_dir in sorted(root.iterdir()):
        meta_file = track_dir / "data.json"
        if not track_dir.is_dir() or not meta_file.exists():
            continue
        with open(meta_file) as f:
            meta = json.load(f)
        metadata[track_dir.name] = {
            "artist": meta.get("artist", "unknown"),
            "genre": meta.get("genre", "unknown"),
            "song": meta.get("song", "unknown"),
        }
    log.info("MoisesDB metadata loaded: %d tracks.", len(metadata))
    return metadata


@lru_cache(maxsize=1)
def _load_chunks_df():
    """Load all MoisesDB chunks_30s/ files → DataFrame.

    Scans the flat ``chunks_30s/`` directory, parses filenames
    ``{uuid}_{stem_name}.wav``, and joins with per-track metadata from
    ``data.json``.

    Returns:
        pandas.DataFrame with columns:
        track_id, artist, genre, song, stem_name, is_mixture, _path
    """
    import pandas as pd

    chunks_root = _chunks_30s_root()
    track_meta = _load_track_metadata()

    rows: list[dict[str, Any]] = []
    for wav_path in sorted(chunks_root.glob("*.wav")):
        fname = wav_path.stem  # e.g. "014f3712-..._bass" or "014f3712-..._mixture"
        # Split on first underscore-after-UUID (UUID is 36 chars)
        track_id = fname[:36]
        stem_name = fname[37:]  # everything after the UUID + underscore

        meta = track_meta.get(track_id, {})
        rows.append({
            "track_id": track_id,
            "artist": meta.get("artist", "unknown"),
            "genre": meta.get("genre", "unknown"),
            "song": meta.get("song", "unknown"),
            "stem_name": stem_name,
            "is_mixture": stem_name == "mixture",
            "_path": str(wav_path),
        })

    df = pd.DataFrame(rows)
    n_mix = int(df["is_mixture"].sum())
    n_stems = int((~df["is_mixture"]).sum())
    log.info("MoisesDB chunks_30s loaded: %d tracks, %d mixtures, %d stems.",
             df["track_id"].nunique(), n_mix, n_stems)
    return df


def _get_mixtures(df=None) -> "pd.DataFrame":
    """Return one mixture WAV per track (240 rows)."""
    if df is None:
        df = _load_chunks_df()
    return df[df["is_mixture"]].reset_index(drop=True)


def _get_stems(df=None) -> "pd.DataFrame":
    """Return all stem WAVs (non-mixture, 1371 rows)."""
    if df is None:
        df = _load_chunks_df()
    return df[~df["is_mixture"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset builders (shared helpers)
# ---------------------------------------------------------------------------

def _audio_dataset(paths: list[str], labels: list, label_name: str = "label"):
    """Build a datasets.Dataset with lazy-decoding 'audio' column + label."""
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio": list(paths), label_name: list(labels)})
    return ds.cast_column("audio", Audio(sampling_rate=_SR, decode=False))


def _id_audio_dataset(ids: list[str], paths: list[str]):
    """Build a datasets.Dataset with 'id' + lazy-decoding 'audio' columns."""
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"id": list(ids), "audio": list(paths)})
    return ds.cast_column("audio", Audio(sampling_rate=_SR, decode=False))


def _pair_audio_dataset(p1: list[str], p2: list[str], labels: list[int]):
    """Build a datasets.Dataset with two lazy audio columns + binary label."""
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio1": list(p1), "audio2": list(p2), "label": list(labels)})
    return (ds.cast_column("audio1", Audio(sampling_rate=_SR, decode=False))
              .cast_column("audio2", Audio(sampling_rate=_SR, decode=False)))


def _stratified_cap_indices(labels: list, k: int, seed: int) -> list[int]:
    """Pick ``k`` indices stratified by label (deterministic).

    Keeps class proportions; guarantees ≥1 per present label while the budget
    allows. Returns all indices unchanged when ``k >= len(labels)``.
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
    keep: list[int] = []
    quotas = {lab: max(1, round(k * len(idxs) / n)) for lab, idxs in by_label.items()}
    for lab, idxs in by_label.items():
        keep.extend(idxs[: quotas[lab]])
    rng.shuffle(keep)
    if len(keep) > k:
        keep = keep[:k]
    return sorted(keep)


def _select_groups(df, group_col: str, cap: int, seed: int):
    """Keep whole label-groups until the row budget ``cap`` is reached."""
    import pandas as pd

    if cap <= 0 or cap >= len(df):
        return df
    rng = random.Random(seed)
    groups = list(df.groupby(group_col).groups.items())
    rng.shuffle(groups)
    keep_idx: list = []
    total = 0
    for _, idx in groups:
        keep_idx.extend(list(idx))
        total += len(idx)
        if total >= cap:
            break
    return df.loc[keep_idx]


# ---------------------------------------------------------------------------
# Task metadata helper
# ---------------------------------------------------------------------------

def _moisesdb_metadata(TaskMetadata, *, name, description, task_type, category,
                       eval_splits, main_score, task_subtypes):
    """Build a TaskMetadata with MoisesDB-local defaults."""
    return TaskMetadata(
        name=name,
        description=description,
        reference="https://github.com/moises-ai/moisesdb",
        dataset={"path": "MoisesDB-local", "revision": "0.1"},
        type=task_type,
        category=category,
        eval_splits=eval_splits,
        eval_langs=["zxx-Zxxx"],          # music, language-agnostic
        main_score=main_score,
        date=("2023-01-01", "2023-12-31"),
        domains=["Music"],
        task_subtypes=task_subtypes,
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{moisesdb2023,
  author = {Pereira, Igor and Martins, Felipe and Pereira, Ryan and Gandra, Yuri},
  title = {{MoisesDB}: A Dataset for Source Separation Beyond 4-Stem Music},
  year = {2023},
}
""",
    )


# ===========================================================================
# Base classes
# ===========================================================================

class _MoisesDBBase:
    """Common base: stores ``max_files`` cap. Mixed in before AbsTask*."""

    def __init__(self, max_files: int = 0, seed: int = _SEED, **kwargs: Any) -> None:
        self.max_files = max_files
        super().__init__(seed=seed, **kwargs)

    def _cap(self, n_available: int) -> int:
        """min(n_available, max_files); max_files <= 0 means 'all'."""
        return n_available if self.max_files <= 0 else min(n_available, self.max_files)


class _MoisesDBDatasetTask(_MoisesDBBase):
    """Base for tasks whose data lives in ``self.dataset`` (cls/clustering/pair)."""

    def _build_dataset(self):
        raise NotImplementedError

    def load_data(self, **kwargs: Any) -> None:
        """Build ``self.dataset`` in-memory from MoisesDB; no HuggingFace hub I/O."""
        if getattr(self, "data_loaded", False):
            return
        self.dataset = self._build_dataset()
        self.dataset_transform()
        self.data_loaded = True


# ===========================================================================
# Lazy import of MTEB base classes
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


# ===========================================================================
# Helper: filtered genre DataFrame (mixtures with ≥ _GENRE_MIN_TRACKS per genre)
# ===========================================================================

def _genre_filtered_mixtures():
    """Return mixture DataFrame, genre-filtered (≥ _GENRE_MIN_TRACKS)."""
    mixtures = _get_mixtures()
    counts = mixtures.groupby("genre").size()
    valid_genres = set(counts[counts >= _GENRE_MIN_TRACKS].index)
    return mixtures[mixtures["genre"].isin(valid_genres)]


# ===========================================================================
# 1. Genre classification (K-fold cross-validation)
# ===========================================================================

class MoisesDBGenreClassification(_MoisesDBDatasetTask, _AbsClassification):
    """Genre classification on MoisesDB (≥5 tracks/genre → 6 classes, 5-fold CV).

    Uses mixture WAVs (one per track) from the chunks_30s/ flat directory.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBGenreClassification",
        description="Music genre classification on MoisesDB mixtures "
                    "(1 mixture per track, genres with ≥5 tracks).",
        task_type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        main_score="accuracy",
        task_subtypes=["Music Genre Classification"],
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    is_cross_validation: bool = True

    def _build_dataset(self):
        from datasets import DatasetDict

        sub = _genre_filtered_mixtures()
        labels = sub["genre"].astype("category").cat.codes.tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("MoisesDBGenreClassification: %d clips, %d genres.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# 2. Genre clustering
# ===========================================================================

class MoisesDBGenreClustering(_MoisesDBDatasetTask, _AbsClustering):
    """Genre clustering on MoisesDB (≥5 tracks/genre → 6 classes).

    Uses mixture WAVs from chunks_30s/.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBGenreClustering",
        description="Music genre clustering on MoisesDB mixtures "
                    "(1 mixture per track, genres with ≥5 tracks).",
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

        sub = _genre_filtered_mixtures()
        labels = sub["genre"].astype("category").cat.codes.tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("MoisesDBGenreClustering: %d clips, %d genres.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# 3. Artist clustering
# ===========================================================================

class MoisesDBArtistClustering(_MoisesDBDatasetTask, _AbsClustering):
    """Artist-id clustering on MoisesDB (artists with ≥2 tracks).

    Uses mixture WAVs from chunks_30s/.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBArtistClustering",
        description="Artist clustering on MoisesDB mixtures "
                    "(1 mixture per track, artists with ≥2 tracks).",
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

        mixtures = _get_mixtures()
        counts = mixtures.groupby("artist").size()
        keep_artists = set(counts[counts >= _ARTIST_MIN_TRACKS_CLUSTER].index)
        sub = mixtures[mixtures["artist"].isin(keep_artists)]
        labels = sub["artist"].tolist()
        paths = sub["_path"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        log.info("MoisesDBArtistClustering: %d clips, %d artists.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})


# ===========================================================================
# 4. Artist audio-to-audio retrieval
# ===========================================================================

class MoisesDBArtistA2ARetrieval(_MoisesDBBase, _AbsRetrieval):
    """Same-artist audio-to-audio retrieval on MoisesDB (artists with ≥3 tracks).

    Uses mixture WAVs from chunks_30s/.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBArtistA2ARetrieval",
        description="Given a MoisesDB mixture (query), retrieve all mixtures by the "
                    "same artist (artists with ≥3 tracks).",
        task_type="Any2AnyRetrieval",
        category="a2a",
        eval_splits=["test"],
        main_score="ndcg_at_10",
        task_subtypes=["Music Genre Classification"],
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return

        mixtures = _get_mixtures()
        counts = mixtures.groupby("artist").size()
        keep_artists = set(counts[counts >= _ARTIST_MIN_TRACKS_RETRIEVAL].index)
        sub = mixtures[mixtures["artist"].isin(keep_artists)]
        sub = _select_groups(sub, "artist", self._cap(len(sub)), self.seed)

        tids = sub["track_id"].tolist()
        paths = sub["_path"].tolist()
        artist_of = dict(zip(tids, sub["artist"]))

        by_artist: dict[str, list[str]] = {}
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
        log.info("MoisesDBArtistA2ARetrieval: %d clips, %d artists.",
                 len(tids), len(by_artist))
        self.data_loaded = True


# ===========================================================================
# 5. Genre audio reranking
# ===========================================================================

class MoisesDBGenreAudioReranking(_MoisesDBBase, _AbsRetrieval):
    """Genre-based audio reranking on MoisesDB.

    Each query gets a fixed candidate list of same-genre positives and
    different-genre negatives; the model must rank positives above negatives.
    Uses mixture WAVs from chunks_30s/.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBGenreAudioReranking",
        description="Given a MoisesDB mixture (query), rerank a candidate list so "
                    f"that {_RERANK_N_POS} same-genre clips rank above "
                    f"{_RERANK_N_NEG} different-genre clips.",
        task_type="AudioReranking",
        category="a2a",
        eval_splits=["test"],
        main_score="map_at_1000",
        task_subtypes=["Music Genre Reranking"],
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return

        sub = _genre_filtered_mixtures()
        genre_of = dict(zip(sub["track_id"], sub["genre"]))
        path_of = dict(zip(sub["track_id"], sub["_path"]))

        by_genre: dict[str, list[str]] = {}
        for t, g in genre_of.items():
            by_genre.setdefault(g, []).append(t)
        valid_genres = {g for g, ts in by_genre.items() if len(ts) > _RERANK_N_POS}

        rng = random.Random(self.seed)
        query_tids = [t for t in sub["track_id"] if genre_of[t] in valid_genres]
        rng.shuffle(query_tids)
        query_tids = query_tids[: self._cap(len(query_tids))]

        all_other = list(sub["track_id"])
        relevant: dict[str, dict[str, int]] = {}
        top_ranked: dict[str, list[str]] = {}
        corpus_tids: set[str] = set()

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

        query_tids_final = [qid[1:] for qid in relevant]
        split = "test"
        self.corpus = {split: _id_audio_dataset(
            [f"c{t}" for t in corpus_tids], [path_of[t] for t in corpus_tids])}
        self.queries = {split: _id_audio_dataset(
            [f"q{t}" for t in query_tids_final], [path_of[t] for t in query_tids_final])}
        self.relevant_docs = {split: relevant}
        self.top_ranked = {split: top_ranked}
        log.info("MoisesDBGenreAudioReranking: %d queries, %d corpus clips.",
                 len(query_tids_final), len(corpus_tids))
        self.data_loaded = True


# ===========================================================================
# 6. Artist pair classification
# ===========================================================================

class MoisesDBArtistPairClassification(_MoisesDBDatasetTask, _AbsPairClassification):
    """Same-artist pair classification on MoisesDB.

    Balanced positive (same artist, different tracks) and negative (different
    artists) pairs. Uses mixture WAVs from chunks_30s/.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBArtistPairClassification",
        description="Classify MoisesDB mixture pairs as same-artist or different-artist "
                    "(balanced, 1 mixture per track).",
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

        mixtures = _get_mixtures()
        counts = mixtures.groupby("artist").size()
        multi = set(counts[counts >= 2].index)
        sub = mixtures[mixtures["artist"].isin(multi)]
        by_artist: dict[str, list[str]] = {}
        for _, row in sub.iterrows():
            by_artist.setdefault(row["artist"], []).append(row["track_id"])
        path_of = dict(zip(mixtures["track_id"], mixtures["_path"]))

        n_pairs = self._cap(_N_PAIRS)
        n_pos = n_neg = n_pairs // 2
        rng = random.Random(self.seed)
        artists = list(by_artist)

        p1, p2, labels = [], [], []
        seen: set[tuple[str, str]] = set()

        # Positive pairs: two distinct tracks from the same artist
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

        # Negative pairs: tracks from two different artists
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

        # Deterministic shuffle
        order = list(range(len(labels)))
        rng.shuffle(order)
        p1 = [p1[i] for i in order]
        p2 = [p2[i] for i in order]
        labels = [labels[i] for i in order]
        log.info("MoisesDBArtistPairClassification: %d pairs (%d pos, %d neg).",
                 len(labels), sum(labels), len(labels) - sum(labels))
        return DatasetDict({"test": _pair_audio_dataset(p1, p2, labels)})


# ===========================================================================
# 7. Instrument classification (GroupKFold to prevent same-track leakage)
# ===========================================================================

class MoisesDBInstrumentClassification(_MoisesDBDatasetTask, _AbsClassification):
    """Stem instrument classification on MoisesDB.

    Uses only the stems (not mixtures) from ``chunks_30s/``, filtered to the
    ``_INSTRUMENT_CLASSES`` whitelist of 7 real, populous instrument classes.
    GroupKFold with group=track_id prevents acoustic leakage across CV folds.
    """

    metadata = _moisesdb_metadata(
        _TaskMetadata,
        name="MoisesDBInstrumentClassification",
        description="Classify MoisesDB stems by instrument. "
                    "GroupKFold by track prevents intra-track leakage. "
                    f"Classes: {sorted(_INSTRUMENT_CLASSES)}.",
        task_type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        main_score="accuracy",
        task_subtypes=["Music Instrument Recognition"],
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
    is_cross_validation: bool = True

    def _build_dataset(self):
        from datasets import DatasetDict

        stems = _get_stems()
        # Filter to the instrument whitelist
        sub = stems[stems["stem_name"].isin(_INSTRUMENT_CLASSES)]

        labels = sub["stem_name"].astype("category").cat.codes.tolist()
        paths = sub["_path"].tolist()
        # Store track_ids for GroupKFold (aligned with paths/labels)
        self._group_track_ids = sub["track_id"].tolist()

        keep = _stratified_cap_indices(labels, self._cap(len(paths)), self.seed)
        paths = [paths[i] for i in keep]
        labels = [labels[i] for i in keep]
        self._group_track_ids = [self._group_track_ids[i] for i in keep]

        log.info("MoisesDBInstrumentClassification: %d stems, %d classes.",
                 len(paths), len(set(labels)))
        return DatasetDict({"train": _audio_dataset(paths, labels)})

    def _evaluate_subset_cross_validation(
        self,
        model,
        data_split,
        *,
        encode_kwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder=None,
        num_proc=None,
        **kwargs,
    ):
        """Override to use GroupKFold instead of KFold.

        All stems from the same track stay in the same fold, preventing
        acoustic leakage. The rest of the logic is identical to the parent.
        """
        from sklearn.model_selection import GroupKFold

        from mteb._create_dataloaders import create_dataloader

        if self.train_split != hf_split:
            raise ValueError(
                f"Performing grouped {self.n_splits}-fold CV, but the dataset "
                f"has train (`{self.train_split}`) and test split (`{hf_split}`)."
            )
        log.info("Performing grouped %d-fold CV (group=track_id) on %d stems.",
                 self.n_splits, len(data_split[self.train_split]))

        ds = data_split[self.train_split]
        num_samples = len(ds)

        # Build group array aligned with dataset rows
        groups = self._group_track_ids[:num_samples]

        scores = []
        idxs = None
        splitter = GroupKFold(n_splits=self.n_splits)

        dataloader_train = create_dataloader(
            ds,
            task_metadata=self.metadata,
            input_column=self.input_column_name,
            num_proc=num_proc,
            **encode_kwargs,
        )
        log.info("GroupKFold CV — encoding all samples...")
        dataset_embeddings = model.encode(
            dataloader_train,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )

        all_predictions = []
        for i, (train_idx, val_idx) in enumerate(
            splitter.split(range(num_samples), groups=groups)
        ):
            train_split_ds = ds.select(train_idx)
            eval_split_ds = ds.select(val_idx)
            train_cache = dataset_embeddings[train_idx]
            test_cache = dataset_embeddings[val_idx]
            log.info("GroupKFold fold %d/%d: train=%d, val=%d",
                     i + 1, self.n_splits, len(train_idx), len(val_idx))
            scores_exp, predictions, idxs, _ = self._run_experiment(
                model,
                train_split_ds,
                eval_split_ds,
                experiment_num=i,
                idxs=idxs,
                encode_kwargs=encode_kwargs,
                hf_split=hf_split,
                hf_subset=hf_subset,
                test_cache=test_cache,
                train_cache=train_cache,
                num_proc=num_proc,
            )
            if prediction_folder:
                all_predictions.append(predictions)
            scores.append(scores_exp)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions, model, prediction_folder,
                hf_subset=hf_subset, hf_split=hf_split,
            )
        return self._calculate_avg_scores(scores)


# ===========================================================================
# Registry + public API
# ===========================================================================

MOISESDB_TASK_REGISTRY: dict[str, type] = {
    "MoisesDBGenreClassification": MoisesDBGenreClassification,
    "MoisesDBGenreClustering": MoisesDBGenreClustering,
    "MoisesDBArtistClustering": MoisesDBArtistClustering,
    "MoisesDBArtistA2ARetrieval": MoisesDBArtistA2ARetrieval,
    "MoisesDBGenreAudioReranking": MoisesDBGenreAudioReranking,
    "MoisesDBArtistPairClassification": MoisesDBArtistPairClassification,
    "MoisesDBInstrumentClassification": MoisesDBInstrumentClassification,
}

MOISESDB_SUITE = list(MOISESDB_TASK_REGISTRY.keys())


def get_moisesdb_tasks(names: list[str] | None = None, *, max_files: int = 0) -> list:
    """Instantiate the requested MoisesDB tasks (or all), propagating ``max_files``.

    Args:
        names:     Subset of MoisesDB task names; ``None`` = all registered tasks.
        max_files: Per-task sample cap (0 = all).

    Returns:
        List of instantiated task objects ready for ``run_maeb``.
    """
    selected = names if names is not None else list(MOISESDB_TASK_REGISTRY)
    return [MOISESDB_TASK_REGISTRY[n](max_files=max_files)
            for n in selected if n in MOISESDB_TASK_REGISTRY]
