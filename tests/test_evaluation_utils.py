# ===============
# Evaluation-set file lists (evaluation/common.py): the FMA test split selected
# through tracks.csv, and the flat clip sets whose reference folders are skipped;
# atomic_save_npy never leaves a *.npy behind when a write fails.
# ===============
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.common import atomic_save_npy, collect_clip_files, collect_fma_files


def test_collect_fma_files_keeps_the_test_split(tmp_path: Path):
    for sub, tid in (("000", 2), ("000", 5), ("001", 1203), ("000", 7)):
        (tmp_path / sub).mkdir(exist_ok=True)
        (tmp_path / sub / f"{tid:06d}.mp3").touch()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "000002.mp3").touch()                 # outputs are never inputs
    cols = pd.MultiIndex.from_tuples([("set", "split"), ("track", "title")])
    tracks = pd.DataFrame([["test", "a"], ["test", "b"], ["test", "c"], ["training", "d"]],
                          index=[2, 5, 1203, 7], columns=cols)
    tracks.to_csv(tmp_path / "tracks.csv")
    files = collect_fma_files(tmp_path, {".mp3"}, tmp_path / "tracks.csv", max_files=0)
    assert [f.stem for f in files] == ["000002", "000005", "001203"]
    assert all("metrics" not in f.parts for f in files)
    assert len(collect_fma_files(tmp_path, {".mp3"}, tmp_path / "tracks.csv", max_files=2)) == 2


def test_collect_clip_files_is_flat_and_sorted(tmp_path: Path):
    for name in ("b.wav", "a.wav", "notes.txt"):
        (tmp_path / name).touch()
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "embeddings" / "c.wav").touch()
    assert [f.name for f in collect_clip_files(tmp_path, 0)] == ["a.wav", "b.wav"]
    assert [f.name for f in collect_clip_files(tmp_path, 1)] == ["a.wav"]


def test_atomic_save_npy_roundtrip(tmp_path: Path):
    x = np.arange(6, dtype=np.float16).reshape(2, 3)
    atomic_save_npy(tmp_path / "emb" / "a.npy", x)
    assert [p.name for p in (tmp_path / "emb").iterdir()] == ["a.npy"]
    np.testing.assert_array_equal(np.load(tmp_path / "emb" / "a.npy"), x)


def test_atomic_save_npy_failed_write_leaves_no_npy(tmp_path: Path, monkeypatch):
    """Quota exceeded mid-write: no partial file for the merge's glob("*.npy") to pick up."""
    atomic_save_npy(tmp_path / "a.npy", np.zeros(4))

    def full_disk(f, data):
        f.write(b"\x93NUMPY")                              # partial header, then the write fails
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(np, "save", full_disk)
    with pytest.raises(OSError, match="quota"):
        atomic_save_npy(tmp_path / "b.npy", np.zeros(4))
    assert [p.name for p in tmp_path.iterdir()] == ["a.npy"]
