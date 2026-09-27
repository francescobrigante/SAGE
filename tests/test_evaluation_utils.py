# ===============
# Evaluation-set file lists (evaluation/common.py): the FMA test split selected
# through tracks.csv, and the flat clip sets whose reference folders are skipped.
# ===============
from pathlib import Path

import pandas as pd

from evaluation.common import collect_clip_files, collect_fma_files


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
