# ===============================================================
# test_providers_multicorpus.py
#
#   Correctness of the Jamendo and M4Singer metadata providers:
#   they map their on-disk layout onto a SORTED, deterministic list
#   of absolute audio paths (the canonical order the rotating sampler
#   chunks on). Synthetic tmp fixtures prove the logic in isolation;
#   real-disk smoke tests (skipped if the corpora are absent) confirm
#   the expected track counts (Jamendo 32,859 / M4Singer 20,896).
# ===============================================================
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from ar_spectra.utils.metadata import jamendo, m4singer
import config

JAMENDO_TRAIN_COUNT = 32_859
M4SINGER_COUNT = 20_896


# ───────────────────────── Jamendo ──────────────────────────────────────────
def _write_tsv(path: Path, rows: list[list[str]]):
    header = ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_jamendo_maps_path_col_onto_audio_root(tmp_path):
    tsv = tmp_path / "train.tsv"
    _write_tsv(tsv, [
        ["track_0000241", "artist_5", "album_33", "41/241.mp3", "340.1", "genre---rock"],
        ["track_0000242", "artist_5", "album_33", "42/242.mp3", "248.5", "genre---rock"],
    ])
    audio = tmp_path / "audio"
    out = jamendo.get_audio_files(str(audio), split_tsv=str(tsv))
    assert out == [str(audio / "41/241.mp3"), str(audio / "42/242.mp3")]


def test_jamendo_is_sorted_and_skips_header(tmp_path):
    tsv = tmp_path / "train.tsv"
    # deliberately unsorted input → provider must return sorted
    _write_tsv(tsv, [
        ["t3", "a", "b", "99/3.mp3", "10", "genre---x"],
        ["t1", "a", "b", "10/1.mp3", "10", "genre---y"],
        ["t2", "a", "b", "50/2.mp3", "10", "genre---z"],
    ])
    out = jamendo.get_audio_files(str(tmp_path / "audio"), split_tsv=str(tsv))
    assert out == sorted(out)
    assert len(out) == 3                       # header not counted as a track
    assert all(p.endswith(".mp3") for p in out)


def test_jamendo_handles_multitag_extra_columns(tmp_path):
    # TAGS may contain extra tab-separated tags after col 3 → PATH must stay col 3
    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "t1\ta\tb\t10/1.mp3\t10\tgenre---rock\tgenre---pop\tinstrument---guitar\n",
        encoding="utf-8",
    )
    out = jamendo.get_audio_files(str(tmp_path / "audio"), split_tsv=str(tsv))
    assert out == [str(tmp_path / "audio" / "10/1.mp3")]


def test_jamendo_missing_tsv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        jamendo.get_audio_files(str(tmp_path / "audio"),
                                split_tsv=str(tmp_path / "nope.tsv"))


@pytest.mark.skipif(
    not (config.JAMENDO_SPLIT_TSV and os.path.exists(config.JAMENDO_SPLIT_TSV)),
    reason="real Jamendo split TSV not mounted",
)
def test_jamendo_real_corpus_count_and_format():
    out = jamendo.get_audio_files(config.JAMENDO_AUDIO)
    assert len(out) == JAMENDO_TRAIN_COUNT
    assert out == sorted(out)
    assert all(p.endswith(".mp3") for p in out)
    assert all(p.startswith(config.JAMENDO_AUDIO) for p in out)


# ───────────────────────── M4Singer ─────────────────────────────────────────
def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_m4singer_recursive_sorted_wav_only(tmp_path):
    root = tmp_path / "audio"
    # nested layout like m4singer/Singer#Song/NNNN.wav, plus a non-wav decoy
    _touch(root / "m4singer/Tenor-4#song/0008.wav")
    _touch(root / "m4singer/Tenor-4#song/0006.wav")
    _touch(root / "m4singer/Alto-1#newboy/0001.wav")
    _touch(root / "m4singer/Alto-1#newboy/notes.txt")     # must be ignored
    out = m4singer.get_audio_files(str(root))
    assert out == sorted(out)
    assert len(out) == 3
    assert all(p.endswith(".wav") for p in out)


def test_m4singer_honours_filelist(tmp_path):
    # get_audio_filenames short-circuits to filelist.txt if present at the root
    root = tmp_path / "audio"
    root.mkdir()
    _touch(root / "a/0001.wav")
    _touch(root / "b/0002.wav")
    (root / "filelist.txt").write_text("b/0002.wav\na/0001.wav\n", encoding="utf-8")
    out = m4singer.get_audio_files(str(root))
    assert out == sorted([str(root / "b/0002.wav"), str(root / "a/0001.wav")])


@pytest.mark.skipif(
    not (config.M4SINGER_AUDIO and os.path.exists(config.M4SINGER_AUDIO)),
    reason="real M4Singer corpus not mounted",
)
def test_m4singer_real_corpus_count():
    out = m4singer.get_audio_files(config.M4SINGER_AUDIO)
    assert len(out) == M4SINGER_COUNT
    assert out == sorted(out)
    assert all(p.endswith(".wav") for p in out)
