# ===============================================================
# test_multicorpus_dataset.py
#
#   MultiCorpusDataset wraps several OnTheFlySTFTDataset corpora in
#   one flat global index space: __len__ sums the children, a global
#   index routes to the owning corpus' item, corpus_sizes reports the
#   post-filter per-corpus counts, and set_epoch reaches every child
#   (driving per-epoch crop variation, as the single-dataset path does).
# ===============================================================
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sage.training.data.dataset as dl

SR = 44100
SEGMENT = 127 * 512
# distinct per-corpus file counts so routing / sizes are unambiguous
COUNTS = [3, 4, 2]


def _save_ramp(path: Path, length: int):
    ramp = torch.arange(length, dtype=torch.float32).unsqueeze(0).repeat(2, 1) / length
    torchaudio.save(str(path), ramp, SR, encoding="PCM_F", bits_per_sample=32)


def _make_corpus(audio_dir: Path, n_files: int, seed: int):
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)   # 4 s ≫ segment
    return dl.OnTheFlySTFTDataset(
        audio_dir=str(audio_dir),
        sample_rate=SR,
        n_fft=2048, hop_length=512, win_length=2048,
        target_frames=128,
        extensions=[".wav"],
        stereo=True, cac=True,
        skip_failed_samples=False,
        seed=seed,
    )


def _make_multicorpus(tmp_path):
    children = [
        _make_corpus(tmp_path / f"c{k}", COUNTS[k], seed=10 + k)
        for k in range(len(COUNTS))
    ]
    return dl.MultiCorpusDataset(children), children


# ── len + corpus_sizes reflect the children exactly ──────────────────────────
def test_len_and_corpus_sizes(tmp_path):
    mc, children = _make_multicorpus(tmp_path)
    assert len(mc) == sum(COUNTS)
    assert mc.corpus_sizes == [len(c) for c in children] == COUNTS


# ── a global index routes to the owning corpus' local item ───────────────────
def test_global_index_routes_to_owning_corpus(tmp_path):
    mc, children = _make_multicorpus(tmp_path)
    # global index of corpus 1, local item 2  →  must equal children[1][2]
    cum = [0]
    for c in COUNTS:
        cum.append(cum[-1] + c)
    g = cum[1] + 2
    S_mc, seg_mc = mc[g]
    S_child, seg_child = children[1][2]
    assert torch.equal(seg_mc, seg_child)     # same child object, same (epoch,index) seed
    assert torch.equal(S_mc, S_child)


# ── set_epoch reaches every child ────────────────────────────────────────────
def test_set_epoch_forwarded_to_all_children(tmp_path):
    mc, children = _make_multicorpus(tmp_path)
    mc.set_epoch(7)
    assert all(c.epoch == 7 for c in children)


# ── advancing the epoch moves the crop (per-epoch variation propagates) ───────
def test_set_epoch_varies_crops(tmp_path):
    mc, _ = _make_multicorpus(tmp_path)
    mc.set_epoch(0)
    a = mc[0][1].clone()
    mc.set_epoch(1)
    b = mc[0][1].clone()
    assert not torch.equal(a, b)              # different epoch → different window


# ── max_files deterministically caps the file list (sorted prefix) ────────────
def test_max_files_caps_deterministically(tmp_path):
    audio_dir = tmp_path / "capped"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)

    def _build(cap):
        return dl.OnTheFlySTFTDataset(
            audio_dir=str(audio_dir),
            sample_rate=SR,
            n_fft=2048, hop_length=512, win_length=2048,
            target_frames=128,
            extensions=[".wav"],
            stereo=True, cac=True,
            skip_failed_samples=False,
            seed=0,
            max_files=cap,
        )

    capped = _build(2)
    full = _build(None)
    assert len(capped) == 2                          # cap honored
    assert len(full) == 6                            # None = no cap (production)
    # deterministic: the cap keeps the sorted prefix
    assert [p.name for p in capped.files] == [p.name for p in full.files[:2]]


# ── filelist cache: write on first build, reload (no rescan) on the second ────
def _make_cache_dataset(audio_dir, cache=None, max_files=None):
    return dl.OnTheFlySTFTDataset(
        audio_dir=str(audio_dir), sample_rate=SR,
        n_fft=2048, hop_length=512, win_length=2048, target_frames=128,
        extensions=[".wav"], stereo=True, cac=True,
        skip_failed_samples=False, seed=0,
        filelist_cache=cache, max_files=max_files,
    )


def test_filelist_cache_write_then_reload(tmp_path):
    audio_dir = tmp_path / "corpus"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)
    cache = tmp_path / "cache" / "corpus.txt"

    first = _make_cache_dataset(audio_dir, cache=cache)
    assert cache.exists()                            # first build persisted the list
    assert len(first) == 5

    # Delete a source file: a real rescan would now find 4. A cache reload trusts the
    # persisted list and still returns 5 → proves the probe was skipped.
    (audio_dir / "000.wav").unlink()
    second = _make_cache_dataset(audio_dir, cache=cache)
    assert len(second) == 5
    assert [p.name for p in second.files] == [p.name for p in first.files]


def test_filelist_cache_absent_falls_back_to_scan(tmp_path):
    audio_dir = tmp_path / "corpus"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)
    cache = tmp_path / "cache" / "missing.txt"       # does not exist yet

    ds = _make_cache_dataset(audio_dir, cache=cache)
    assert len(ds) == 3                              # full scan happened
    assert cache.exists()                            # and the cache was created


def test_filelist_cache_disabled_when_capped(tmp_path):
    audio_dir = tmp_path / "corpus"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)
    cache = tmp_path / "cache" / "capped.txt"

    ds = _make_cache_dataset(audio_dir, cache=cache, max_files=2)
    assert len(ds) == 2                              # cap honored
    assert not cache.exists()                        # a capped list must NOT be persisted


def test_no_cache_by_default_leaves_no_file(tmp_path):
    audio_dir = tmp_path / "corpus"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)

    ds = _make_cache_dataset(audio_dir, cache=None)  # single-corpus default path
    assert len(ds) == 3
    # no stray cache artifacts anywhere under tmp
    assert list(tmp_path.rglob("*.txt")) == []
