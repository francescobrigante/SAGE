# ===============================================================
# test_crop_rng.py
#
#   Verifies the per-item crop-RNG fix in OnTheFlySTFTDataset:
#   crops are a pure function of (base_seed, epoch, file index) —
#   deterministic, vary across epochs (train), stay fixed when the
#   epoch never advances (val), and the shared-memory epoch counter
#   propagates to forked persistent DataLoader workers.
# ===============================================================
import sys
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sage.training.data.dataset as dl

N_FILES = 8
RAMP_LEN = 100_000          # > segment_samples (127*512 = 65024) so we always crop


def _fake_load(path, target_sample_rate, expected_channels):
    """Return a deterministic stereo ramp: sample t has value t, so the first
    sample of a crop reveals its start offset."""
    ramp = torch.arange(RAMP_LEN, dtype=torch.float32).unsqueeze(0).repeat(2, 1)  # (2, L)
    return ramp, target_sample_rate, False, False


def _make_dataset(tmp_path: Path, seed: int = 123):
    for i in range(N_FILES):
        (tmp_path / f"{i:03d}.wav").touch()
    return dl.OnTheFlySTFTDataset(
        audio_dir=str(tmp_path),
        sample_rate=44100,
        n_fft=2048, hop_length=512, win_length=2048,
        target_frames=128,
        extensions=[".wav"],
        stereo=True, cac=True,
        skip_failed_samples=False,
        seed=seed,
    )


def _starts(ds, indices):
    """Crop start offset for each index (read from the ramp's first sample)."""
    out = []
    for i in indices:
        _, seg = ds[i]
        out.append(int(round(seg[0, 0].item())))
    return out


# ── probe disabled so empty dummy files survive scanning; load mocked ─────────
def _patches():
    return patch.object(dl, "pick_probe_fn", return_value=None), \
           patch.object(dl, "load_waveform", side_effect=_fake_load)


# ── T1: deterministic — same (epoch, index) → identical crop ──────────────────
def test_crop_deterministic_same_epoch(tmp_path):
    p1, p2 = _patches()
    with p1, p2:
        ds = _make_dataset(tmp_path)
        a = _starts(ds, range(N_FILES))
        b = _starts(ds, range(N_FILES))
    assert a == b


# ── T2: validation behaviour — epoch never advances → crops fixed ─────────────
def test_val_crops_fixed_without_set_epoch(tmp_path):
    p1, p2 = _patches()
    with p1, p2:
        ds = _make_dataset(tmp_path)
        first = _starts(ds, range(N_FILES))
        # simulate several "val passes" with no set_epoch call
        for _ in range(3):
            assert _starts(ds, range(N_FILES)) == first


# ── T3: training behaviour — advancing the epoch changes the crops ────────────
def test_crops_vary_across_epochs(tmp_path):
    p1, p2 = _patches()
    with p1, p2:
        ds = _make_dataset(tmp_path)
        e0 = _starts(ds, range(N_FILES))
        ds.set_epoch(1)
        e1 = _starts(ds, range(N_FILES))
        ds.set_epoch(2)
        e2 = _starts(ds, range(N_FILES))
    assert e0 != e1
    assert e1 != e2
    assert e0 != e2


# ── T4: reproducible across instances with the same seed ──────────────────────
def test_same_seed_same_crops(tmp_path):
    pa = tmp_path / "da"; pb = tmp_path / "db"
    pa.mkdir(); pb.mkdir()
    p1, p2 = _patches()
    with p1, p2:
        ds_a = _make_dataset(pa, seed=777)
        ds_b = _make_dataset(pb, seed=777)
        ds_a.set_epoch(3)
        ds_b.set_epoch(3)
        assert _starts(ds_a, range(N_FILES)) == _starts(ds_b, range(N_FILES))


# ── T5: a different seed yields a different crop pattern ───────────────────────
def test_different_seed_differs(tmp_path):
    pa = tmp_path / "sa"; pb = tmp_path / "sb"
    pa.mkdir(); pb.mkdir()
    p1, p2 = _patches()
    with p1, p2:
        ds_a = _make_dataset(pa, seed=1)
        ds_b = _make_dataset(pb, seed=2)
        assert _starts(ds_a, range(N_FILES)) != _starts(ds_b, range(N_FILES))


# ── T6: shared-memory epoch reaches forked persistent workers ─────────────────
@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="requires fork start method to test worker inheritance",
)
def test_epoch_propagates_to_persistent_workers(tmp_path):
    p1, p2 = _patches()
    with p1, p2:
        ds = _make_dataset(tmp_path)
        loader = DataLoader(
            ds, batch_size=2, num_workers=2, shuffle=False,
            persistent_workers=True,
            collate_fn=lambda b: [int(round(x[1][0, 0].item())) for x in b],
            multiprocessing_context=mp.get_context("fork"),
        )
        epoch0 = [s for batch in loader for s in batch]
        ds.set_epoch(1)
        epoch1 = [s for batch in loader for s in batch]
        loader._iterator = None  # release workers
    assert len(epoch0) == N_FILES
    # With the bug, persistent workers keep the epoch-0 seed → epoch1 == epoch0.
    # With the fix, the shared-memory epoch reaches the workers → crops change.
    assert epoch0 != epoch1
