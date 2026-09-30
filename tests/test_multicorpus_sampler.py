# ===============================================================
# test_multicorpus_sampler.py
#
#   MultiCorpusRotatingSampler contract: every epoch emits ALL global
#   indices of the non-rotating corpora plus one disjoint chunk of the
#   rotating corpus; consecutive chunks are disjoint; the chunked region
#   is fully covered every K epochs; output is deterministic given the
#   seed; DDP ranks partition the epoch list with equal length; and the
#   per-rank length is constant across epochs (stable steps_per_epoch).
# ===============================================================
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from sage.training.data.sampling import MultiCorpusRotatingSampler

# corpus 0 = FMA-like, 1 = Jamendo-like (both static), 2 = M4-like (rotating)
SIZES = [10, 6, 8]
K = 4
ROT = 2
SEED = 94
M4_OFFSET = SIZES[0] + SIZES[1]            # 16
FMA_GLOBALS = set(range(0, SIZES[0]))                       # {0..9}
JAM_GLOBALS = set(range(SIZES[0], SIZES[0] + SIZES[1]))     # {10..15}
M4_GLOBALS = set(range(M4_OFFSET, M4_OFFSET + SIZES[2]))    # {16..23}


def _sampler(world=1, rank=0, sizes=SIZES, k=K, rot=ROT, seed=SEED):
    return MultiCorpusRotatingSampler(
        sizes, k=k, rotate_corpus=rot, base_seed=seed, num_replicas=world, rank=rank
    )


def _epoch_set(world=1, **kw):
    """Full (single-rank) set of global indices for the current epoch."""
    s = _sampler(world=world, **kw)
    return s


def _indices_for_epoch(epoch, **kw):
    s = _sampler(**kw)
    s.set_epoch(epoch)
    return list(iter(s))


def _m4_part(indices):
    return {i for i in indices if i >= M4_OFFSET}


# ── (a) every epoch contains ALL static (FMA+Jam) globals, exactly once ──────
def test_all_static_corpora_present_every_epoch():
    for e in range(2 * K):
        idx = _indices_for_epoch(e)
        static = [i for i in idx if i < M4_OFFSET]
        assert set(static) == FMA_GLOBALS | JAM_GLOBALS      # all present
        assert len(static) == len(FMA_GLOBALS | JAM_GLOBALS)  # exactly once each


# ── (b) rotating-corpus chunks of consecutive epochs are disjoint ────────────
def test_consecutive_m4_chunks_disjoint():
    for e in range(2 * K):
        a = _m4_part(_indices_for_epoch(e))
        b = _m4_part(_indices_for_epoch(e + 1))
        assert a and b
        assert a.isdisjoint(b), f"epochs {e},{e+1} share M4 tracks"


# ── (c) the chunked region is fully covered, disjointly, over K epochs ───────
def test_m4_full_coverage_over_k_epochs():
    seen = []
    for e in range(K):
        seen.extend(_m4_part(_indices_for_epoch(e)))
    chunk_size = SIZES[ROT] // K
    assert len(seen) == K * chunk_size                # no overlaps across the K chunks
    assert len(set(seen)) == K * chunk_size           # all distinct
    assert set(seen) <= M4_GLOBALS                    # within the rotating corpus


# ── (d) deterministic given the seed ─────────────────────────────────────────
def test_determinism_same_seed():
    for e in (0, 3, 7):
        assert _indices_for_epoch(e, seed=SEED) == _indices_for_epoch(e, seed=SEED)
    # different seed → different ordering/chunk selection somewhere
    same = all(_indices_for_epoch(e, seed=1) == _indices_for_epoch(e, seed=2) for e in range(K))
    assert not same


# ── (e) DDP ranks partition the padded epoch list with equal length ──────────
def test_ddp_partition_equal_and_complete():
    world = 4
    epoch = 1
    shards = []
    for r in range(world):
        s = _sampler(world=world, rank=r)
        s.set_epoch(epoch)
        shards.append(list(iter(s)))

    lengths = [len(sh) for sh in shards]
    assert len(set(lengths)) == 1                      # equal length on every rank
    assert lengths[0] == len(_sampler(world=world, rank=0))   # == __len__

    # strided shards partition the padded list → union covers every unique epoch index
    union = set().union(*shards)
    ref = _sampler(world=1, rank=0)
    ref.set_epoch(epoch)
    assert union == set(iter(ref))                     # all unique indices represented
    assert union == FMA_GLOBALS | JAM_GLOBALS | _m4_part(list(iter(ref)))

    # pairwise position-disjoint (no index served twice except via intended padding):
    total_items = sum(lengths)
    assert total_items == world * lengths[0]


# ── (f) per-rank length is constant across epochs (stable scheduler steps) ────
def test_len_constant_across_epochs():
    s = _sampler(world=3, rank=0)
    lens = []
    for e in range(2 * K):
        s.set_epoch(e)
        lens.append(len(list(iter(s))))
        assert len(s) == lens[-1]
    assert len(set(lens)) == 1


# ── (g) no-rotation mode: all corpora in full every epoch, no chunking ───────
def test_no_rotation_mode():
    s = _sampler(rot=None)
    s.set_epoch(5)
    idx = list(iter(s))
    assert set(idx) == FMA_GLOBALS | JAM_GLOBALS | M4_GLOBALS
    assert s.current_chunk == -1
    assert len(idx) == sum(SIZES)


# ── (h) remainder: indivisible rotating size → equal chunks, remainder dropped ─
def test_remainder_chunks_equal_size():
    sizes = [10, 6, 9]                                  # 9 not divisible by K=4 → chunk 2, drop 1
    s = _sampler(sizes=sizes, k=4, rot=2)
    chunk_sizes = []
    seen = set()
    for e in range(4):
        s.set_epoch(e)
        m4 = {i for i in iter(s) if i >= sizes[0] + sizes[1]}
        chunk_sizes.append(len(m4))
        seen |= m4
    assert chunk_sizes == [2, 2, 2, 2]                 # all equal (floor), remainder dropped
    assert len(seen) == 8                              # 4*2 distinct, one M4 track never seen


# ── input validation ─────────────────────────────────────────────────────────
def test_invalid_args_raise():
    with pytest.raises(ValueError):
        MultiCorpusRotatingSampler([10, 0, 8], rotate_corpus=2)      # zero-size corpus
    with pytest.raises(ValueError):
        MultiCorpusRotatingSampler([10, 6, 8], rotate_corpus=5)      # bad rotate index
    with pytest.raises(ValueError):
        MultiCorpusRotatingSampler([10, 6, 2], k=4, rotate_corpus=2)  # chunk size 0
    with pytest.raises(ValueError):
        MultiCorpusRotatingSampler([10, 6, 8], num_replicas=2, rank=5)  # bad rank
