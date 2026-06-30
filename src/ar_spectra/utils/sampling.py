# ===================================================================
# Multi-Corpus Rotating Sampler
#
#   deterministic DDP-aware sampler over a MultiCorpusDataset: every
#   epoch sees all of the non-rotating corpora and one disjoint chunk
#   of the rotating corpus (M4Singer), rotating through K chunks.
# ===================================================================

"""
``MultiCorpusRotatingSampler`` emits **global** indices into a
``MultiCorpusDataset`` (the flat ``ConcatDataset`` index space).

Per epoch it yields, deterministically:

* **every** item of each non-rotating corpus (FMA-full, Jamendo) — one crop/file
  (the per-item crop *window* still changes each epoch via the dataset's epoch seed);
* one of **K disjoint chunks** of the rotating corpus (M4Singer), selected by
  ``epoch % K``. The chunking is a single frozen permutation sliced into K equal
  contiguous blocks, so consecutive epochs never share rotating-corpus tracks and
  the chunked region is fully covered every K epochs.

The per-epoch index list is shuffled (so corpora interleave) and then sharded
across DDP ranks. The shard length is **constant across epochs** (the rotating
chunk has fixed size), giving a stable ``steps_per_epoch`` for the LR scheduler.

DDP note: pass this as ``sampler=`` and set ``use_distributed_sampler=False`` on the
Lightning Trainer, else Lightning wraps it in a ``DistributedSampler`` and the
rotation breaks.
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from config import DEFAULT_SEED


def _resolve_ddp(num_replicas: Optional[int], rank: Optional[int]) -> tuple[int, int]:
    """Fill in (num_replicas, rank) from the active process group when not given."""
    initialized = dist.is_available() and dist.is_initialized()
    if num_replicas is None:
        num_replicas = dist.get_world_size() if initialized else 1
    if rank is None:
        rank = dist.get_rank() if initialized else 0
    if not (0 <= rank < num_replicas):
        raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")
    return num_replicas, rank


class MultiCorpusRotatingSampler(Sampler[int]):
    """Deterministic, DDP-aware rotating sampler over concatenated corpora.

    Args:
        corpus_sizes: Per-corpus item counts in ``ConcatDataset`` order
            (e.g. ``dataset.corpus_sizes``).
        k: Number of disjoint chunks to rotate the rotating corpus through.
        rotate_corpus: Index (into ``corpus_sizes``) of the corpus to rotate;
            ``None`` disables rotation (all corpora seen in full every epoch).
        base_seed: Seed for the frozen rotation permutation and the per-epoch shuffle.
        num_replicas: DDP world size; auto-detected from the process group if ``None``.
        rank: DDP rank; auto-detected if ``None``.
    """

    def __init__(
        self,
        corpus_sizes: Sequence[int],
        k: int = 4,
        rotate_corpus: Optional[int] = None,
        base_seed: int = DEFAULT_SEED,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if any(n <= 0 for n in corpus_sizes):
            raise ValueError(f"corpus_sizes must be positive, got {corpus_sizes}")
        if rotate_corpus is not None and not (0 <= rotate_corpus < len(corpus_sizes)):
            raise ValueError(f"rotate_corpus {rotate_corpus} out of range for {len(corpus_sizes)} corpora")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.corpus_sizes = list(corpus_sizes)            # per-corpus item counts (ConcatDataset order)
        self.k = int(k)                                    # number of rotation chunks
        self.rotate_corpus = rotate_corpus                 # index of the rotating corpus, or None
        self.base_seed = int(base_seed)                    # seed for permutation + epoch shuffle
        self.num_replicas, self.rank = _resolve_ddp(num_replicas, rank)
        self.epoch = 0                                     # advanced via set_epoch each train epoch

        # Global offset of each corpus in the flat ConcatDataset index space.
        self._offsets: List[int] = []
        acc = 0
        for n in self.corpus_sizes:
            self._offsets.append(acc)
            acc += n

        # Static global indices of every non-rotating corpus (seen in full every epoch).
        self._static_globals: List[int] = []
        for c, n in enumerate(self.corpus_sizes):
            if c == self.rotate_corpus:
                continue
            off = self._offsets[c]
            self._static_globals.extend(range(off, off + n))

        # Frozen rotation: one permutation of the rotating corpus, sliced into K equal
        # contiguous chunks (remainder dropped so every chunk — and hence every epoch —
        # has the same size, keeping steps_per_epoch constant).
        self._chunks: List[List[int]] = []
        if self.rotate_corpus is not None:
            n_rot = self.corpus_sizes[self.rotate_corpus]
            chunk_size = n_rot // self.k
            if chunk_size == 0:
                raise ValueError(
                    f"rotating corpus size {n_rot} too small for k={self.k} (chunk size 0)"
                )
            g = torch.Generator().manual_seed(self.base_seed)
            perm = torch.randperm(n_rot, generator=g).tolist()
            off = self._offsets[self.rotate_corpus]
            for j in range(self.k):
                local = perm[j * chunk_size:(j + 1) * chunk_size]
                self._chunks.append([off + i for i in local])
            self._chunk_size = chunk_size
        else:
            self._chunk_size = 0

        # Total (pre-shard) items per epoch and the resulting constant per-rank length.
        self._epoch_total = len(self._static_globals) + self._chunk_size
        self._per_rank_len = math.ceil(self._epoch_total / self.num_replicas)

    # ── epoch wiring ─────────────────────────────────────────────────────────
    def set_epoch(self, epoch: int) -> None:
        """Select the rotation chunk and reshuffle for ``epoch``."""
        self.epoch = int(epoch)

    @property
    def current_chunk(self) -> int:
        """Index of the rotating-corpus chunk used this epoch (``-1`` if no rotation)."""
        return self.epoch % self.k if self.rotate_corpus is not None else -1

    # ── index construction ───────────────────────────────────────────────────
    def _epoch_global_indices(self) -> List[int]:
        """Full (pre-shard) shuffled global index list for the current epoch."""
        indices = list(self._static_globals)
        if self.rotate_corpus is not None:
            indices.extend(self._chunks[self.current_chunk])
        # Deterministic per-epoch shuffle so corpora interleave (seed varies with epoch).
        g = torch.Generator().manual_seed(self.base_seed ^ (self.epoch + 1))
        order = torch.randperm(len(indices), generator=g).tolist()
        return [indices[i] for i in order]

    def _sharded_indices(self) -> List[int]:
        """This rank's slice of the padded epoch list (equal length on every rank)."""
        indices = self._epoch_global_indices()
        total = self._per_rank_len * self.num_replicas
        if len(indices) < total:                           # pad by repeating from the front
            pad = total - len(indices)
            indices = indices + indices[:pad]
        return indices[self.rank:total:self.num_replicas]  # strided partition (DistributedSampler-style)

    def __iter__(self):
        return iter(self._sharded_indices())

    def __len__(self) -> int:
        return self._per_rank_len
