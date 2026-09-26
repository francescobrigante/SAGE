# ===============================================================
# test_multicorpus_integration.py
#
#   End-to-end of the train.py multi-corpus path WITHOUT Lightning:
#   MultiCorpusDataset + MultiCorpusRotatingSampler + collate_stft in a
#   real DataLoader (num_workers>0, persistent_workers). Confirms batches
#   have the right shape, the per-epoch item count equals len(sampler),
#   advancing the sampler+dataset epoch rotates the M4 chunk AND moves the
#   crop window, and a single-GPU pass covers every static-corpus item.
# ===============================================================

import torch
import torchaudio
from pathlib import Path

import sage.training.data.dataset as dl
from sage.training.data.sampling import MultiCorpusRotatingSampler
from sage.training.initialization import collate_stft
from sage.training.callbacks import MultiCorpusEpochSetter
from torch.utils.data import DataLoader
from types import SimpleNamespace

SR = 44100
SEGMENT = 127 * 512
COUNTS = [5, 4, 8]            # fma-like, jam-like, m4-like(rotating); M4=8, k=4 → chunk 2
K = 4
ROT = 2


def _save_ramp(path: Path, length: int):
    ramp = torch.arange(length, dtype=torch.float32).unsqueeze(0).repeat(2, 1) / length
    torchaudio.save(str(path), ramp, SR, encoding="PCM_F", bits_per_sample=32)


def _corpus(audio_dir: Path, n: int, seed: int, partial_read: bool):
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _save_ramp(audio_dir / f"{i:03d}.wav", length=SR * 4)
    return dl.OnTheFlySTFTDataset(
        audio_dir=str(audio_dir), sample_rate=SR,
        n_fft=2048, hop_length=512, win_length=2048, target_frames=128,
        extensions=[".wav"], stereo=True, cac=True,
        skip_failed_samples=True, partial_read=partial_read, seed=seed,
    )


def _build(tmp_path):
    corpora = [_corpus(tmp_path / f"c{k}", COUNTS[k], 10 + k, partial_read=(k != 2))
               for k in range(3)]
    ds = dl.MultiCorpusDataset(corpora)
    sampler = MultiCorpusRotatingSampler(
        ds.corpus_sizes, k=K, rotate_corpus=ROT, base_seed=94, num_replicas=1, rank=0
    )
    return ds, sampler


def _loader(ds, sampler, workers):
    return DataLoader(
        ds, batch_size=2, num_workers=workers, sampler=sampler, shuffle=False,
        drop_last=False, collate_fn=collate_stft,
        persistent_workers=(workers > 0), prefetch_factor=(2 if workers > 0 else None),
    )


# ── batches have the right shapes and the epoch yields len(sampler) items ────
def test_dataloader_batches_and_count(tmp_path):
    ds, sampler = _build(tmp_path)
    sampler.set_epoch(0)
    loader = _loader(ds, sampler, workers=0)
    n_items = 0
    for S, wav in loader:
        assert S.shape[1:] == (4, 1025, 128)      # cac stereo STFT (4, F=n_fft//2+1, T=128)
        assert wav.shape[1:] == (2, SEGMENT)
        n_items += S.shape[0]
    assert n_items == len(sampler)                # every scheduled item delivered


# ── advancing the epoch rotates the M4 chunk (different M4 files seen) ────────
def test_epoch_rotates_m4_chunk(tmp_path):
    ds, sampler = _build(tmp_path)
    m4_off = COUNTS[0] + COUNTS[1]
    seen = []
    for e in range(K):
        sampler.set_epoch(e)
        idx = list(iter(sampler))
        seen.append({i for i in idx if i >= m4_off})
    for e in range(K - 1):
        assert seen[e].isdisjoint(seen[e + 1])    # consecutive epochs: disjoint M4
    assert len(set().union(*seen)) == K * (COUNTS[2] // K)   # full M4 coverage over K


# ── persistent workers: epoch advance moves the crop window (WS1 bug guard) ──
def test_persistent_workers_crop_varies(tmp_path):
    ds, sampler = _build(tmp_path)
    cb = MultiCorpusEpochSetter()
    loader = _loader(ds, sampler, workers=2)

    def first_wav(epoch):
        # drive both sampler+dataset epoch exactly like the callback does in training
        cb.on_train_epoch_start(
            SimpleNamespace(train_dataloader=loader, current_epoch=epoch), pl_module=None
        )
        for _S, wav in loader:
            return wav[0].clone()

    a = first_wav(0)
    b = first_wav(1)
    assert not torch.equal(a, b)                  # crop window moved across epochs


# ── single pass covers every static-corpus (FMA+Jam) item exactly once ───────
def test_static_corpora_fully_covered(tmp_path):
    ds, sampler = _build(tmp_path)
    sampler.set_epoch(2)
    m4_off = COUNTS[0] + COUNTS[1]
    idx = [i for i in iter(sampler) if i < m4_off]
    assert sorted(idx) == list(range(m4_off))     # all FMA+Jam globals, once each
