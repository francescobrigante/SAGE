# ===============================================================
# test_compute_clap.py — Unit tests for CLAP cosine similarity
# evaluation logic. Tests cosine_sim edge cases, the 10-s windowing of
# embed_clap, the embedding cache and the CSV output — no CLAP weights required.
# ===============================================================
import csv
import pytest
import numpy as np
import torch

from evaluation.metrics.clap import cosine_sim


# ── T1: cosine_sim correctness ────────────────────────────────────────────────

def test_cosine_identical_vectors():
    """Identical vectors → cosine = 1.0."""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    """Orthogonal vectors → cosine = 0.0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_sim(a, b) == pytest.approx(0.0)


def test_cosine_antiparallel_vectors():
    """Antiparallel (opposite) vectors → cosine = -1.0."""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(a, -a) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    """Zero vector → cosine = 0.0 (safe division)."""
    a = np.array([1.0, 2.0, 3.0])
    z = np.zeros(3)
    assert cosine_sim(a, z) == pytest.approx(0.0)
    assert cosine_sim(z, z) == pytest.approx(0.0)


def test_cosine_scaled_vectors_invariant():
    """Cosine similarity is scale-invariant: sim(a, k*a) = 1.0 for k > 0."""
    a = np.array([1.0, -2.0, 3.0, -4.0])
    assert cosine_sim(a, 100.0 * a) == pytest.approx(1.0)
    assert cosine_sim(a, 0.001 * a) == pytest.approx(1.0)


def test_cosine_high_dimensional():
    """Cosine works correctly in high-D (typical CLAP embedding = 512-D)."""
    rng = np.random.RandomState(42)
    a = rng.randn(512)
    b = rng.randn(512)
    expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine_sim(a, b) == pytest.approx(expected, abs=1e-6)


# ── T2: embed_clap — 10-s windows, 1-s hop, one embedding per window ────────

class _MockCLAP:
    """Stands in for laion_clap.CLAP_Module: embedding = first D samples of each window."""
    sr = 48000

    def __init__(self, D=8):
        self.D = D
        self.model = self
        self.calls = []

    def get_audio_embedding_from_data(self, x, use_tensor=True):
        self.calls.append(tuple(x.shape))
        return x[:, : self.D]


def test_embed_clap_windows_and_dtype():
    pytest.importorskip("laion_clap")
    from evaluation.metrics.clap import embed_clap
    ml = _MockCLAP()
    wav = 0.1 * torch.randn(2, 12 * ml.sr)                    # 12 s stereo at the model rate
    emb = embed_clap(ml, wav, ml.sr, "cpu")
    assert ml.calls == [(12, 10 * ml.sr)]                     # 12 windows (1-s hop), each 10 s (zero-padded)
    assert emb.shape == (12, ml.D) and emb.dtype == np.float16


# ── T3: cache + paper scoring (cosine of the window-averaged embeddings) ─────

def test_load_or_embed_computes_once_then_reads_cache(tmp_path):
    from evaluation.common import load_or_embed, target_cache_path
    calls = []

    def embed_fn(ml, wav, sr, device):
        calls.append(1)
        return np.arange(6, dtype=np.float16).reshape(3, 2)

    cache = target_cache_path(tmp_path, "clap-music", "track_000")
    first = load_or_embed(None, embed_fn, torch.zeros(2, 10), 44100, "cpu", cache)
    second = load_or_embed(None, embed_fn, torch.zeros(2, 10), 44100, "cpu", cache)
    assert len(calls) == 1 and cache.is_file()
    assert first.dtype == np.float32 and np.array_equal(first, second)


def test_clap_score_is_cosine_of_window_means():
    rng = np.random.RandomState(0)
    t, p = rng.randn(3, 512).astype(np.float32), rng.randn(3, 512).astype(np.float32)
    expected = np.dot(t.mean(0), p.mean(0)) / (np.linalg.norm(t.mean(0)) * np.linalg.norm(p.mean(0)))
    assert cosine_sim(t.mean(0), p.mean(0)) == pytest.approx(expected, abs=1e-6)
    assert cosine_sim(t.mean(0), t.mean(0)) == pytest.approx(1.0, abs=1e-6)


# ── T4: CSV output ────────────────────────────────────────────────────────────

def test_write_csv_round_trip_ignores_extra_keys(tmp_path):
    from evaluation.common import write_csv
    rows = [{"file": "a.wav", "cosine": 0.85, "debug": "x"}, {"file": "b.wav", "cosine": 0.91}]
    path = tmp_path / "sub" / "clap_music.csv"
    write_csv(path, ["file", "cosine"], rows)
    back = list(csv.DictReader(open(path)))
    assert [r["file"] for r in back] == ["a.wav", "b.wav"] and set(back[0]) == {"file", "cosine"}
    assert float(back[1]["cosine"]) == pytest.approx(0.91)
