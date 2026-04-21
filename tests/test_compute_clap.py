# ===============================================================
# test_compute_clap.py — Unit tests for CLAP cosine similarity
# evaluation logic. Tests cosine_sim edge cases, load_pooled_embedding,
# and the per-pair scoring loop — no CLAP model weights required.
# ===============================================================
import sys
import csv
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from compute_clap_score import cosine_sim
from eval_dataloader import load_pooled_embedding


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


# ── T2: load_pooled_embedding ─────────────────────────────────────────────────

def test_load_pooled_embedding_mean_pool(tmp_path):
    """Multi-chunk (T, D) embedding should be mean-pooled to (D,)."""
    emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    path = tmp_path / "emb.npy"
    np.save(path, emb)

    pooled = load_pooled_embedding(path)
    expected = emb.mean(axis=0)  # [3.0, 4.0]
    assert pooled.shape == (2,)
    assert np.allclose(pooled, expected)


def test_load_pooled_embedding_single_chunk(tmp_path):
    """Single-chunk (1, D) embedding should pass through (still mean-pooled, same result)."""
    emb = np.array([[7.0, 8.0, 9.0]])  # (1, 3)
    path = tmp_path / "single.npy"
    np.save(path, emb)

    pooled = load_pooled_embedding(path)
    assert pooled.shape == (3,)
    assert np.allclose(pooled, [7.0, 8.0, 9.0])


def test_load_pooled_embedding_1d_array(tmp_path):
    """1-D array (D,) saved as .npy should be handled via atleast_2d → (1, D) → mean → (D,)."""
    emb = np.array([1.0, 2.0, 3.0, 4.0])  # (4,)
    path = tmp_path / "flat.npy"
    np.save(path, emb)

    pooled = load_pooled_embedding(path)
    assert pooled.shape == (4,)
    assert np.allclose(pooled, emb)


# ── T3: end-to-end scoring loop (mocked CLAP) ────────────────────────────────

def test_clap_scoring_loop_correct_scores(tmp_path):
    """
    Simulate the per-pair scoring loop from compute_clap_score.py:
    load cached embeddings → pool → cosine_sim → aggregate.
    """
    D = 512
    rng = np.random.RandomState(0)
    n_pairs = 5

    # Create cached target and pred embeddings
    target_embs = []
    pred_embs = []
    target_cache_dir = tmp_path / "model_name" / "target"
    pred_cache_dir = tmp_path / "model_name" / "preds_ckpt"
    target_cache_dir.mkdir(parents=True)
    pred_cache_dir.mkdir(parents=True)

    expected_scores = []
    for i in range(n_pairs):
        # Random embeddings (n_chunks, D)
        t_emb = rng.randn(3, D).astype(np.float32)
        p_emb = rng.randn(3, D).astype(np.float32)
        target_embs.append(t_emb)
        pred_embs.append(p_emb)

        t_path = target_cache_dir / f"track_{i:03d}.npy"
        p_path = pred_cache_dir / f"track_{i:03d}.npy"
        np.save(t_path, t_emb)
        np.save(p_path, p_emb)

        # Expected: pooled cosine
        t_pooled = t_emb.mean(axis=0)
        p_pooled = p_emb.mean(axis=0)
        expected_scores.append(cosine_sim(t_pooled, p_pooled))

    # Simulate the scoring loop
    actual_scores = []
    for i in range(n_pairs):
        t_path = target_cache_dir / f"track_{i:03d}.npy"
        p_path = pred_cache_dir / f"track_{i:03d}.npy"
        s = cosine_sim(load_pooled_embedding(t_path), load_pooled_embedding(p_path))
        actual_scores.append(s)

    for i, (exp, act) in enumerate(zip(expected_scores, actual_scores)):
        assert act == pytest.approx(exp, abs=1e-6), (
            f"Pair {i}: expected {exp:.6f}, got {act:.6f}"
        )


def test_clap_identical_embeddings_score_one(tmp_path):
    """Identical target and pred embeddings should produce cosine = 1.0."""
    D = 512
    emb = np.random.RandomState(42).randn(4, D).astype(np.float32)

    t_path = tmp_path / "same_t.npy"
    p_path = tmp_path / "same_p.npy"
    np.save(t_path, emb)
    np.save(p_path, emb)

    s = cosine_sim(load_pooled_embedding(t_path), load_pooled_embedding(p_path))
    assert s == pytest.approx(1.0, abs=1e-6)


# ── T4: CSV round-trip ────────────────────────────────────────────────────────

def test_clap_csv_output(tmp_path):
    """Verify CLAP results can be written and read back from CSV correctly."""
    results = [
        {"target_file": "track_001.wav", "pred_file": "track_001.wav", "clap_music": 0.85, "clap_audio": 0.72},
        {"target_file": "track_002.wav", "pred_file": "track_002.wav", "clap_music": 0.91, "clap_audio": 0.88},
    ]
    csv_path = tmp_path / "clap_scores.csv"
    fieldnames = ["target_file", "pred_file", "clap_music", "clap_audio"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(fieldnames)
    assert float(rows[0]["clap_music"]) == pytest.approx(0.85)
    assert float(rows[1]["clap_audio"]) == pytest.approx(0.88)
