# ===============================================================
# test_compute_fad.py — Unit tests for FAD computation logic.
# Tests compute_stats (μ, Σ), calc_frechet_distance math, cache
# path resolution, and CSV output — using synthetic embeddings,
# no real audio model weights required.
# ===============================================================
import sys
import csv
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from compute_fad import compute_stats
from fadtk.fad import calc_frechet_distance
from unittest.mock import MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

D = 16  # embedding dimension used across tests


def _make_fad_mock(model_name: str = "test_model") -> MagicMock:
    """Create a minimal mock of fadtk.FrechetAudioDistance with ml.name."""
    fad = MagicMock()
    fad.ml.name = model_name
    return fad


def _create_cached_embeddings(
    cache_dir: Path,
    model_name: str,
    subfolder: str,
    stems: list[str],
    rng: np.random.RandomState,
    n_chunks: int = 3,
) -> tuple[list[Path], list[np.ndarray]]:
    """Write fake .npy embeddings and return (audio_file_paths, raw_arrays)."""
    emb_dir = cache_dir / model_name / subfolder
    emb_dir.mkdir(parents=True, exist_ok=True)

    files = []
    arrays = []
    for stem in stems:
        # Create a dummy audio path (only stem matters for matching)
        audio_file = cache_dir / f"{stem}.flac"
        audio_file.touch()
        files.append(audio_file)

        # Embedding: (n_chunks, D) — simulates chunked output from batch_embed_files
        emb = rng.randn(n_chunks, D).astype(np.float32)
        np.save(emb_dir / f"{stem}.npy", emb)
        arrays.append(emb)

    return files, arrays


# ── T1: compute_stats produces correct μ and Σ ───────────────────────────────

def test_compute_stats_mean_correct(tmp_path):
    """Mean of concatenated embeddings should match numpy reference."""
    rng = np.random.RandomState(0)
    fad = _make_fad_mock()
    files, arrays = _create_cached_embeddings(
        tmp_path, fad.ml.name, "target", ["a", "b", "c"], rng
    )

    mu, cov = compute_stats(fad, files, cache_dir=tmp_path, subfolder="target")

    all_embs = np.concatenate(arrays, axis=0)  # (9, D)
    expected_mu = all_embs.mean(axis=0)

    assert mu is not None
    assert np.allclose(mu, expected_mu, atol=1e-6), (
        f"Mean mismatch: max diff = {np.abs(mu - expected_mu).max():.2e}"
    )


def test_compute_stats_cov_correct(tmp_path):
    """Covariance of concatenated embeddings should match numpy reference."""
    rng = np.random.RandomState(1)
    fad = _make_fad_mock()
    files, arrays = _create_cached_embeddings(
        tmp_path, fad.ml.name, "target", ["x", "y"], rng
    )

    mu, cov = compute_stats(fad, files, cache_dir=tmp_path, subfolder="target")

    all_embs = np.concatenate(arrays, axis=0)  # (6, D)
    expected_cov = np.cov(all_embs, rowvar=False)

    assert cov is not None
    assert cov.shape == (D, D)
    assert np.allclose(cov, expected_cov, atol=1e-5), (
        f"Covariance mismatch: max diff = {np.abs(cov - expected_cov).max():.2e}"
    )


def test_compute_stats_returns_none_for_missing_cache(tmp_path):
    """If no cached embeddings exist, compute_stats must return (None, None)."""
    fad = _make_fad_mock("nonexistent_model")
    files = [tmp_path / "fake_file.flac"]
    files[0].touch()

    mu, cov = compute_stats(fad, files, cache_dir=tmp_path, subfolder="target")
    assert mu is None and cov is None


# ── T2: calc_frechet_distance math ───────────────────────────────────────────

def test_fad_identical_distributions_is_zero():
    """FAD between identical Gaussians should be ~0."""
    rng = np.random.RandomState(42)
    mu = rng.randn(D)
    cov = np.eye(D) + rng.randn(D, D) * 0.1
    cov = cov @ cov.T  # ensure positive definite

    fad_score = calc_frechet_distance(mu, cov, mu, cov)
    assert abs(fad_score) < 1e-6, f"FAD(same, same) should be ~0, got {fad_score:.2e}"


def test_fad_different_means_is_positive():
    """FAD between Gaussians with different means should be > 0."""
    mu1 = np.zeros(D)
    mu2 = np.ones(D) * 3.0
    cov = np.eye(D)

    fad_score = calc_frechet_distance(mu1, cov, mu2, cov)
    assert fad_score > 0, f"FAD should be positive for different means, got {fad_score:.6f}"
    # Analytical: ||μ1 - μ2||² = D * 9 = 144 (for D=16), plus trace term = 0
    expected = np.sum((mu1 - mu2) ** 2)
    assert abs(fad_score - expected) < 1e-3, (
        f"FAD for same cov should be ||Δμ||²={expected}, got {fad_score:.6f}"
    )


def test_fad_different_covariances_is_positive():
    """FAD between Gaussians with same mean but different covariances should be > 0."""
    mu = np.zeros(D)
    cov1 = np.eye(D)
    cov2 = np.eye(D) * 4.0  # scaled identity

    fad_score = calc_frechet_distance(mu, cov1, mu, cov2)
    assert fad_score > 0, f"FAD should be positive for different covariances, got {fad_score:.6f}"


# ── T3: end-to-end FAD pipeline (mocked) ─────────────────────────────────────

def test_end_to_end_fad_pipeline(tmp_path):
    """
    Full pipeline: create cached embeddings for target and pred, compute stats,
    compute FAD. Identical distributions → FAD ≈ 0. Different → FAD > 0.
    """
    rng = np.random.RandomState(99)
    fad = _make_fad_mock("pipeline_model")
    stems = [f"track_{i:03d}" for i in range(10)]

    # Target embeddings
    target_files, target_arrays = _create_cached_embeddings(
        tmp_path, fad.ml.name, "target", stems, rng, n_chunks=5
    )

    # Pred embeddings — SAME distribution (re-seeded identically)
    rng_same = np.random.RandomState(99)
    pred_files_same, _ = _create_cached_embeddings(
        tmp_path, fad.ml.name, "preds_same", stems, rng_same, n_chunks=5
    )

    mu_t, cov_t = compute_stats(fad, target_files, cache_dir=tmp_path, subfolder="target")
    mu_p, cov_p = compute_stats(fad, pred_files_same, cache_dir=tmp_path, subfolder="preds_same")
    fad_same = calc_frechet_distance(mu_t, cov_t, mu_p, cov_p)

    assert abs(fad_same) < 1e-6, f"FAD of identical embedding sets should be ≈0, got {fad_same:.6f}"

    # Pred embeddings — DIFFERENT distribution (shifted mean)
    rng_diff = np.random.RandomState(0)
    pred_files_diff, pred_arrays_diff = _create_cached_embeddings(
        tmp_path, fad.ml.name, "preds_diff", stems, rng_diff, n_chunks=5
    )
    # Shift all embeddings by +5 to create a mean shift
    for stem in stems:
        path = tmp_path / fad.ml.name / "preds_diff" / f"{stem}.npy"
        emb = np.load(path)
        np.save(path, emb + 5.0)

    mu_p2, cov_p2 = compute_stats(fad, pred_files_diff, cache_dir=tmp_path, subfolder="preds_diff")
    fad_diff = calc_frechet_distance(mu_t, cov_t, mu_p2, cov_p2)

    assert fad_diff > 10.0, (
        f"FAD of shifted embeddings should be large, got {fad_diff:.6f}"
    )


# ── T4: CSV round-trip ────────────────────────────────────────────────────────

def test_fad_csv_output(tmp_path):
    """Verify FAD result can be written and read back from CSV."""
    fad_score = 42.123456
    csv_path = tmp_path / "fad_result.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "score"])
        writer.writeheader()
        writer.writerow({"model": "mert", "score": fad_score})

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1
    assert rows[0]["model"] == "mert"
    assert float(rows[0]["score"]) == pytest.approx(fad_score)
