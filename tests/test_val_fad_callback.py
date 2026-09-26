# =============================================================================
# Unit tests for ValFADCallback (CPU-only, login-node friendly).
#
# These cover the PURE logic (sharding, target-file intersection, target stats,
# Frechet-from-embeddings) plus the lazy-import contract and graceful-disable
# behaviour. The full GPU encode→decode→CLAP→all_gather pass is validated by the
# debug-queue smoke test, not here (fadtk/torchaudio live only in the venv).
# =============================================================================
import sys
import numpy as np
import pytest

from sage.training.callbacks import ValFADCallback


def test_lazy_import_isolation():
    """Merely importing the callback must NOT pull in fadtk (heavy, venv-only).

    Checked in a clean subprocess so it is independent of other tests that may
    legitimately trigger the lazy fadtk import via on_fit_start().
    """
    import subprocess
    code = (
        "import sys\n"
        "from sage.training.callbacks import ValFADCallback\n"
        "assert 'fadtk' not in sys.modules, 'fadtk leaked into module import'\n"
        "print('OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"isolation check failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"


def test_shard_world_size_1_is_full():
    files = list(range(10))
    assert ValFADCallback._shard(files, 0, 1) == files


def test_shard_disjoint_cover_world_size_2():
    files = list(range(11))  # odd count → unequal shards
    r0 = ValFADCallback._shard(files, 0, 2)
    r1 = ValFADCallback._shard(files, 1, 2)
    assert set(r0).isdisjoint(r1)
    assert sorted(r0 + r1) == files  # partition: disjoint + covers all


def test_build_file_list_intersection(tmp_path):
    """Only files with a cached target .npy survive."""
    from pathlib import Path
    tgt = tmp_path / "clap-laion-music" / "target"
    tgt.mkdir(parents=True)
    stems = ["000001", "000002", "000003", "000004", "000005"]
    files = [Path(f"/fake/{s}.mp3") for s in stems]
    for s in ("000001", "000003", "000004"):  # only 3 of 5 have cache
        np.save(tgt / f"{s}.npy", np.zeros((2, 512), dtype=np.float32))
    kept = ValFADCallback._build_file_list(files, tgt)
    assert [f.stem for f in kept] == ["000001", "000003", "000004"]


def test_target_stats_shapes_and_values(tmp_path):
    from pathlib import Path
    tgt = tmp_path / "clap-laion-music" / "target"
    tgt.mkdir(parents=True)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 512)).astype(np.float32)
    b = rng.standard_normal((5, 512)).astype(np.float32)
    np.save(tgt / "000001.npy", a)
    np.save(tgt / "000002.npy", b)
    files = [Path("/fake/000001.mp3"), Path("/fake/000002.mp3")]
    mu, cov = ValFADCallback._target_stats(files, tgt)
    expected = np.concatenate([a, b], axis=0)
    assert mu.shape == (512,)
    assert cov.shape == (512, 512)
    np.testing.assert_allclose(mu, expected.mean(0), rtol=1e-5)
    np.testing.assert_allclose(cov, np.cov(expected, rowvar=False), rtol=1e-5)


def test_frechet_from_embs_concats_and_passes_pred_stats():
    """_frechet_from_embs concatenates the per-file embedding arrays and feeds
    pred mu/cov to calc_fd, returning its float result."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((6, 8)).astype(np.float32)
    mu_t = np.zeros(8, dtype=np.float32)
    cov_t = np.eye(8, dtype=np.float32)

    captured = {}

    def fake_calc_fd(m1, c1, m2, c2):
        captured["args"] = (m1, c1, m2, c2)
        return 42.0

    out = ValFADCallback._frechet_from_embs(mu_t, cov_t, [a, b], fake_calc_fd)
    assert out == 42.0
    all_p = np.concatenate([a, b], axis=0)
    np.testing.assert_allclose(captured["args"][0], mu_t)
    np.testing.assert_allclose(captured["args"][1], cov_t)
    np.testing.assert_allclose(captured["args"][2], all_p.mean(0), rtol=1e-5)
    np.testing.assert_allclose(captured["args"][3], np.cov(all_p, rowvar=False), rtol=1e-5)


def test_frechet_identical_sets_is_zero():
    """FAD of a distribution against itself ≈ 0 (using a real Frechet impl)."""
    sqrtm = pytest.importorskip("scipy.linalg").sqrtm

    def real_fd(m1, c1, m2, c2):
        diff = m1 - m2
        covmean = sqrtm(c1 @ c2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))

    rng = np.random.default_rng(2)
    emb = rng.standard_normal((50, 8)).astype(np.float32)
    mu, cov = emb.mean(0), np.cov(emb, rowvar=False)
    score = ValFADCallback._frechet_from_embs(mu, cov, [emb], real_fd)
    assert abs(score) < 1e-3


class _DummyTrainer:
    def __init__(self):
        self.world_size = 1
        self.global_rank = 0
        self.current_epoch = 0
        self.sanity_checking = False


def test_disabled_is_noop():
    """enabled=False → hooks return immediately, never become ready, never raise."""
    cb = ValFADCallback(cache_dir="/nope", fma_csv_path="", audio_root="/nope", enabled=False)
    cb.on_fit_start(_DummyTrainer(), object())
    cb.on_validation_epoch_end(_DummyTrainer(), object())  # _ready is False → no-op
    assert cb._ready is False


def test_failure_disables_gracefully(tmp_path):
    """enabled=True but deps/paths unavailable (fadtk absent on login node) →
    on_fit_start disables itself instead of raising into training."""
    cb = ValFADCallback(
        cache_dir=str(tmp_path), fma_csv_path="", audio_root=str(tmp_path), enabled=True
    )

    class _PLM:
        device = "cpu"

        class autoencoder:  # noqa: N801
            encoder = None

    cb.on_fit_start(_DummyTrainer(), _PLM())
    assert cb._ready is False
    # subsequent validation hook must also be a safe no-op
    cb.on_validation_epoch_end(_DummyTrainer(), _PLM())
