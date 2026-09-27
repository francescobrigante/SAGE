# ===============
# Smoke tests of the SOTA baselines (extra `[baselines]`): the packages import, every adapter of
# Tables 2/4 is registered, and the channel-axis normalisation is right. Building an adapter
# downloads its weights, so the reconstruction round-trip is behind the `weights` marker.
# ===============
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

BASELINES = ["codicodec", "music2latent", "sao-vae", "same", "same-s"]


def test_baseline_packages_import():
    pytest.importorskip("music2latent", reason="extra [baselines] not installed")
    import codicodec, diffusers, stable_audio_3  # noqa: F401
    from diffusers import AutoencoderOobleck  # noqa: F401  (Stable Audio Open VAE)


def test_every_paper_baseline_has_an_adapter():
    from evaluation import codecs
    for name in BASELINES + ["sage"]:
        assert name in codecs.ADAPTERS, name


@pytest.mark.parametrize("layout", ["CT", "TC", "BCT"])
def test_channel_axis_normalisation(layout):
    from evaluation.codecs import _to_channel_time
    x = np.random.default_rng(0).normal(size=(2, 1000)).astype(np.float32)
    arr = {"CT": x, "TC": x.T, "BCT": x[None]}[layout]
    out = _to_channel_time(arr, audio_channels=2)
    assert out.shape == (2, 1000) and torch.allclose(out, torch.from_numpy(x))


@pytest.mark.weights
@pytest.mark.parametrize("name", BASELINES)
def test_adapter_round_trip_with_real_weights(name):
    from evaluation.codecs import build_adapter
    ad = build_adapter(name, device="cpu")
    wav = 0.1 * torch.randn(2, 44100 * 2)
    rec = ad.reconstruct(wav)
    assert rec.shape[0] == 2 and torch.isfinite(rec).all()
