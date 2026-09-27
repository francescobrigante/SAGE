"""
SAME-MRSTFT reconstruction-loss tests (experimental loss).

Validates the new spectral primitives:
  - adaptive_log_mag: zero on identical input, invariant to a common scaling.
  - MRSTFTSame: scalar/finite output, exactly zero on identical waveforms (all three
    sub-terms vanish), nonzero on differing waveforms, mono + stereo (mid/side) paths,
    and that it can be added to a run through loss_config.extra.
"""

import torch

from sage.nn.losses.experimental import adaptive_log_mag, MRSTFTSame


def test_adaptive_log_mag_zero_on_identical():
    x = torch.rand(2, 4, 33, 20).abs() + 1e-2       # (B, C, F, T) magnitudes
    assert torch.allclose(adaptive_log_mag(x, x), torch.zeros(()), atol=1e-7)


def test_adaptive_log_mag_scale_invariant():
    # Scaling BOTH magnitudes by the same factor leaves the σ-normalized loss unchanged.
    x = torch.rand(2, 4, 33, 20).abs() + 1e-2
    y = torch.rand(2, 4, 33, 20).abs() + 1e-2
    base = adaptive_log_mag(x, y)
    scaled = adaptive_log_mag(7.5 * x, 7.5 * y)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_mrstft_same_zero_on_identical_waveform():
    loss = MRSTFTSame(sample_rate=44100)
    wav = torch.randn(2, 2, 8192)                   # (B, C, N) stereo
    out = loss(wav, wav.clone())
    assert out.ndim == 0 and torch.isfinite(out)
    assert torch.allclose(out, torch.zeros(()), atol=1e-5)   # SC, log-mag, IF/GD all vanish


def test_mrstft_same_positive_on_difference():
    loss = MRSTFTSame(sample_rate=44100)
    a = torch.randn(2, 2, 8192)
    b = torch.randn(2, 2, 8192)
    out = loss(a, b)
    assert torch.isfinite(out) and out > 0.0


def test_mrstft_same_mono_path():
    loss = MRSTFTSame(sample_rate=44100)
    a = torch.randn(2, 1, 8192)
    out = loss(a, a.clone())
    assert torch.isfinite(out) and torch.allclose(out, torch.zeros(()), atol=1e-5)


def test_mrstft_same_mid_side_expands_channels():
    # ms_lr stereo → L/R/mid/side (4 channels); disabling keeps 2.
    loss_ms = MRSTFTSame(sample_rate=44100, ms_lr=True)
    loss_lr = MRSTFTSame(sample_rate=44100, ms_lr=False)
    wav = torch.randn(1, 2, 4096)
    assert loss_ms._to_channels(wav).shape[1] == 4
    assert loss_lr._to_channels(wav).shape[1] == 2


def test_mrstft_same_backprop():
    loss = MRSTFTSame(sample_rate=44100)
    a = torch.randn(2, 2, 8192, requires_grad=True)
    b = torch.randn(2, 2, 8192)
    out = loss(a, b)
    out.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_mrstft_same_registers_in_loss_manager():
    # Not a paper term: it enters a run through loss_config.extra.
    from sage.training.loss_manager import LossManager

    class _DummyAE(torch.nn.Module):
        bottleneck = None

    lm = LossManager(
        autoencoder=_DummyAE(),
        sample_rate=44100,
        loss_config={"extra": [{"name": "mrstft_same_loss", "weight": 1.0, "input_key": "decoded",
                                "target_key": "reals",
                                "loss": {"_target_": "sage.nn.losses.experimental.MRSTFTSame",
                                         "fft_sizes": [256, 512]}}]},
    )
    names = [getattr(m, "name", "") for m in lm.losses_gen.losses]
    assert names == ["mrstft_same_loss"]
    assert isinstance(lm.losses_gen.losses[0].loss_module, MRSTFTSame)
