# =============================================================================
# Tests for the DAC discriminator wrapper (DACGANLoss): RpGAN / hinge / lsgan /
# sigmoid_relgan dispatch, the preprocess toggle, and backward-compatibility of
# the legacy `use_hinge` flag.
# =============================================================================

import os
import sys


import pytest
import torch

from ar_spectra.models.discriminators import DACGANLoss, DACDiscriminator
from ar_spectra.models.discriminators.types import get_relativistic_losses

SR = 44100


def _make(loss_type, **kw):
    # Tiny config: 1 MRD window + 2 MPD periods → fast on CPU/login node.
    return DACGANLoss(
        channels=1, sample_rate=SR, fft_sizes=[512], periods=[2, 3],
        loss_type=loss_type, **kw,
    )


@pytest.mark.parametrize("loss_type", ["hinge", "rpgan", "sigmoid_relgan", "lsgan"])
def test_loss_finite_and_grad(loss_type):
    torch.manual_seed(0)
    gan = _make(loss_type)
    reals = torch.randn(2, 1, 8192)
    fakes = torch.randn(2, 1, 8192, requires_grad=True)
    dis, adv, fm = gan.loss(reals=reals, fakes=fakes)
    for t in (dis, adv, fm):
        assert t.ndim == 0, f"expected scalar, got shape {tuple(t.shape)}"
        assert torch.isfinite(t), f"{loss_type}: non-finite loss"
    # Generator path must backprop into the generated waveform.
    adv.backward()
    assert fakes.grad is not None and torch.isfinite(fakes.grad).all()


def test_rpgan_matches_relativistic_helper():
    """The dispatched dis-loss must equal the per-disc averaged relativistic helper."""
    torch.manual_seed(1)
    gan = _make("rpgan").eval()  # weight_norm convs are deterministic (no BN/dropout)
    reals = torch.randn(2, 1, 8192)
    fakes = torch.randn(2, 1, 8192)
    with torch.no_grad():
        d_real = gan.discriminator(reals)
        d_fake = gan.discriminator(fakes)
        ref = sum(
            get_relativistic_losses(fr[-1], fg[-1])[0]
            for fr, fg in zip(d_real, d_fake)
        ) / len(d_fake)
        dis, _, _ = gan.loss(reals=reals, fakes=fakes)
    assert torch.allclose(dis, ref, atol=1e-5)
    assert dis.item() >= 0.0  # softplus ⇒ non-negative


def test_preprocess_toggle_changes_disc_input():
    """preprocess=True (DC-removal + peak-normalize) must alter the discriminator output."""
    torch.manual_seed(2)
    d_on = DACDiscriminator(channels=1, sample_rate=SR, fft_sizes=[512], periods=[2], preprocess=True)
    d_off = DACDiscriminator(channels=1, sample_rate=SR, fft_sizes=[512], periods=[2], preprocess=False)
    d_off.load_state_dict(d_on.state_dict())  # only preprocessing differs
    x = 3.0 * torch.randn(2, 1, 8192) + 1.0   # DC offset + scale != 1 → preprocess is a no-op only if disabled
    with torch.no_grad():
        out_on = d_on(x)[0][-1]
        out_off = d_off(x)[0][-1]
    assert out_on.shape == out_off.shape
    assert not torch.allclose(out_on, out_off, atol=1e-5)


@pytest.mark.parametrize("use_hinge,expected", [(True, "hinge"), (False, "lsgan")])
def test_backward_compat_use_hinge(use_hinge, expected):
    """Legacy `use_hinge` must map to the right loss_type and still run."""
    gan = DACGANLoss(channels=1, sample_rate=SR, fft_sizes=[512], periods=[2], use_hinge=use_hinge)
    assert gan.loss_type == expected
    reals = torch.randn(1, 1, 8192)
    fakes = torch.randn(1, 1, 8192)
    dis, adv, fm = gan.loss(reals=reals, fakes=fakes)
    assert all(torch.isfinite(t) for t in (dis, adv, fm))


def test_unknown_loss_type_raises():
    with pytest.raises(ValueError):
        DACGANLoss(channels=1, sample_rate=SR, fft_sizes=[512], periods=[2], loss_type="bogus")


@pytest.mark.parametrize("channels", [1, 2])
def test_stereo_and_mono_channels(channels):
    """N13 runs at audio_channels=2 (stereo) — the DAC disc must handle (B, C, T)."""
    torch.manual_seed(3)
    gan = DACGANLoss(channels=channels, sample_rate=SR, fft_sizes=[2048, 1024, 512],
                     periods=[2, 3, 5, 7, 11], loss_type="rpgan", preprocess=False)
    reals = torch.randn(2, channels, 8192)
    fakes = torch.randn(2, channels, 8192, requires_grad=True)
    dis, adv, fm = gan.loss(reals=reals, fakes=fakes)
    for t in (dis, adv, fm):
        assert torch.isfinite(t) and t.ndim == 0
    adv.backward()
    assert fakes.grad is not None and torch.isfinite(fakes.grad).all()
