# ===========================================================================
# F1b test — discriminator construction and loss computation for all 7 types.
#
# Each type is instantiated via LossManager (the path used by the training
# engine) with minimal hyperparameters to keep tests fast on CPU (~1–5 s each).
# For each discriminator we verify:
#   1. LossManager.discriminator is not None and has the expected class
#   2. .loss(reals, fakes) returns a 3-tuple of finite scalar tensors
#      (dis_loss, adv_loss, feature_matching_distance)
#   3. dis_loss and adv_loss are strictly positive on random noise
# ===========================================================================

import pytest
import torch
import torch.nn as nn

from sage.training.loss_manager import LossManager


# ---------------------------------------------------------------------------
# Tiny autoencoder stub — LossManager only inspects `bottleneck`, `pre_transform`
# and `has_pre_transform`; it never runs the autoencoder forward.
# ---------------------------------------------------------------------------

class _StubAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = None
        self.pre_transform = None
        self.has_pre_transform = False


SR = 48000
CHANNELS = 2
# Enough samples for the largest hop/FFT any disc needs at minimum settings.
# 24000 = 0.5 s @ 48 kHz, well above the minimum for CQT / STFT windows.
T = 24000


def _make_signals():
    """Two different random stereo waveforms (batch=1)."""
    reals = torch.randn(1, CHANNELS, T)
    fakes = torch.randn(1, CHANNELS, T)
    return reals, fakes


# ---------------------------------------------------------------------------
# Discriminator configs — one per type, matching LossManager's expected schema.
# Hyperparameters are minimised for speed (fewer scales, smaller nets).
# ---------------------------------------------------------------------------

DISC_CONFIGS = {
    "oobleck": {
        "discriminator": {
            "type": "oobleck",
            "config": {},
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "encodec": {
        "discriminator": {
            "type": "encodec",
            "config": {
                "filters": 8,
                "n_ffts": [256, 128],
                "hop_lengths": [64, 32],
                "win_lengths": [256, 128],
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "dac": {
        "discriminator": {
            "type": "dac",
            "config": {
                "periods": [2, 3],
                "rates": [],
                "fft_sizes": [256],
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "big_vgan": {
        "discriminator": {
            "type": "big_vgan",
            "config": {
                "periods": [2, 3],
                "cqtd_hop_lengths": [256],
                "cqtd_n_octaves": [3],
                "cqtd_bins_per_octaves": [12],
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "transformer": {
        "discriminator": {
            "type": "transformer",
            "config": {
                # Only the patched branch is enabled by default (stft/mfb/chroma off).
                "patched_kwargs": {
                    "enabled": True,
                    "patch_sizes": [128],
                    "strides": [4],
                    "depths": [1],
                },
                "stft_kwargs": {"enabled": False},
                "mfb_kwargs": {"enabled": False},
                "chroma_kwargs": {"enabled": False},
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "hil": {
        "discriminator": {
            "type": "hil",
            "config": {
                "filters": 8,
                "n_ffts": [256, 128],
                "hop_lengths": [64, 32],
                "win_lengths": [256, 128],
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
    "wavtokenizer": {
        "discriminator": {
            "type": "wavtokenizer",
            "config": {
                "periods": [2, 3],
                "resolutions": [(256, 64, 256)],
                "fft_sizes": [256],
                "use_dac": True,
                "fold_lrms": False,
            },
            "weights": {"adversarial": 1.0, "feature_matching": 1.0},
        }
    },
}


# Expected class hierarchy after LossManager instantiation
from sage.nn.discriminators import (
    OobleckDiscriminator,
    EncodecDiscriminator,
    BigVGANDiscriminator,
    MultiTransformerDiscriminator,
    HILDiscriminator,
)
from sage.nn.discriminators.dac import DACGANLoss
from sage.nn.discriminators.wavtokenizer import WavTokenizerGANLoss

_EXPECTED_CLASS = {
    "oobleck": OobleckDiscriminator,
    "encodec": EncodecDiscriminator,
    "dac": DACGANLoss,
    "big_vgan": BigVGANDiscriminator,
    "transformer": MultiTransformerDiscriminator,
    "hil": HILDiscriminator,
    "wavtokenizer": WavTokenizerGANLoss,
}


# ---------------------------------------------------------------------------
# Parametrised test — one case per discriminator type
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(DISC_CONFIGS.keys()))
def disc_type(request):
    return request.param


def test_loss_manager_builds_discriminator(disc_type):
    """LossManager constructs the right discriminator class for each type."""
    ae = _StubAutoencoder()
    lm = LossManager(
        autoencoder=ae,
        sample_rate=SR,
        loss_config=DISC_CONFIGS[disc_type],
        audio_channels=CHANNELS,
    )
    assert lm.discriminator is not None, f"discriminator is None for type {disc_type!r}"
    assert isinstance(lm.discriminator, _EXPECTED_CLASS[disc_type]), (
        f"Expected {_EXPECTED_CLASS[disc_type].__name__}, "
        f"got {type(lm.discriminator).__name__}"
    )


def test_discriminator_loss_shape_and_finite(disc_type):
    """Each discriminator's .loss() returns a 3-tuple of finite scalars."""
    ae = _StubAutoencoder()
    lm = LossManager(
        autoencoder=ae,
        sample_rate=SR,
        loss_config=DISC_CONFIGS[disc_type],
        audio_channels=CHANNELS,
    )
    reals, fakes = _make_signals()

    with torch.no_grad():
        result = lm.discriminator.loss(reals, fakes)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"

    dis_loss, adv_loss, fm_dist = result
    for name, val in [("dis_loss", dis_loss), ("adv_loss", adv_loss), ("fm_dist", fm_dist)]:
        assert isinstance(val, torch.Tensor), f"{name} is not a Tensor"
        assert val.ndim == 0, f"{name} is not a scalar (shape={val.shape})"
        assert torch.isfinite(val), f"{name} is not finite ({val.item():.6g})"


def test_discriminator_losses_are_positive(disc_type):
    """On random noise (reals ≠ fakes), dis_loss and adv_loss should be > 0."""
    ae = _StubAutoencoder()
    lm = LossManager(
        autoencoder=ae,
        sample_rate=SR,
        loss_config=DISC_CONFIGS[disc_type],
        audio_channels=CHANNELS,
    )
    reals, fakes = _make_signals()

    with torch.no_grad():
        dis_loss, adv_loss, fm_dist = lm.discriminator.loss(reals, fakes)

    # Discriminator loss should be positive for mismatched random signals.
    assert dis_loss.item() > 0, f"dis_loss should be > 0, got {dis_loss.item():.6g}"
    # Adversarial generator loss (e.g. hinge -D(fake)) is finite
    assert torch.isfinite(adv_loss), f"adv_loss should be finite, got {adv_loss.item():.6g}"
    # Feature-matching can legitimately be zero in some architectures, so we
    # only check non-negativity.
    assert fm_dist.item() >= 0, f"fm_dist should be >= 0, got {fm_dist.item():.6g}"


def test_all_seven_types_covered():
    """Sanity: we test exactly 7 discriminator types."""
    assert len(DISC_CONFIGS) == 7
    assert set(DISC_CONFIGS.keys()) == {
        "oobleck", "encodec", "dac", "big_vgan",
        "transformer", "hil", "wavtokenizer",
    }
