# ===========================================================================
# F1b test — complex model configuration (is_complex=True).
#
# Verifies that:
#   CC1  The swin_cplx_baseline.yaml config composes via Hydra without errors
#   CC2  Encoder, decoder, and bottleneck instantiate from the composed config
#   CC3  is_complex=True is propagated to encoder and decoder
#   CC4  Encoder accepts a complex64 input and returns a complex output
#   CC5  Decoder accepts a complex latent and returns a complex reconstruction
#   CC6  Full round-trip (encoder → bottleneck → decoder) produces a finite
#        complex reconstruction matching the input's spatial shape
#   CC7  The SAGEAutoencoder wrapper wires everything correctly with is_complex
# ===========================================================================

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

import train  # noqa: F401  (registers ${mul:} / ${config:} resolvers)

from sage.model.encoder import SAGEEncoder
from sage.model.decoder import SAGEDecoder
from sage.nn.complex.bottleneck import ComplexVAEBottleneck
from sage.model.autoencoder import SAGEAutoencoder

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


# ---------------------------------------------------------------------------
# Helper to compose the complex config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cplx_cfg():
    """Compose swin_cplx_baseline config once for the module."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=["models=swin_cplx_baseline"])
    return cfg


# ---------------------------------------------------------------------------
# CC1: config composes without errors
# ---------------------------------------------------------------------------

def test_complex_config_composes(cplx_cfg):
    """swin_cplx_baseline.yaml composes cleanly via Hydra."""
    m = cplx_cfg.models.model
    assert m.encoder.is_complex is True
    assert m.decoder.is_complex is True
    assert m.encoder.in_channels == 2    # complex stereo
    assert m.decoder.in_channels == 2


# ---------------------------------------------------------------------------
# CC2: components instantiate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder(cplx_cfg):
    return instantiate(cplx_cfg.models.model.encoder)


@pytest.fixture(scope="module")
def decoder(cplx_cfg):
    return instantiate(cplx_cfg.models.model.decoder)


@pytest.fixture(scope="module")
def bottleneck(cplx_cfg):
    return instantiate(cplx_cfg.models.model.bottleneck)


def test_components_instantiate(encoder, decoder, bottleneck):
    """Encoder, decoder, and bottleneck instantiate from the composed config."""
    assert isinstance(encoder, SAGEEncoder)
    assert isinstance(decoder, SAGEDecoder)
    assert isinstance(bottleneck, ComplexVAEBottleneck)


# ---------------------------------------------------------------------------
# CC3: is_complex propagated
# ---------------------------------------------------------------------------

def test_is_complex_propagated(encoder, decoder):
    """is_complex=True is set on both encoder and decoder."""
    assert encoder.is_complex is True
    assert decoder.is_complex is True


# ---------------------------------------------------------------------------
# CC4 & CC5: forward passes with complex tensors
# ---------------------------------------------------------------------------

B = 1  # batch size for speed

def _cplx_input():
    """Random complex64 input matching the complex model's expected shape."""
    # Shape: (B, 2, 1024, 128) — stereo complex STFT
    return torch.randn(B, 2, 1024, 128, dtype=torch.complex64)


def test_encoder_forward_complex(encoder):
    """Encoder accepts complex64 input and returns complex output."""
    x = _cplx_input()
    encoder.eval()
    with torch.no_grad():
        z, info = encoder(x)
    assert torch.is_complex(z), f"Encoder output should be complex, got dtype={z.dtype}"
    assert z.ndim == 3, f"Expected 3D output (B, C, T), got {z.ndim}D"
    assert z.shape[0] == B
    assert torch.isfinite(z.real).all(), "Encoder output contains non-finite real parts"
    assert torch.isfinite(z.imag).all(), "Encoder output contains non-finite imag parts"


def test_decoder_forward_complex(decoder, encoder, bottleneck):
    """Decoder accepts complex latent and returns complex reconstruction."""
    x = _cplx_input()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_raw, enc_info = encoder(x)
        z, bn_info = bottleneck.encode(z_raw, return_info=True)
        recon = decoder(z)
    assert torch.is_complex(recon), f"Decoder output should be complex, got dtype={recon.dtype}"
    # Reconstruction shape must match input
    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} ≠ input shape {x.shape}"
    )
    assert torch.isfinite(recon.real).all(), "Reconstruction contains non-finite real parts"
    assert torch.isfinite(recon.imag).all(), "Reconstruction contains non-finite imag parts"


# ---------------------------------------------------------------------------
# CC6: full round-trip
# ---------------------------------------------------------------------------

def test_full_roundtrip_complex(encoder, decoder, bottleneck):
    """End-to-end encode → bottleneck → decode produces a finite complex tensor."""
    x = _cplx_input()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_raw, enc_info = encoder(x)
        z, bn_info = bottleneck.encode(z_raw, return_info=True)
        recon = decoder(z)

    assert recon.shape == x.shape
    assert torch.is_complex(recon)
    # Reconstruction of random input won't be close, but must be finite
    assert torch.isfinite(recon.real).all()
    assert torch.isfinite(recon.imag).all()


# ---------------------------------------------------------------------------
# CC7: SAGEAutoencoder wrapper with is_complex
# ---------------------------------------------------------------------------

def test_autoencoder_wrapper_complex(cplx_cfg):
    """SAGEAutoencoder.from_config builds a working complex model."""
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(cplx_cfg.models.model, resolve=True)
    ae = SAGEAutoencoder.from_config(model_cfg)
    ae.eval()

    x = _cplx_input()
    with torch.no_grad():
        latents, enc_info, bn_info = ae.encode(x, return_info=True)
        recon = ae.decode(latents, encoder_info=enc_info)

    assert recon is not None, "SAGEAutoencoder returned no reconstruction"
    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} ≠ input shape {x.shape}"
    )
    assert torch.isfinite(recon.real).all()
    assert torch.isfinite(recon.imag).all()


# ---------------------------------------------------------------------------
# CC8: data section overrides (cac=false, model_channels=2)
# ---------------------------------------------------------------------------

def test_complex_config_data_overrides(cplx_cfg):
    """Complex config disables CAC and sets model_channels=2."""
    data = cplx_cfg.data
    assert data.train_dataset.cac is False
    assert data.eval_dataset.cac is False
    assert data.demo.istft_params.cac is False
    assert data.model_channels == 2
