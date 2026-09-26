"""
Smoke tests for SAT-ported discriminators.

Tests verify:
  - Forward pass produces correct output shape
  - .loss() returns (dis_loss, adv_loss, fm_dist) tensors, all grad-carrying
  - Pretransforms produce correct encoded_channels and downsampling_ratio
  - No crash on 2-channel (stereo) audio

Run with:
    python -m pytest tests/test_discriminators_sat.py -v -x
"""

import sys, os

import pytest
import torch

DEVICE = 'cpu'
B = 2   # batch size
SR = 44100
T = SR  # 1 second of stereo audio at 44.1 kHz (large enough for all pretransforms)


@pytest.fixture(scope='module')
def stereo():
    return torch.randn(B, 2, T, device=DEVICE)


@pytest.fixture(scope='module')
def mono():
    return torch.randn(B, 1, T, device=DEVICE)


# ---------------------------------------------------------------------------
# Pretransforms
# ---------------------------------------------------------------------------

class TestPretransforms:

    def test_complex_stft_shape(self):
        from ar_spectra.models.pretransforms import ComplexSTFTPretransform
        pt = ComplexSTFTPretransform(channels=2, n_fft=1024)
        x = torch.randn(B, 2, T)
        z = pt.encode(x)
        assert z.shape[1] == pt.encoded_channels, f'Expected {pt.encoded_channels} ch, got {z.shape[1]}'
        assert z.ndim == 3

    def test_patched_shape(self):
        from ar_spectra.models.pretransforms import PatchedPretransform
        pt = PatchedPretransform(channels=2, patch_size=29)
        x = torch.randn(B, 2, T)
        z = pt.encode(x)
        assert z.shape[1] == pt.encoded_channels
        assert z.ndim == 3

    def test_patched_decode_roundtrip(self):
        from ar_spectra.models.pretransforms import PatchedPretransform
        pt = PatchedPretransform(channels=2, patch_size=16)
        x = torch.randn(B, 2, 1024)
        z = pt.encode(x)
        x_rec = pt.decode(z)
        assert x_rec.shape == x.shape

    def test_wavelet_shape(self):
        from ar_spectra.models.pretransforms import WaveletPretransform
        pt = WaveletPretransform(channels=2, levels=4)
        x = torch.randn(B, 2, 8192)  # must be multiple of 2^4
        z = pt.encode(x)
        assert z.shape[1] == pt.encoded_channels
        assert z.shape[-1] == 8192 // pt.downsampling_ratio

    def test_wavelet_decode_roundtrip(self):
        from ar_spectra.models.pretransforms import WaveletPretransform
        pt = WaveletPretransform(channels=2, levels=4)
        x = torch.randn(B, 2, 8192)
        z = pt.encode(x)
        x_rec = pt.decode(z)
        assert x_rec.shape == x.shape, f'{x_rec.shape} != {x.shape}'


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:

    def test_forward_shape(self):
        from ar_spectra.blocks.transformer_sat import TransformerBlock
        block = TransformerBlock(
            256, dim_heads=64, add_rope=True, norm_type='dyt',
            attn_kwargs={'qk_norm': 'dyt', 'differential': True},
            ff_kwargs={'mult': 2.0},
        )
        x = torch.randn(B, 32, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_sliding_window(self):
        from ar_spectra.blocks.transformer_sat import TransformerBlock
        block = TransformerBlock(128, dim_heads=64, add_rope=True, norm_type='rms_norm',
                                  attn_kwargs={'differential': False})
        x = torch.randn(B, 64, 128)
        out = block(x, self_attention_flash_sliding_window=[4, 4])
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# TransformerResamplingBlock
# ---------------------------------------------------------------------------

class TestTransformerResamplingBlock:

    def test_stride_reduction(self):
        from ar_spectra.blocks.transformer_sat import TransformerResamplingBlock
        trb = TransformerResamplingBlock(64, 128, stride=4, sliding_window=[2, 2],
                                         transformer_depth=2)
        x = torch.randn(B, 64, 64)
        out = trb(x)
        assert out.shape == (B, 128, 16), f'Expected (B, 128, 16), got {out.shape}'

    def test_return_features(self):
        from ar_spectra.blocks.transformer_sat import TransformerResamplingBlock
        trb = TransformerResamplingBlock(32, 64, stride=2, sliding_window=[3, 3],
                                         transformer_depth=3)
        x = torch.randn(B, 32, 32)
        out, feats = trb(x, return_features=True)
        assert len(feats) == 4  # initial + one per transformer block


# ---------------------------------------------------------------------------
# PQMF
# ---------------------------------------------------------------------------

class TestPQMF:

    def test_analysis_shape(self):
        from ar_spectra.models.discriminators.hil import PQMF
        pqmf = PQMF(subbands=4, taps=62, cutoff_freq=0.142, beta=9.0)
        x = torch.randn(B, 1, 4096)
        y = pqmf(x)
        assert y.shape == (B, 4, 4096 // 4), f'Expected (B, 4, 1024), got {y.shape}'

    def test_synthesis_shape(self):
        from ar_spectra.models.discriminators.hil import PQMF
        pqmf = PQMF(subbands=4, taps=62, cutoff_freq=0.142, beta=9.0)
        x = torch.randn(B, 1, 4096)
        y = pqmf.analysis(x)
        x_rec = pqmf.synthesis(y)
        # synthesis may differ by a few samples — just check shape is close
        assert x_rec.shape[0] == B and x_rec.shape[1] == 1


# ---------------------------------------------------------------------------
# MultiTransformerDiscriminator
# ---------------------------------------------------------------------------

class TestMultiTransformerDiscriminator:

    @pytest.fixture
    def disc(self):
        from ar_spectra.models.discriminators.transformer import MultiTransformerDiscriminator
        return MultiTransformerDiscriminator(
            in_channels=2,
            loss_type='rpgan',
            patched_kwargs={'enabled': True, 'depths': [2, 2, 2]},
        )

    def test_forward(self, disc, stereo):
        logits, fmaps = disc(stereo)
        assert len(logits) == 3, f'Expected 3 scales (patched), got {len(logits)}'
        for l in logits:
            assert l.ndim == 3  # (B, 1, T')

    def test_loss(self, disc, stereo):
        reals = stereo
        fakes = torch.randn_like(stereo)
        dis_loss, adv_loss, fm_dist = disc.loss(reals, fakes)
        assert dis_loss.requires_grad
        assert adv_loss.requires_grad
        for t in (dis_loss, adv_loss, fm_dist):
            assert t.ndim == 0  # scalar
            assert not torch.isnan(t), f'NaN in {t}'

    def test_hinge_loss(self, stereo):
        from ar_spectra.models.discriminators.transformer import MultiTransformerDiscriminator
        disc = MultiTransformerDiscriminator(
            in_channels=2, loss_type='hinge',
            patched_kwargs={'enabled': True, 'depths': [1, 1, 1]},
        )
        dis_loss, adv_loss, fm_dist = disc.loss(stereo, torch.randn_like(stereo))
        assert not torch.isnan(dis_loss)


# ---------------------------------------------------------------------------
# HILDiscriminator (only filter bank, skip chroma to speed up test)
# ---------------------------------------------------------------------------

class TestHILDiscriminator:

    @pytest.fixture
    def disc(self):
        from ar_spectra.models.discriminators.hil import HILDiscriminator
        return HILDiscriminator(
            filters=32,
            in_channels=2,
            loss_type='rpgan',
            normalize_losses=False,
        )

    def test_forward(self, disc, stereo):
        logits, fmaps = disc(stereo)
        assert len(logits) > 0

    def test_loss(self, disc, stereo):
        reals = stereo
        fakes = torch.randn_like(stereo)
        dis_loss, adv_loss, fm_dist = disc.loss(reals, fakes)
        assert not torch.isnan(dis_loss)
        assert not torch.isnan(adv_loss)


# ---------------------------------------------------------------------------
# loss_manager routing smoke test
# ---------------------------------------------------------------------------

class TestLossManagerRouting:

    def test_transformer_disc_type_accepted(self):
        """Verify loss_manager correctly routes disc_type='transformer'."""
        from ar_spectra.training.loss_manager import LossManager

        class FakeBottleneck:
            pass

        class FakeAE:
            bottleneck = FakeBottleneck()
            has_pre_transform = False

        loss_config = {
            'discriminator': {
                'type': 'transformer',
                'config': {
                    'patched_kwargs': {'enabled': True, 'depths': [1, 1, 1]},
                },
                'weights': {'adversarial': 1.0, 'feature_matching': 1.0},
            }
        }
        lm = LossManager(FakeAE(), sample_rate=44100, loss_config=loss_config, audio_channels=2)
        assert lm.discriminator is not None
        from ar_spectra.models.discriminators.transformer import MultiTransformerDiscriminator
        assert isinstance(lm.discriminator, MultiTransformerDiscriminator)
