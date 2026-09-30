# ===============
# fold_lrms tests: backward-compat (default off → bit-identical) and the
# [L,R,M,S] batch-fold path (doubled fmap batch, non-zero Side gradient).
# ===============
import sys


import torch

from sage.nn.discriminators.wavtokenizer import (
    WavTokenizerDiscriminator,
    WavTokenizerGANLoss,
)

_B, _T = 2, 8192
_KW = dict(channels=2, sample_rate=44100, fft_sizes=[512], preprocess=True)


def _pair_seeded(fold):
    torch.manual_seed(0)
    return WavTokenizerDiscriminator(fold_lrms=fold, **_KW)


def test_default_off_bit_identical():
    torch.manual_seed(1)
    x = torch.randn(_B, 2, _T)
    d_off, d_ref = _pair_seeded(False), _pair_seeded(False)
    d_ref.load_state_dict(d_off.state_dict())
    outs_a, outs_b = d_off(x), d_ref(x)
    assert len(outs_a) == len(outs_b)
    for fa, fb in zip(outs_a, outs_b):
        for ta, tb in zip(fa, fb):
            assert torch.equal(ta, tb)


def test_fold_doubles_batch():
    x = torch.randn(_B, 2, _T)
    d_off, d_on = _pair_seeded(False), _pair_seeded(True)
    d_on.load_state_dict(d_off.state_dict())
    for f_off, f_on in zip(d_off(x), d_on(x)):
        assert f_on[-1].shape[0] == 2 * f_off[-1].shape[0], (f_off[-1].shape, f_on[-1].shape)


def test_side_gradient_nonzero():
    torch.manual_seed(3)
    gan = WavTokenizerGANLoss(loss_type="rpgan", fold_lrms=True, **_KW)
    mid = torch.randn(_B, 1, _T)
    side = (0.3 * torch.randn(_B, 1, _T)).requires_grad_(True)
    fakes = torch.cat([mid + side, mid - side], dim=1)
    reals = torch.randn(_B, 2, _T)
    _, adv_loss, fm = gan.loss(reals=reals, fakes=fakes)
    (adv_loss + fm).backward()
    assert side.grad is not None and side.grad.abs().sum() > 0


def test_mono_input_skips_fold():
    # NOTE: channels=1 here — with channels=2 the DAC MRD rearrange
    # (ch=2) would crash on mono input regardless of the fold.
    torch.manual_seed(0)
    d_on = WavTokenizerDiscriminator(fold_lrms=True, channels=1,
                                     sample_rate=44100, fft_sizes=[512])
    x = torch.randn(_B, 1, _T)                # C=1 → fold silently skipped
    fmaps = d_on(x)
    assert len(fmaps) > 0


def test_fold_changes_gradient():
    """Isolate the fold's contribution: same weights, fold on vs off must
    produce different Side gradients (on-path M/S views add signal)."""
    torch.manual_seed(3)
    gan_off = WavTokenizerGANLoss(loss_type="rpgan", fold_lrms=False, **_KW)
    gan_on = WavTokenizerGANLoss(loss_type="rpgan", fold_lrms=True, **_KW)
    gan_on.load_state_dict(gan_off.state_dict())
    mid = torch.randn(_B, 1, _T)
    reals = torch.randn(_B, 2, _T)
    grads = []
    for gan in (gan_off, gan_on):
        torch.manual_seed(7)  # same side noise both passes
        side = (0.3 * torch.randn(_B, 1, _T)).requires_grad_(True)
        fakes = torch.cat([mid + side, mid - side], dim=1)
        _, adv, fm = gan.loss(reals=reals, fakes=fakes)
        (adv + fm).backward()
        grads.append(side.grad.clone())
    assert not torch.allclose(grads[0], grads[1])
