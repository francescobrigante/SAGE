# ===============
# Stereo keys of the WavTokenizer discriminator (PLAN_WAVTOKENIZER_STEREO M2): defaults equal to
# the paper's critic (signature of main's code), the shared DAC gain (ILD and Side level survive,
# no 1/eps gradient on mono tracks), the joint critics (both channels used, M/S magnitudes see the
# inter-channel phase), and the guard rails on invalid combinations.
# ===============
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sage.nn.discriminators.dac import peak_normalize
from sage.nn.discriminators.wavtokenizer import WavTokenizerDiscriminator, WavTokenizerGANLoss

SIGNATURE = Path(__file__).parent / "data" / "wavtok_disc_signature_main.json"
PAPER = dict(channels=2, sample_rate=44100, periods=[2, 3, 5, 7, 11], fft_sizes=[2048, 1024, 512],
             preprocess=True, fold_lrms=True)
SMALL = dict(channels=2, sample_rate=44100, periods=[2, 3], resolutions=[(512, 128, 512)], fft_sizes=[512],
             preprocess=True, fold_lrms=True)
A1 = dict(preprocess_gain="shared")
A3 = dict(preprocess_gain="shared", mpd_stereo="lrms", mrd_stereo="lrms", dac_stereo="lrms")
T = 8192


def _disc(seed=0, **kw):
    torch.manual_seed(seed)
    return WavTokenizerDiscriminator(**{**SMALL, **kw})


def _stereo(seed=1, b=2, t=T, side=0.4):
    g = torch.Generator().manual_seed(seed)
    m, s = torch.randn(b, t, generator=g), torch.randn(b, t, generator=g)
    return 0.3 * torch.stack([m + side * s, m - side * s], dim=1)              # (B, 2, T)


def _with_side(x, gain):
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return torch.stack([m + gain * s, m - gain * s], dim=1)


# ── 1. defaults = the paper's critic ─────────────────────────────────────────

def test_defaults_reproduce_the_signature_of_main():
    ref = json.loads(SIGNATURE.read_text())
    torch.manual_seed(ref["seed_init"])
    disc = WavTokenizerDiscriminator(**ref["config"])
    g = torch.Generator().manual_seed(1)
    x = 0.3 * torch.randn(2, 2, 16384, generator=g)
    x[1, 1] = x[1, 0]
    with torch.no_grad():
        fmaps = disc(x)
    assert sum(p.numel() for p in disc.parameters()) == ref["n_params"]
    assert len(fmaps) == len(ref["fmaps"])
    for fm, fm_ref in zip(fmaps, ref["fmaps"]):
        assert len(fm) == len(fm_ref)
        for t, (shape, total, abs_total) in zip(fm, fm_ref):
            assert list(t.shape) == shape
            assert float(t.double().sum()) == pytest.approx(total, rel=1e-5, abs=1e-6)
            assert float(t.double().abs().sum()) == pytest.approx(abs_total, rel=1e-5)
    gan = WavTokenizerGANLoss(loss_type="rpgan", **ref["config"])
    gan.discriminator.load_state_dict(disc.state_dict())
    with torch.no_grad():
        losses = gan.loss(reals=x, fakes=x + 0.05 * torch.randn(x.shape, generator=g))
    assert [float(v) for v in losses] == pytest.approx(ref["loss"], rel=1e-5)


def _main_order_forward(disc, x):
    """main's WavTokenizerDiscriminator.forward (cdfe553), on the sub-discriminators of `disc`."""
    if disc.fold_lrms and x.ndim == 3 and x.shape[1] == 2:
        x = torch.cat([x, torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)], dim=0)
    if x.ndim == 2:
        x_mono, x_dac = x, x.unsqueeze(1)
    else:
        x_mono, x_dac = (x.squeeze(1) if x.shape[1] == 1 else x.flatten(0, 1)), x
    fmaps = [d(x_mono)[1] for d in disc.vocos_mpd.discriminators]
    fmaps += [d(x_mono)[1] for d in disc.vocos_mrd.discriminators]
    if disc.use_dac:
        fmaps += disc.dac_disc(peak_normalize(x_dac) if disc.preprocess else x_dac)   # main: inside dac_disc
    return fmaps


def _grad_case(case, forward=None):
    """Seeded GAN loss on a default critic; the gradient reaching the fake input, and the losses."""
    torch.manual_seed(0)
    gan = WavTokenizerGANLoss(loss_type="rpgan", **case["config"])
    if forward is not None:
        gan.discriminator.forward = lambda x: forward(gan.discriminator, x)
    g = torch.Generator().manual_seed(1)
    real = 0.3 * torch.randn(*case["shape"], generator=g)
    fake = (real + 0.05 * torch.randn(*case["shape"], generator=g)).requires_grad_(True)
    losses = gan.loss(reals=real, fakes=fake)
    (losses[1] + losses[2]).backward()
    return fake.grad, [float(v) for v in losses]


GRAD_CASES = json.loads(SIGNATURE.read_text())["grad_cases"]


@pytest.mark.parametrize("case", GRAD_CASES, ids=[c["name"] for c in GRAD_CASES])
def test_defaults_give_main_generator_gradient_bit_for_bit(case):
    """Same graph order as main: the gradient reaching the generator is equal bit for bit, and its
    float64 sums match those computed with main's code (tests/data, make_disc_signature.py) up to the
    float32 rounding of the CPU kernels: the signed sum cancels, so its tolerance is relative to
    sum(|g|) (AVX2 vs AVX512 kernels: 4e-7 of it)."""
    grad, losses = _grad_case(case)
    grad_main_order, _ = _grad_case(case, forward=_main_order_forward)
    assert torch.equal(grad, grad_main_order)
    assert float(grad.double().sum()) == pytest.approx(case["grad_sum"], abs=1e-5 * case["grad_abs_sum"])
    assert float(grad.double().abs().sum()) == pytest.approx(case["grad_abs_sum"], rel=1e-5)
    assert losses == pytest.approx(case["loss"], rel=1e-5)


def test_explicit_default_keys_are_bit_identical():
    x = _stereo()
    a = _disc()
    b = _disc(preprocess_gain="per_channel", mpd_stereo="fold", mrd_stereo="fold", dac_stereo="fold")
    b.load_state_dict(a.state_dict())
    for fa, fb in zip(a(x), b(x)):
        assert all(torch.equal(ta, tb) for ta, tb in zip(fa, fb))


# ── 2-3. the DAC preprocess: per channel vs shared gain ──────────────────────

def _rows(disc, x):
    """dac_input split into the folded [L, R] rows and [M, S] rows: (B, 2, T) each."""
    d = disc.dac_input(x)
    return d[: x.shape[0]], d[x.shape[0]:]


def test_per_channel_gain_is_blind_to_side_level_and_ild():
    disc, x = _disc(), _stereo()
    _, ms = _rows(disc, x)
    _, ms_quiet = _rows(disc, _with_side(x, 0.25))                          # Side -12 dB
    assert torch.allclose(ms[:, 1], ms_quiet[:, 1], atol=1e-6)              # S row unchanged
    lr, _ = _rows(disc, x)
    lr_ild, _ = _rows(disc, torch.stack([x[:, 0], 0.5 * x[:, 1]], dim=1))  # R -6 dB
    assert torch.allclose(lr[:, 1], lr_ild[:, 1], atol=1e-6)                # R row unchanged


def test_shared_gain_keeps_side_level_and_ild():
    disc, x = _disc(**A1), _stereo()
    _, ms = _rows(disc, x)
    _, ms_quiet = _rows(disc, _with_side(x, 0.25))
    assert not torch.allclose(ms[:, 1], ms_quiet[:, 1], atol=1e-3)
    lr, _ = _rows(disc, torch.stack([x[:, 0], 0.5 * x[:, 1]], dim=1))
    rms = lr.square().mean(-1).sqrt()                                       # (B, 2)
    x_ild = torch.stack([x[:, 0], 0.5 * x[:, 1]], dim=1)
    rms_in = x_ild.sub(x_ild.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    assert torch.allclose(rms[:, 1] / rms[:, 0], rms_in[:, 1] / rms_in[:, 0], rtol=1e-3)   # ILD survives


def test_shared_gain_is_one_gain_per_example_with_ms_derived_after():
    disc, x = _disc(**A1), _stereo()
    lr, ms = _rows(disc, x)
    assert torch.allclose(lr.abs().flatten(1).max(-1)[0], torch.full((x.shape[0],), 0.8), atol=1e-6)
    lr_ild, _ = _rows(disc, torch.stack([x[:, 0], 0.5 * x[:, 1]], dim=1))
    assert (lr_ild[:, 1].abs().max(-1)[0] < 0.6).all()                       # R -6 dB stays below L
    assert torch.allclose(ms[:, 0], lr[:, 0] + lr[:, 1], atol=1e-6)
    assert torch.allclose(ms[:, 1], lr[:, 0] - lr[:, 1], atol=1e-6)
    assert torch.allclose(disc.dac_input(0.3 * x), disc.dac_input(x), atol=1e-6)   # invariant to volume


def test_shared_gain_is_per_example_not_per_batch():
    """Real and fake go through separate forwards, and within one each example keeps its own gain."""
    disc, x = _disc(**A1), _stereo()
    quieter = torch.cat([x[:1], 0.1 * x[1:]])                               # example 1 at -20 dB
    assert torch.allclose(disc.dac_input(quieter), disc.dac_input(x), atol=1e-6)


def test_shared_gain_normalises_mono_2d_input():
    disc = _disc(channels=1, fold_lrms=False, **A1)
    y = disc.dac_input(0.1 * _stereo()[:, 0])                               # (B, T)
    assert y.shape == (2, 1, T)
    assert torch.allclose(y.abs().flatten(1).max(-1)[0], torch.full((2,), 0.8), atol=1e-6)


# ── 4-5. gradients on mono tracks and digital silence ────────────────────────

def _adv_grad_norm(kw, real, fake):
    torch.manual_seed(0)
    gan = WavTokenizerGANLoss(loss_type="rpgan", **{**SMALL, **kw})
    fake = fake.clone().requires_grad_(True)
    _, adv, fm = gan.loss(reals=real, fakes=fake)
    (adv + fm).backward()
    return fake.grad.norm().item()


def _mono_pair(eps):
    g = torch.Generator().manual_seed(2)
    m, n = 0.3 * torch.randn(2, T, generator=g), torch.randn(2, T, generator=g)
    real = torch.stack([m, m], dim=1)                                        # S == 0
    return real, torch.stack([m + eps * n, m - eps * n], dim=1)              # residual Side of size eps


def test_per_channel_gain_explodes_on_mono_tracks_shared_does_not():
    g_pc = [_adv_grad_norm({}, *_mono_pair(e)) for e in (1e-3, 1e-6)]
    g_sh = [_adv_grad_norm(A1, *_mono_pair(e)) for e in (1e-3, 1e-6)]
    assert g_pc[1] / g_pc[0] > 100                                           # grows like 1/eps
    assert g_sh[1] / g_sh[0] < 5                                             # bounded


def test_digital_silence_gives_finite_gradients():
    real, fake = _stereo(), _stereo(seed=3)
    real[..., T // 4: T // 2] = 0.0                                          # a silent stretch
    for kw in ({}, A1, A3):
        assert torch.isfinite(torch.tensor(_adv_grad_norm(kw, real, fake)))
    silent = torch.zeros(2, 2, T)                                            # a fully silent crop: both modes
    for kw in ({}, A1, A3):                                                  # amplify the fake to 0.8, but stay finite
        assert torch.isfinite(torch.tensor(_adv_grad_norm(kw, silent, 1e-7 * _stereo(seed=4))))


# ── 6-7. joint critics ───────────────────────────────────────────────────────

def test_joint_critics_use_both_channels():
    disc, x = _disc(**A3), _stereo().requires_grad_(True)
    fmaps = disc(x)
    assert len(fmaps) == len(SMALL["periods"]) + len(SMALL["resolutions"]) + len(SMALL["fft_sizes"])
    assert all(fm[-1].shape[0] == x.shape[0] for fm in fmaps)                # one row per stereo example
    sum(fm[-1].sum() for fm in fmaps).backward()
    g_l, g_r = x.grad[:, 0].norm(), x.grad[:, 1].norm()
    assert 0.1 < (g_l / g_r).item() < 10
    with torch.no_grad():
        base = torch.cat([fm[-1].flatten() for fm in disc(x)])
        no_r = torch.cat([fm[-1].flatten() for fm in disc(torch.stack([x[:, 0], 0 * x[:, 1]], 1))])
        swap = torch.cat([fm[-1].flatten() for fm in disc(x.flip(1))])
    assert (base - no_r).abs().max() > 1e-4
    assert (base - swap).abs().max() > 1e-4                                  # not L/R symmetric


def test_magnitude_mrd_on_lrms_sees_cos_ipd_but_not_its_sign():
    """|M|, |S| per bin depend on cos(IPD) only. Exact for a tone on a bin centre (12 periods per
    rectangular 512-sample frame: no leakage), away from the reflect-padded edge frames."""
    disc = _disc(**A3)
    mrd = disc.vocos_mrd.discriminators[0]                                   # resolution (512, 128, 512)
    f0 = 12 * 44100 / 512
    t = torch.arange(T, dtype=torch.float64) / 44100
    tone = lambda phi: torch.stack([torch.cos(2 * torch.pi * f0 * t),                         # noqa: E731
                                    torch.cos(2 * torch.pi * f0 * t + phi)]).unsqueeze(0).float()
    spec = lambda phi: mrd.spectrogram(disc._joint(tone(phi)))[..., 4:-4]                       # noqa: E731
    assert spec(0.0).shape[1] == 4                                           # |L|, |R|, |M|, |S|
    assert not torch.allclose(spec(0.0), spec(torch.pi / 2), rtol=1e-2, atol=1e-2)
    assert torch.allclose(spec(torch.pi / 2), spec(-torch.pi / 2), rtol=1e-4, atol=1e-2)   # documented limit


def test_joint_layouts_add_few_parameters():
    n = lambda d: sum(p.numel() for p in d.parameters())                    # noqa: E731
    c0, a3 = (n(WavTokenizerDiscriminator(**{**PAPER, **kw})) for kw in ({}, A3))
    assert 0 < a3 - c0 < 0.01 * c0                                          # only the first layers widen


def test_fold_lrms_only_touches_the_branches_that_fold():
    """In A3 every branch is joint: fold_lrms has no effect."""
    x = _stereo()
    on, off = _disc(**A3), _disc(**A3, fold_lrms=False)
    off.load_state_dict(on.state_dict())
    for fa, fb in zip(on(x), off(x)):
        assert torch.equal(fa[-1], fb[-1])


# ── 9. checkpoints ───────────────────────────────────────────────────────────

def test_init_from_disc_refuses_a_critic_of_another_layout(tmp_path):
    import train
    c0 = WavTokenizerGANLoss(loss_type="rpgan", **PAPER)
    ckpt = tmp_path / "c0.ckpt"
    torch.save({"state_dict": {f"engine.loss_manager.discriminator.{k}": v for k, v in c0.state_dict().items()}}, ckpt)
    a3 = WavTokenizerGANLoss(loss_type="rpgan", **{**PAPER, **A3})
    wrapper = SimpleNamespace(engine=SimpleNamespace(loss_manager=SimpleNamespace(discriminator=a3)))
    with pytest.raises(RuntimeError, match="size mismatch"):
        train._load_discriminator_weights(wrapper, str(ckpt))


# ── guard rails ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [dict(mpd_stereo="ms"), dict(preprocess_gain="global"),
                                 dict(mpd_stereo="lr"), dict(mrd_stereo="lr"),   # [L, R] layout: not a mode
                                 dict(dac_stereo="lr", preprocess_gain="shared"),
                                 dict(dac_stereo="lrms"),                     # joint DAC with per-channel gain
                                 dict(channels=1, mpd_stereo="lrms")])
def test_invalid_combinations_are_refused(bad):
    with pytest.raises(ValueError):
        WavTokenizerDiscriminator(**{**SMALL, **bad})


def test_joint_dac_without_preprocess_is_allowed():
    WavTokenizerDiscriminator(**{**SMALL, "preprocess": False, "dac_stereo": "lrms"})
