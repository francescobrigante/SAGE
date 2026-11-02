# ===============
# Speed-up keys of the WavTokenizer critic, checked against the default critic on every stereo arm
# layout (C0, A1, A2, A3, A4): freeze_disc_in_gen_step and inplace_act give the same losses and the
# same generator / critic gradients bit for bit; channels_last gives them to float rounding, and its
# weights stay channels-last through .to(), load_state_dict and the optimizer state. The discriminator
# step keys: dis_loss and fused_feature_matching are bit-exact.
# ===============
import contextlib
import types

import pytest
import torch
import torch.nn as nn

from sage.nn.discriminators.wavtokenizer import WavTokenizerGANLoss
from sage.training.engine import AutoencoderEngine, frozen_parameters

BASE = dict(channels=2, sample_rate=44100, periods=[2, 3], resolutions=[(512, 128, 512)], fft_sizes=[512],
            preprocess=True, fold_lrms=True)
ARMS = {
    "c0": {},
    "a1": dict(preprocess_gain="shared"),
    "a2": dict(preprocess_gain="shared", mpd_stereo="lr", mrd_stereo="lrms", dac_stereo="lr"),
    "a3": dict(preprocess_gain="shared", mpd_stereo="lrms", mrd_stereo="lrms", dac_stereo="lrms"),
    "a4": dict(preprocess_gain="shared", resolutions=[(1024, 256, 1024)], fft_sizes=[1024],
               dac_bands=[0.0, 0.011337868, 0.045351474, 0.181405896, 0.453514739, 1.0]),
}
T = 8192
CL_RTOL = 1e-5            # channels_last vs default, float32 on CPU
CL_GRAD_ATOL = 1e-2       # critic weight gradients of either float32 layout vs float64, as one vector
CL_SEEDS = 4              # batches over which the float32 layouts are compared


def _gan(arm, seed=0, **kw):
    torch.manual_seed(seed)
    return WavTokenizerGANLoss(loss_type="rpgan", **{**BASE, **ARMS[arm], **kw})


def _pair(seed=1, b=2):
    g = torch.Generator().manual_seed(seed)
    m, s = torch.randn(b, T, generator=g), torch.randn(b, T, generator=g)
    real = 0.3 * torch.stack([m + 0.4 * s, m - 0.4 * s], dim=1)                 # (B, 2, T)
    fake = real + 0.05 * torch.randn(real.shape, generator=g)
    return real, fake


def _gen_step(gan, real, fake, freeze=False):
    """Generator side: L_adv + L_fm backward into the fake. Returns losses, fake grad, critic grads."""
    fake = fake.clone().requires_grad_(True)
    with frozen_parameters(gan, enabled=freeze):
        _, adv, fm = gan.loss(reals=real, fakes=fake)
    (0.1 * adv + 0.2 * fm).backward()
    grads = [None if p.grad is None else p.grad.clone() for p in gan.parameters()]
    gan.zero_grad(set_to_none=True)
    return (adv.detach(), fm.detach()), fake.grad, grads


def _disc_step(gan, real, fake):
    """Critic side: the disc-step loss (fake detached) backward into the critic."""
    dis, _, _ = gan.loss(reals=real, fakes=fake.detach())
    dis.backward()
    grads = [p.grad.clone() for p in gan.parameters()]
    gan.zero_grad(set_to_none=True)
    return dis.detach(), grads


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ── frozen_parameters ────────────────────────────────────────────────────────

def test_frozen_parameters_restores_only_what_it_froze():
    m = nn.Linear(3, 3)
    m.bias.requires_grad_(False)
    with frozen_parameters(m):
        assert not any(p.requires_grad for p in m.parameters())
    assert m.weight.requires_grad and not m.bias.requires_grad


def test_frozen_parameters_restores_on_error_and_noop_when_disabled():
    m = nn.Linear(3, 3)
    with pytest.raises(RuntimeError):
        with frozen_parameters(m):
            raise RuntimeError("boom")
    assert all(p.requires_grad for p in m.parameters())
    with frozen_parameters(m, enabled=False):
        assert all(p.requires_grad for p in m.parameters())
    with frozen_parameters(None):
        pass


# ── freeze_disc_in_gen_step ──────────────────────────────────────────────────

@pytest.mark.parametrize("arm", list(ARMS))
def test_frozen_critic_gives_the_generator_gradient_bit_for_bit(arm):
    gan = _gan(arm)
    real, fake = _pair()
    losses, g_ref, d_grads = _gen_step(gan, real, fake, freeze=False)
    losses_f, g_frozen, d_grads_f = _gen_step(gan, real, fake, freeze=True)
    assert all(torch.equal(a, b) for a, b in zip(losses, losses_f))
    assert torch.equal(g_ref, g_frozen)
    assert all(g is not None for g in d_grads)            # today: critic weight gradients, discarded later
    assert all(g is None for g in d_grads_f)              # frozen: none computed
    assert all(p.requires_grad for p in gan.parameters())


def test_frozen_critic_builds_no_graph_on_the_real_path():
    gan = _gan("c0")
    real, fake = _pair()
    with frozen_parameters(gan):
        fmaps_real = gan.discriminator(real)
        fmaps_fake = gan.discriminator(fake.clone().requires_grad_(True))
    assert all(t.grad_fn is None for fm in fmaps_real for t in fm)
    assert all(t.grad_fn is not None for fm in fmaps_fake for t in fm)


class _StubAE(nn.Module):
    """One-parameter autoencoder whose waveform output is a fixed stereo clip scaled by its spectrogram."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.encoder = types.SimpleNamespace(is_complex=False)
        self.has_pre_transform = self.pre_transform_applies_to_target = self.pre_transform_applies_inverse = False
        self.register_buffer("clip", _pair(seed=7)[0])                         # (2, 2, T)

    def encode(self, x, return_info=False, **kw):
        latents = x.reshape(x.shape[0], 4, -1)[..., :4] * self.w
        return (latents, {}) if return_info else latents

    def decode(self, z, encoder_info=None, apply_inverse=False):
        return z.mean(dim=2)[:, :, None, None].expand(z.shape[0], 4, 16, 8) * self.w

    def istft(self, sp, target_length=None):
        return self.clip[: sp.shape[0]] * (1 + 0.1 * sp.mean(dim=(1, 2, 3)))[:, None, None] * self.w


def _stub_engine(freeze, loss_only=False, **critic):
    """The engine of test_o1_phase_compute, with a real critic behind a stub loss manager."""
    eng = AutoencoderEngine.__new__(AutoencoderEngine)
    nn.Module.__init__(eng)
    eng.autoencoder = _StubAE()
    gan = _gan("c0", periods=[2], fft_sizes=[512], **critic)
    lm = nn.Module()
    lm.discriminator = gan
    lm.use_disc = True
    lm.losses_gen = lambda info: (info["decoded"].mean() + 0.1 * info["loss_adv"]
                                  + 0.2 * info["feature_matching_distance"], {})
    lm.losses_disc = lambda info: (info["loss_dis"], {})
    eng.loss_manager = lm
    for k, v in dict(teacher_model=None, use_disc=True, warmup_steps=0, warmup_mode="adv",
                     encoder_freeze_on_warmup=False, freeze_encoder=False, force_input_mono=False,
                     latent_mask_ratio=0.0, audio_channels=2, sample_rate=44100,
                     stft_params={"n_fft": 2048, "hop_length": 512}, adv_grad_every=0,
                     freeze_disc_in_gen_step=freeze, disc_step_loss_only=loss_only).items():
        setattr(eng, k, v)
    return eng


def test_engine_key_same_generator_update_and_disc_step():
    torch.manual_seed(3)
    batch = (torch.randn(2, 4, 16, 8), 0.3 * torch.randn(2, 2, T))
    outs = {}
    for freeze in (False, True):
        eng = _stub_engine(freeze, loss_only=freeze)
        gen = eng.compute(batch, gen_step=1, disc_phase=False)
        gen["gen_total"].backward()
        d_grads = [p.grad for p in eng.loss_manager.discriminator.parameters()]
        g_ae = eng.autoencoder.w.grad.clone()
        eng.loss_manager.discriminator.zero_grad(set_to_none=True)
        dis = eng.compute(batch, gen_step=1, disc_phase=True)
        dis["disc_total"].backward()
        outs[freeze] = (gen["gen_total"].detach(), g_ae, d_grads,
                        [p.grad.clone() for p in eng.loss_manager.discriminator.parameters()])
        assert all(p.requires_grad for p in eng.loss_manager.discriminator.parameters())
    assert torch.equal(outs[False][0], outs[True][0])
    assert torch.equal(outs[False][1], outs[True][1])
    assert all(g is not None for g in outs[False][2]) and all(g is None for g in outs[True][2])
    assert all(torch.equal(a, b) for a, b in zip(outs[False][3], outs[True][3]))


def test_trainer_default_is_off():
    from omegaconf import OmegaConf
    from pathlib import Path
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "trainer.yaml")
    assert cfg.trainer.freeze_disc_in_gen_step is False
    assert cfg.trainer.disc_step_loss_only is False
    critic = WavTokenizerGANLoss(loss_type="rpgan", **BASE)
    assert not critic.fused_feature_matching


# ── inplace_act ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("arm", list(ARMS))
def test_inplace_act_is_bit_identical(arm):
    ref, ipl = _gan(arm), _gan(arm, inplace_act=True)
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), ipl.state_dict().values()))
    real, fake = _pair()
    with torch.no_grad():
        for fa, fb in zip(ref.discriminator(real), ipl.discriminator(real)):
            assert all(torch.equal(ta, tb) for ta, tb in zip(fa, fb))
    for freeze in (False, True):
        (la, ga, da), (lb, gb, db) = _gen_step(ref, real, fake, freeze), _gen_step(ipl, real, fake, freeze)
        assert all(torch.equal(a, b) for a, b in zip(la, lb))
        assert torch.equal(ga, gb)
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(da, db))
    (dis_a, gr_a), (dis_b, gr_b) = _disc_step(ref, real, fake), _disc_step(ipl, real, fake)
    assert torch.equal(dis_a, dis_b)
    assert all(torch.equal(a, b) for a, b in zip(gr_a, gr_b))


def test_inplace_act_reaches_every_branch():
    gan = _gan("c0", inplace_act=True).discriminator
    assert all(d.inplace_act for d in gan.vocos_mpd.discriminators)
    assert all(d.inplace_act for d in gan.vocos_mrd.discriminators)
    acts = [m for m in gan.dac_disc.modules() if isinstance(m, nn.LeakyReLU)]
    assert acts and all(m.inplace for m in acts)


def test_inplace_act_keeps_fewer_activations_for_the_backward():
    def saved_bytes(gan, real, fake):
        storages = {}                                    # distinct buffers kept alive for the backward
        def pack(t):
            storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = gan.loss(reals=real, fakes=fake.detach())
        return sum(storages.values())
    real, fake = _pair()
    assert saved_bytes(_gan("c0", inplace_act=True), real, fake) < 0.75 * saved_bytes(_gan("c0"), real, fake)


# ── channels_last ────────────────────────────────────────────────────────────

def _norm_rel(a, b):
    """‖a - b‖ / ‖b‖ over a list of tensors taken as one vector (None entries skipped)."""
    pairs = [(x.double(), y.double()) for x, y in zip(a, b) if y is not None]
    num = sum(float((x - y).pow(2).sum()) for x, y in pairs)
    den = sum(float(y.pow(2).sum()) for _, y in pairs)
    return (num / max(den, 1e-300)) ** 0.5


@contextlib.contextmanager
def _float64_windows():
    """audiotools casts the STFT window to float32 (DAC branch): float64 windows inside this block."""
    import audiotools
    orig = audiotools.AudioSignal.__dict__["get_window"]
    audiotools.AudioSignal.get_window = staticmethod(lambda wt, wl, dev: orig.__func__(wt, wl, dev).double())
    try:
        yield
    finally:
        audiotools.AudioSignal.get_window = orig


@pytest.fixture
def float64_stft_window():
    with _float64_windows():
        yield


@pytest.mark.parametrize("arm", list(ARMS))
def test_channels_last_is_the_same_critic_in_float64(arm, float64_stft_window):
    """In float64 rounding is out of the picture: the NHWC critic is the default critic to 1e-10."""
    ref, ncl = _gan(arm).double(), _gan(arm, channels_last=True).double()
    assert all(_cl(p) for p in ncl.parameters())                          # .double() keeps NHWC
    real, fake = (t.double() for t in _pair())
    with torch.no_grad():
        for fa, fb in zip(ref.discriminator(real), ncl.discriminator(real)):
            assert all(_rel(tb, ta) < 1e-10 for ta, tb in zip(fa, fb))
    (la, ga, _), (lb, gb, _) = _gen_step(ref, real, fake), _gen_step(ncl, real, fake)
    assert all(_rel(b, a) < 1e-10 for a, b in zip(la, lb)) and _rel(gb, ga) < 1e-10
    (dis_a, gr_a), (dis_b, gr_b) = _disc_step(ref, real, fake), _disc_step(ncl, real, fake)
    assert _rel(dis_b, dis_a) < 1e-10
    assert all(_rel(b, a) < 1e-10 for a, b in zip(gr_a, gr_b))


def _cl_errors(arm, seed):
    """Generator- and critic-gradient errors of both float32 layouts against the float64 critic."""
    real, fake = _pair(seed=seed)
    ref64 = _gan(arm).double()
    with _float64_windows():
        (l64, g64, _), (d64, gr64) = (_gen_step(ref64, real.double(), fake.double()),
                                      _disc_step(ref64, real.double(), fake.double()))
    err = {}
    for name, gan in (("default", _gan(arm)), ("nhwc", _gan(arm, channels_last=True))):
        (l, g, _), (d, gr) = _gen_step(gan, real, fake), _disc_step(gan, real, fake)
        assert all(_rel(a, b) < CL_RTOL for a, b in zip(l, l64)) and _rel(d, d64) < CL_RTOL
        err[name] = (_norm_rel([g], [g64]), _norm_rel(gr, gr64))
    return err


@pytest.mark.parametrize("arm", list(ARMS))
def test_channels_last_matches_to_float_rounding(arm):
    """float32, over CL_SEEDS batches, both layouts against the float64 critic. Losses: per value.
    Generator gradient (what trains SAGE): NHWC no farther than the default layout (median over the
    batches; on one batch either layout can be the unlucky one, a pre-activation rounding across a
    LeakyReLU kink). Critic weight gradients: on CPU the NHWC weight-gradient kernels are measurably
    less accurate (median ~1e-3 vs ~1e-4 of the default, local/disc_speed/cl_seeds.py), so both
    layouts only get the absolute bound CL_GRAD_ATOL; the GPU (cuDNN) check is in the same script."""
    errs = [_cl_errors(arm, seed) for seed in range(1, CL_SEEDS + 1)]
    med = lambda name, i: sorted(e[name][i] for e in errs)[len(errs) // 2]
    assert med("nhwc", 0) < 2 * med("default", 0) + 1e-6
    assert all(e[name][1] < CL_GRAD_ATOL for e in errs for name in e)


def _cl(t):
    return t.dim() != 4 or t.is_contiguous(memory_format=torch.channels_last)


def test_channels_last_convs_get_nhwc_weight_and_input():
    gan = _gan("c0", channels_last=True)
    seen = []
    def check(conv, args):
        seen.append(_cl(conv.weight) and _cl(args[0]))
    convs = [m for m in gan.modules() if isinstance(m, nn.Conv2d)]
    for m in convs:
        m.register_forward_pre_hook(check)                  # runs after weight_norm's and channels_last's
    real, fake = _pair()
    gan.loss(reals=real, fakes=fake)
    assert len(seen) >= 2 * len(convs) and all(seen)


def test_channels_last_survives_load_and_optimizer():
    ncl = _gan("c0", channels_last=True)
    params4d = [p for p in ncl.parameters() if p.dim() == 4]
    assert params4d and all(_cl(p) for p in params4d)
    ncl.load_state_dict(_gan("c0", seed=5).state_dict())                # checkpoint of a default critic
    assert all(_cl(p) for p in ncl.parameters())
    ncl.to(torch.float32)
    assert all(_cl(p) for p in ncl.parameters())
    opt = torch.optim.AdamW(ncl.parameters(), lr=1e-4)
    real, fake = _pair()
    ncl.loss(reals=real, fakes=fake)[0].backward()
    opt.step()
    assert all(_cl(p) for p in ncl.parameters())
    assert all(_cl(s["exp_avg"]) for s in opt.state.values())
    ref = _gan("c0")
    ref.load_state_dict(ncl.state_dict())                                 # and back into a default critic
    assert all(p.is_contiguous() for p in ref.parameters())



# ── discriminator step: dis_loss, fused feature matching ─────────────────────

def _dis_grads(gan, fn, real, fake):
    dis = fn(gan, real, fake)
    dis.backward()
    grads = [p.grad.clone() for p in gan.parameters()]
    gan.zero_grad(set_to_none=True)
    return dis.detach(), grads


@pytest.mark.parametrize("arm", list(ARMS))
def test_dis_loss_is_the_first_output_of_loss_bit_for_bit(arm):
    gan = _gan(arm)
    real, fake = _pair()
    a = _dis_grads(gan, lambda g, r, f: g.loss(reals=r, fakes=f)[0], real, fake)
    b = _dis_grads(gan, lambda g, r, f: g.dis_loss(r, f), real, fake)
    assert torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("freeze", [False, True])
def test_fused_feature_matching_is_bit_identical(arm, freeze):
    ref, fused = _gan(arm), _gan(arm, fused_feature_matching=True)
    real, fake = _pair()
    (la, ga, ca), (lb, gb, cb) = _gen_step(ref, real, fake, freeze), _gen_step(fused, real, fake, freeze)
    assert all(torch.equal(a, b) for a, b in zip(la, lb)) and torch.equal(ga, gb)
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(ca, cb))


def test_fused_feature_matching_keeps_int8_signs_for_the_backward():
    real, fake = _pair()
    sizes = {}
    for fused in (False, True):
        gan = _gan("c0", fused_feature_matching=True) if fused else _gan("c0")
        saved = {}
        def pack(t):
            saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t
        f = fake.clone().requires_grad_(True)
        with frozen_parameters(gan), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _, adv, fm = gan.loss(reals=real, fakes=f)
        sizes[fused] = sum(saved.values())
    assert sizes[True] < sizes[False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
