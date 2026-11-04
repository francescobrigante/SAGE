# ===============
# Speed-up keys of the WavTokenizer critic, checked against the default critic on every stereo arm
# layout (C0, A1, A3): freeze_disc_in_gen_step and inplace_act give the same losses and the
# same generator / critic gradients bit for bit; channels_last gives them to float rounding, and its
# weights stay channels-last through .to(), load_state_dict and the optimizer state. The discriminator
# step keys: dis_loss and fused_feature_matching are bit-exact.
# fused_bias_act: on CPU the plain path (bit-identical); its autograd Function (forced on CPU against
# the cuDNN order of operations, and with the Triton kernels on CUDA) gives the same fmaps, losses and
# gradients bit for bit; with a Triton other than the tested version it falls back to the plain path.
# ===============
import contextlib
import inspect
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sage.nn.discriminators.fused_act import BiasLeakyReLUConv2d
from sage.nn.discriminators.wavtokenizer import WavTokenizerDiscriminator, WavTokenizerGANLoss
from sage.training.engine import AutoencoderEngine, frozen_parameters

BASE = dict(channels=2, sample_rate=44100, periods=[2, 3], resolutions=[(512, 128, 512)], fft_sizes=[512],
            preprocess=True, fold_lrms=True)
ARMS = {
    "c0": {},
    "a1": dict(preprocess_gain="shared"),
    "a3": dict(preprocess_gain="shared", mpd_stereo="lrms", mrd_stereo="lrms", dac_stereo="lrms"),
}
T = 8192
CL_RTOL = 1e-5            # channels_last vs default, float32 on CPU
CL_GRAD_ATOL = 1e-2       # critic weight gradients of either float32 layout vs float64, as one vector
CL_GEN_ATOL = 1e-3        # generator gradient of either float32 layout vs float64 (median over CL_SEEDS)
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
    assert not (critic.fused_feature_matching or critic.discriminator.branch_streams)
    assert not (critic.discriminator.fused_bias_act or critic.discriminator.fused_bias_grad)
    assert not any(isinstance(m, BiasLeakyReLUConv2d) for m in critic.modules())


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
    Generator gradient (what trains SAGE): both layouts within float32 rounding of the float64 critic,
    CL_GEN_ATOL on the median over the batches (on one batch either layout can be the unlucky one, a
    pre-activation rounding across a LeakyReLU kink). Both are 1e-4 .. 4e-4 on AVX512 and AVX2 CPU
    kernels, but their ratio depends on the CPU (NHWC up to 4x the default on a GitHub runner), so the
    bound is absolute. Critic weight gradients: on CPU the NHWC weight-gradient kernels are measurably
    less accurate (median ~1e-3 vs ~1e-4 of the default), so both layouts only get the absolute bound
    CL_GRAD_ATOL."""
    errs = [_cl_errors(arm, seed) for seed in range(1, CL_SEEDS + 1)]
    med = lambda name, i: sorted(e[name][i] for e in errs)[len(errs) // 2]
    assert med("default", 0) < CL_GEN_ATOL and med("nhwc", 0) < CL_GEN_ATOL
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



# ── discriminator step: dis_loss, fused feature matching, branch streams ─────

# the arms, a stereo critic without fold_lrms and a mono critic
LAYOUTS = {**ARMS, "c0_nofold": dict(fold_lrms=False), "mono": dict(channels=1, fold_lrms=False)}


def _layout_gan(name, **kw):
    torch.manual_seed(0)
    return WavTokenizerGANLoss(loss_type="rpgan", **{**BASE, **LAYOUTS[name], **kw})


def _layout_pair(name):
    real, fake = _pair()
    return (real[:, :1], fake[:, :1]) if LAYOUTS[name].get("channels") == 1 else (real, fake)


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


def test_branch_streams_is_a_noop_on_cpu():
    ref, st = _gan("a3"), _gan("a3", branch_streams=True)
    real, fake = _pair()
    a = _dis_grads(ref, lambda g, r, f: g.dis_loss(r, f), real, fake)
    b = _dis_grads(st, lambda g, r, f: g.dis_loss(r, f), real, fake)
    assert torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams")
@pytest.mark.parametrize("arm", ["c0", "a3"])
def test_branch_streams_same_scores_and_gradients_on_cuda(arm):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ref, st = _gan(arm).cuda(), _gan(arm, branch_streams=True).cuda()
    real, fake = (t.cuda() for t in _pair())
    with torch.no_grad():
        for fa, fb in zip(ref.discriminator(real), st.discriminator(real)):
            assert all(torch.equal(x, y) for x, y in zip(fa, fb))
    (la, ga, _), (lb, gb, _) = _gen_step(ref, real, fake, True), _gen_step(st, real, fake, True)
    assert all(torch.equal(a, b) for a, b in zip(la, lb)) and _rel(gb, ga) < 1e-6
    a = _dis_grads(ref, lambda g, r, f: g.dis_loss(r, f), real, fake)
    b = _dis_grads(st, lambda g, r, f: g.dis_loss(r, f), real, fake)
    assert torch.equal(a[0], b[0]) and all(_rel(y, x) < 1e-6 for x, y in zip(a[1], b[1]))


# ── fused_bias_act ───────────────────────────────────────────────────────────

FUSED_KEYS = {"plain": {}, "inplace": dict(inplace_act=True), "nhwc": dict(channels_last=True),
              "inplace_nhwc": dict(inplace_act=True, channels_last=True)}


def _equal(a, b):
    return all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b))


def _conv_input_grads(gan, real, fake):
    """Generator step (critic frozen): the gradient reaching the input of every activated convolution,
    in call order. On CUDA these come from cuDNN's deterministic dgrad only; the gradient of the fake
    waveform then also crosses the reflect-pad backwards (STFT centring, period padding), which
    accumulate with atomics: it varies at ~1e-8 from run to run on the same critic."""
    grads, handles = {}, []
    def hook(m, args):
        key = (id(m), len([k for k in grads if k[0] == id(m)]))
        grads[key] = None
        if args[0].requires_grad:
            args[0].register_hook(lambda g, key=key: grads.__setitem__(key, g.clone()))
    convs = [m for m in gan.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1]
    handles = [m.register_forward_pre_hook(hook) for m in convs]
    try:
        _gen_step(gan, real, fake, freeze=True)
    finally:
        for h in handles:
            h.remove()
    order = {id(m): i for i, m in enumerate(convs)}
    return [grads[k] for k in sorted(grads, key=lambda k: (order[k[0]], k[1]))]


def _assert_same_critic(ref, fused, real, fake):
    """fmaps, losses, generator gradient (critic trainable and frozen) and critic gradients: bit for bit.
    On CUDA the gradient of the fake waveform is compared at the convolution inputs (bit for bit) and
    after the atomic reflect-pad backwards to their run-to-run noise (relative 1e-6)."""
    with torch.no_grad():
        for fa, fb in zip(ref.discriminator(real), fused.discriminator(real)):
            assert _equal(fa, fb)
    for freeze in (False, True):
        (la, ga, ca), (lb, gb, cb) = _gen_step(ref, real, fake, freeze), _gen_step(fused, real, fake, freeze)
        assert _equal(la, lb) and _equal(ca, cb)
        assert torch.equal(ga, gb) if ga.device.type == "cpu" else _rel(gb, ga) < 1e-6
    if real.is_cuda:
        ia, ib = _conv_input_grads(ref, real, fake), _conv_input_grads(fused, real, fake)
        assert len(ia) == len(ib) and any(g is not None for g in ia) and _equal(ia, ib)
    a = _dis_grads(ref, lambda g, r, f: g.dis_loss(r, f), real, fake)
    b = _dis_grads(fused, lambda g, r, f: g.dis_loss(r, f), real, fake)
    assert torch.equal(a[0], b[0]) and _equal(a[1], b[1])


@pytest.mark.parametrize("keys", list(FUSED_KEYS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_fused_bias_act_is_the_plain_path_on_cpu(layout, keys):
    ref = _layout_gan(layout, **FUSED_KEYS[keys])
    fused = _layout_gan(layout, **FUSED_KEYS[keys], fused_bias_act=True, fused_bias_grad=True)
    assert _equal(ref.state_dict().values(), fused.state_dict().values())
    assert list(ref.state_dict()) == list(fused.state_dict())
    _assert_same_critic(ref, fused, *_layout_pair(layout))


@pytest.mark.parametrize("keys", list(FUSED_KEYS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_fused_bias_act_function_is_bit_identical(layout, keys, monkeypatch):
    """The fused autograd Function forced on CPU (torch-op forward) against autograd through what cuDNN
    computes: the bias-free convolution, then + bias, then the LeakyReLU. (The CPU convolution adds the
    bias inside its accumulation, so the plain CPU critic differs from both by rounding.) Same values and
    the same gradients bit for bit: the Function's backward is autograd's."""
    def cudnn_like(self, x):
        z = self._conv_forward(x, self.weight, None)
        return z if self.bias is None else z + self.bias.view(1, -1, 1, 1)
    monkeypatch.setattr(nn.Conv2d, "forward", cudnn_like)                 # plain convs (+ every conv_post)
    monkeypatch.setattr(BiasLeakyReLUConv2d, "_fused", lambda self, x: True)
    ref = _layout_gan(layout, **FUSED_KEYS[keys])
    fused = _layout_gan(layout, **FUSED_KEYS[keys], fused_bias_act=True)
    _assert_same_critic(ref, fused, *_layout_pair(layout))


def test_fused_bias_act_reaches_every_conv_with_an_activation():
    gan = _gan("c0", fused_bias_act=True, inplace_act=True).discriminator
    fused = [m for m in gan.modules() if isinstance(m, BiasLeakyReLUConv2d)]
    subs = list(gan.vocos_mpd.discriminators) + list(gan.vocos_mrd.discriminators)
    n_dac = sum(len(stack) for d in gan.dac_disc.discriminators for stack in d.band_convs)
    assert len(fused) == 5 * len(subs) + n_dac
    assert all(d.fused_act for d in subs) and all(m.act_inplace and m.act_slope == 0.1 for m in fused)
    assert not any(isinstance(m, nn.LeakyReLU) for m in gan.modules())
    posts = [d.conv_post for d in subs + list(gan.dac_disc.discriminators)]
    assert all(type(p) is nn.Conv2d for p in posts)                       # the logit convs keep their bias add


def test_fused_bias_act_checkpoints_load_both_ways():
    fused, ref = _gan("c0", fused_bias_act=True), _gan("c0", seed=5)
    fused.load_state_dict(ref.state_dict())
    assert _equal(fused.state_dict().values(), ref.state_dict().values())
    _gan("c0").load_state_dict(fused.state_dict())


def test_fused_bias_act_needs_the_tested_triton(monkeypatch):
    """Another Triton version than TRITON_VERSION (or none): one warning, then the plain path for every
    conv, with the same outputs and gradients. CPU: the CUDA checks of _fused see a stand-in input."""
    from sage.nn.discriminators import fused_act
    pytest.importorskip("triton")
    cuda_input = types.SimpleNamespace(is_cuda=True, dtype=torch.float32, numel=lambda: 1)  # what is checked
    warned = []
    monkeypatch.setattr(fused_act, "warn", lambda msg, **kw: warned.append(msg))
    fused = _gan("c0", fused_bias_act=True, fused_bias_grad=True)
    conv = next(m for m in fused.modules() if isinstance(m, BiasLeakyReLUConv2d))
    monkeypatch.setattr(fused_act.triton, "__version__", fused_act.TRITON_VERSION)
    monkeypatch.setattr(fused_act, "_triton_ok", None)
    assert conv._fused(cuda_input) and fused_act._kernel_ok(cuda_input) and not warned
    for version in ("0.0.0", None):                          # another version; Triton not importable
        monkeypatch.setattr(fused_act, "_triton_ok", None)
        if version is None:
            monkeypatch.setattr(fused_act, "triton", None)
        else:
            monkeypatch.setattr(fused_act.triton, "__version__", version)
        assert not conv._fused(cuda_input) and not conv._fused(cuda_input)
        assert not fused_act._kernel_ok(cuda_input) and len(warned) == 1
        warned.clear()
    _assert_same_critic(_gan("c0"), fused, *_pair())


def test_fused_bias_grad_needs_fused_bias_act():
    with pytest.raises(ValueError, match="fused_bias_grad"):
        _gan("c0", fused_bias_grad=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last])
def test_fused_bias_act_kernels_match_torch_on_cuda(memory_format):
    from sage.nn.discriminators import fused_act
    torch.manual_seed(0)
    z = torch.randn(3, 37, 11, 5, device="cuda").contiguous(memory_format=memory_format)
    z.as_strided((z.numel(),), (1,))[:6] = torch.tensor([0.0, -0.0, 1e-41, -1e-41, float("nan"), -float("inf")])
    b = torch.randn(37, device="cuda")
    ref = torch.nn.functional.leaky_relu(z + b.view(1, -1, 1, 1), 0.1)
    for _ in range(2):                          # first call through Triton's JIT, then the compiled launcher
        y = fused_act._bias_leaky_relu_forward(z, b, 0.1)
        assert torch.equal(y.isnan(), ref.isnan()) and torch.equal(y.nan_to_num(), ref.nan_to_num())
        assert y.stride() == z.stride()
    g = torch.randn_like(y)
    gz_ref = torch.ops.aten.leaky_relu_backward(g, y.nan_to_num(), 0.1, True)
    for _ in range(2):
        out = fused_act._leaky_relu_backward_bias_sum(g, y.nan_to_num(), 0.1)
        if memory_format is torch.contiguous_format:
            assert out is None                                            # NCHW: autograd's two kernels
            return
        assert torch.equal(out[0], gz_ref)
        assert torch.allclose(out[1], gz_ref.double().sum((0, 2, 3)).float(), rtol=1e-5, atol=1e-5)
    assert any(k[0] is fused_act._leaky_relu_bwd_bias_sum for k in fused_act._COMPILED)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
@pytest.mark.parametrize("keys", list(FUSED_KEYS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_fused_bias_act_is_bit_identical_on_cuda(layout, keys):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ref = _layout_gan(layout, **FUSED_KEYS[keys]).cuda()
    fused = _layout_gan(layout, **FUSED_KEYS[keys], fused_bias_act=True).cuda()
    _assert_same_critic(ref, fused, *(t.cuda() for t in _layout_pair(layout)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_fused_bias_grad_changes_only_the_bias_gradients_on_cuda(layout):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    keys = dict(inplace_act=True, channels_last=True)
    ref = _layout_gan(layout, **keys, fused_bias_act=True).cuda()
    fused = _layout_gan(layout, **keys, fused_bias_act=True, fused_bias_grad=True).cuda()
    real, fake = (t.cuda() for t in _layout_pair(layout))
    (la, ga, _), (lb, gb, _) = _gen_step(ref, real, fake, True), _gen_step(fused, real, fake, True)
    assert _equal(la, lb) and _rel(gb, ga) < 1e-6                         # no bias gradient in the G step
    assert _equal(_conv_input_grads(ref, real, fake), _conv_input_grads(fused, real, fake))
    a = _dis_grads(ref, lambda g, r, f: g.dis_loss(r, f), real, fake)
    b = _dis_grads(fused, lambda g, r, f: g.dis_loss(r, f), real, fake)
    assert torch.equal(a[0], b[0])
    fused_bias = [id(p) in {id(m.bias) for m in fused.modules() if isinstance(m, BiasLeakyReLUConv2d)}
                  for p in fused.parameters()]
    assert all(torch.equal(x, y) for x, y, is_b in zip(a[1], b[1], fused_bias) if not is_b)
    # biases: another summation order (production critics: 1.4e-5 to 2.1e-5 apart as one vector, and the
    # fused sums are the closer ones to a float64 sum of the same gradient, local/disc_speed/fused)
    assert _norm_rel([y for y, is_b in zip(b[1], fused_bias) if is_b],
                     [x for x, is_b in zip(a[1], fused_bias) if is_b]) < 1e-4



# ── EMA as multi-tensor ops ──────────────────────────────────────────────────

def test_ema_foreach_is_bit_identical():
    from sage.training.callbacks import EMACallback
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
    model(torch.randn(4, 8))                                             # num_batches_tracked: an int buffer
    module = types.SimpleNamespace(autoencoder=model, _ema_update_this_batch=True)
    cbs = [EMACallback(decay=0.9998, foreach=f) for f in (False, True)]
    for cb in cbs:
        cb.on_fit_start(None, module)
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn_like(p))
        model(torch.randn(4, 8))
        for cb in cbs:
            cb.on_train_batch_end(None, module, None, None, 0)
    a, b = cbs[0].ema_state_dict, cbs[1].ema_state_dict
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert int(b["1.num_batches_tracked"]) == 4



# ── +speedup overrides ───────────────────────────────────────────────────────

CONFIGS = Path(__file__).parents[1] / "configs"
CRITIC_SPEED_KEYS = ("inplace_act", "fused_feature_matching", "channels_last", "branch_streams")
FUSED_SPEED_KEYS = ("fused_bias_act", "fused_bias_grad")
TRAINER_SPEED_KEYS = ("freeze_disc_in_gen_step", "disc_step_loss_only", "ema_foreach", "log_every_n_batches",
                      "progress_bar_refresh_rate", "lr_monitor", "detach_step_output", "ddp_skip_idle_buckets",
                      "ddp_gradient_as_bucket_view", "ddp_broadcast_buffers")
# Recipes with the speed-ups on, and whether they keep the fused keys (fold critics only).
RECIPES = {"pretrain": True, "decoder_ft": True, "stereo_proxy": True, "stereo_proxy_a1": True,
           "stereo_proxy_a3": False, "stereo_proxy_a3_fm": False,
           "stereo_proxy_ft": False}


def _compose(*overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = OmegaConf.to_container(compose("main", list(overrides)), resolve=False)
    return cfg, cfg["trainer"]["trainer"], cfg["trainer"]["loss_config"]["discriminator"]["config"]


def _yaml_keys(d, prefix=""):
    """Dotted leaf keys of a speedup file (its defaults list aside)."""
    out = set()
    for k, v in d.items():
        if k != "defaults":
            out |= _yaml_keys(v, f"{prefix}{k}.") if isinstance(v, dict) else {prefix + k}
    return out


@pytest.mark.parametrize("arm", ["stereo_proxy", "stereo_proxy_a3", "pretrain"])
@pytest.mark.parametrize("speedup", ["critic_fast", "critic_fast_fused"])
def test_speedup_override_changes_only_speed_keys(arm, speedup):
    """+speedup=... adds the speed keys to the plain recipe (+speedup=legacy) and changes nothing else."""
    ref, ref_t, ref_d = _compose(f"+experiment={arm}", "+speedup=legacy")
    fast, t, d = _compose(f"+experiment={arm}", f"+speedup=[legacy,{speedup}]")
    assert t["freeze_disc_in_gen_step"] and t["disc_step_loss_only"] and t["ema_foreach"]
    keys = CRITIC_SPEED_KEYS + (FUSED_SPEED_KEYS if speedup.endswith("fused") else ())
    assert all(d[k] is True for k in keys)
    assert speedup.endswith("fused") or d["fused_bias_act"] is False
    for k in ("freeze_disc_in_gen_step", "disc_step_loss_only", "ema_foreach"):
        t[k] = ref_t[k]
    for k in keys:
        d[k] = ref_d[k]
    assert fast == ref


# ── speed-ups on in the recipes, +speedup=legacy turns them off ─────────────

@pytest.mark.parametrize("recipe", list(RECIPES))
def test_recipe_has_the_speedups_on(recipe):
    """The recipe = recipe + legacy + critic_fast_fused + loop_fast, except the fused keys, off on the joint
    critics: speedup/recipe.yaml holds the keys of those two files, and they still apply after legacy."""
    rec, t, d = _compose(f"+experiment={recipe}")
    fast, _, fast_d = _compose(f"+experiment={recipe}", "+speedup=[legacy,critic_fast_fused,loop_fast]")
    assert t["freeze_disc_in_gen_step"] and t["log_every_n_batches"] == 1 and t["ddp_skip_idle_buckets"]
    assert all(d[k] is True for k in CRITIC_SPEED_KEYS)
    assert all(d[k] is RECIPES[recipe] for k in FUSED_SPEED_KEYS)
    for k in FUSED_SPEED_KEYS:
        fast_d[k] = d[k]
    assert fast == rec


@pytest.mark.parametrize("recipe", list(RECIPES))
def test_legacy_turns_every_speedup_off(recipe):
    """+speedup=legacy: every speed key at its plain value (trainer.yaml, the critic's constructor), which
    the recipes had before the speed-ups; nothing else of the recipe changes."""
    from omegaconf import OmegaConf
    plain_t = OmegaConf.load(CONFIGS / "trainer.yaml").trainer
    plain_d = {n: p.default for cls in (WavTokenizerGANLoss, WavTokenizerDiscriminator)
               for n, p in inspect.signature(cls.__init__).parameters.items()}
    rec, rec_t, rec_d = _compose(f"+experiment={recipe}")
    leg, t, d = _compose(f"+experiment={recipe}", "+speedup=legacy")
    assert all(t[k] == plain_t[k] for k in TRAINER_SPEED_KEYS)
    assert all(d[k] is plain_d[k] is False for k in CRITIC_SPEED_KEYS + FUSED_SPEED_KEYS)
    for k in TRAINER_SPEED_KEYS:
        t[k] = rec_t[k]
    for k in CRITIC_SPEED_KEYS + FUSED_SPEED_KEYS:
        d[k] = rec_d[k]
    assert leg == rec


def test_legacy_covers_every_speedup_key():
    """A speed key added to an override must be added to legacy.yaml too."""
    from omegaconf import OmegaConf
    files = [OmegaConf.to_container(OmegaConf.load(f)) for f in sorted((CONFIGS / "speedup").glob("*.yaml"))]
    legacy = OmegaConf.to_container(OmegaConf.load(CONFIGS / "speedup" / "legacy.yaml"))
    assert set().union(*map(_yaml_keys, files)) == _yaml_keys(legacy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
