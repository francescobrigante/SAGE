# ===============
# O1 tests: phase-aware compute(). Tier 1 = gen losses only on gen steps, disc loss
# only on disc steps. Tier 2 = generator forward under no_grad on disc steps, so the
# disc backward never reaches the generator. A stub AE/LossManager isolates the routing.
# ===============
import types

import pytest
import torch
import torch.nn as nn

from sage.training.engine import AutoencoderEngine


class _StubAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.encoder = types.SimpleNamespace(is_complex=False)
        self.has_pre_transform = False
        self.pre_transform_applies_to_target = False
        self.pre_transform_applies_inverse = False
        self.n_encode = 0

    def encode(self, x, return_info=False, **kw):
        self.n_encode += 1
        B = x.shape[0]
        latents = x.reshape(B, 4, -1)[..., :4] * self.w
        info = {"feature_shape": (2, 2)}
        return (latents, info) if return_info else latents

    def decode(self, z, encoder_info=None, apply_inverse=False):
        B = z.shape[0]
        return z.mean(dim=2)[:, :, None, None].expand(B, 4, 16, 8) * self.w

    def istft(self, sp, target_length=None):
        B = sp.shape[0]
        N = int(target_length or 64)
        return sp.mean(dim=(2, 3))[:, :2, None].expand(B, 2, N) * self.w

    def stft(self, wav):
        B = wav.shape[0]
        return torch.randn(B, 4, 16, 8)

    def _pack_complex(self, sp):
        return sp


class _StubLM(nn.Module):
    def __init__(self, use_disc=True):
        super().__init__()
        self.dw = nn.Parameter(torch.ones(()))
        self.use_disc = use_disc
        self.gen_calls = 0
        self.disc_calls = 0
        self.loss_calls = 0
        self.discriminator = self

    def loss(self, reals, fakes):
        self.loss_calls += 1
        loss_dis = self.dw * fakes.mean() + self.dw * reals.mean()
        loss_adv = self.dw * fakes.mean()
        fm = self.dw * fakes.mean()
        return loss_dis, loss_adv, fm

    def losses_gen(self, info):
        self.gen_calls += 1
        total = info["decoded"].mean()
        total = total + info.get("loss_adv", torch.zeros(()))
        total = total + info.get("feature_matching_distance", torch.zeros(()))
        return total, {"recon": info["decoded"].mean()}

    def losses_disc(self, info):
        self.disc_calls += 1
        return info["loss_dis"], {"discriminator_loss": info["loss_dis"]}


def _make_engine(use_disc=True, warmup_steps=0):
    eng = AutoencoderEngine.__new__(AutoencoderEngine)
    nn.Module.__init__(eng)
    eng.autoencoder = _StubAE()
    eng.loss_manager = _StubLM(use_disc=use_disc)
    eng.teacher_model = None
    eng.use_disc = use_disc
    eng.warmup_steps = warmup_steps
    eng.warmup_mode = "adv"
    eng.encoder_freeze_on_warmup = False
    eng.freeze_encoder = False
    eng.force_input_mono = False
    eng.latent_mask_ratio = 0.0
    eng.audio_channels = 2
    eng.sample_rate = 44100
    eng.stft_params = {"n_fft": 2048, "hop_length": 512}
    return eng


def _batch():
    return torch.randn(2, 4, 16, 8), torch.randn(2, 2, 64)


# ── Tier 1: loss routing per phase ────────────────────────────────────────────
def test_gen_step_runs_only_gen_losses():
    eng = _make_engine()
    out = eng.compute(_batch(), gen_step=10, disc_phase=False)
    assert out["phase"] == "gen"
    assert eng.loss_manager.gen_calls == 1
    assert eng.loss_manager.disc_calls == 0
    assert out["gen_total"] is not None
    assert out["disc_total"] is None
    assert out["disc_breakdown"] == {}


def test_disc_step_runs_only_disc_loss():
    eng = _make_engine()
    out = eng.compute(_batch(), gen_step=10, disc_phase=True)
    assert out["phase"] == "disc"
    assert eng.loss_manager.disc_calls == 1
    assert eng.loss_manager.gen_calls == 0
    assert out["disc_total"] is not None
    assert out["gen_total"] is None
    assert out["gen_breakdown"] == {}


def test_feature_matching_only_in_gen_step():
    eng = _make_engine()
    gen = eng.compute(_batch(), gen_step=10, disc_phase=False)["loss_info"]
    disc = eng.compute(_batch(), gen_step=10, disc_phase=True)["loss_info"]
    assert "loss_adv" in gen and "feature_matching_distance" in gen
    assert "loss_dis" not in gen
    assert "loss_dis" in disc
    assert "loss_adv" not in disc and "feature_matching_distance" not in disc


# ── Tier 2: no generator graph / backward on disc steps ───────────────────────
def test_disc_step_generator_forward_is_no_grad():
    eng = _make_engine()
    out = eng.compute(_batch(), gen_step=10, disc_phase=True)
    assert out["loss_info"]["decoded"].requires_grad is False
    assert out["loss_info"]["latents"].requires_grad is False


def test_disc_loss_does_not_backprop_into_generator():
    eng = _make_engine()
    out = eng.compute(_batch(), gen_step=10, disc_phase=True)
    g = torch.autograd.grad(out["disc_total"], eng.autoencoder.w, allow_unused=True)[0]
    assert g is None
    g_dw = torch.autograd.grad(out["disc_total"], eng.loss_manager.dw, retain_graph=True)[0]
    assert g_dw is not None


def test_gen_step_keeps_generator_graph():
    eng = _make_engine()
    out = eng.compute(_batch(), gen_step=10, disc_phase=False)
    assert out["loss_info"]["decoded"].requires_grad is True
    g = torch.autograd.grad(out["gen_total"], eng.autoencoder.w, retain_graph=True)[0]
    assert g is not None


# ── warmup gating preserved ───────────────────────────────────────────────────
def test_gen_step_during_warmup_skips_discriminator():
    eng = _make_engine(warmup_steps=100)
    out = eng.compute(_batch(), gen_step=0, disc_phase=False)
    assert out["phase"] == "gen"
    assert eng.loss_manager.loss_calls == 0
    assert float(out["loss_info"]["loss_adv"]) == 0.0
    assert float(out["loss_info"]["feature_matching_distance"]) == 0.0


def test_disc_step_during_adv_warmup_trains_disc():
    eng = _make_engine(warmup_steps=100)
    out = eng.compute(_batch(), gen_step=0, disc_phase=True)
    assert out["phase"] == "disc"
    assert eng.loss_manager.loss_calls == 1
    assert out["disc_total"] is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
