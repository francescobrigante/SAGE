# ===============
# Smoke tests of the stereo-discriminator proxy recipes on a tiny SAGE (CPU, 2 batches): every arm
# trains and writes a checkpoint with the diagnostics on, and the diagnostics are read-only (same
# seed, diagnostics on or off -> identical weights of autoencoder and discriminator).
# ===============
from __future__ import annotations

import pytest
import torch

from conftest import RUN, SMOKE_DISC, TINY_MODEL, _data_overrides, run_training

pytestmark = pytest.mark.smoke

DIAG_ON = ["trainer.trainer.critic_diag_every_n_epochs=1", "trainer.trainer.adv_grad_every=1"]
DIAG_OFF = ["trainer.trainer.critic_diag_every_n_epochs=0", "trainer.trainer.adv_grad_every=0"]
GATE_OPEN = ["trainer.loss_config.semantic_distill.detach_warmup_steps=0"]


def _arm(arm: str) -> list[str]:
    exp = "stereo_proxy" if arm == "c0" else f"stereo_proxy_{arm}"
    return [f"+experiment={exp}", f"trainer.wandb.name=smoke_{arm}"]


def _weights(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    return {k: v for k, v in sd.items() if k.startswith(("engine.autoencoder.", "engine.loss_manager.discriminator."))}


# A4 keeps its own scales (only the MPD is cut to one period); the others use the one-per-branch critic.
SCALES = {"a4": ["++trainer.loss_config.discriminator.config.periods=[2]"]}


@pytest.mark.parametrize("arm", ["c0", "a1", "a2", "a3", "a4"])
def test_arm_trains_with_the_diagnostics_on(arm, audio_dir, tmp_path):
    disc = SCALES.get(arm, SMOKE_DISC)
    ckpt = run_training(_arm(arm) + GATE_OPEN + TINY_MODEL + disc + _data_overrides(audio_dir) + RUN + DIAG_ON,
                        tmp_path)
    w = _weights(ckpt)
    assert any(k.startswith("engine.loss_manager.discriminator.") for k in w)
    assert all(torch.isfinite(v).all() for v in w.values() if v.is_floating_point())


def _count_calls(monkeypatch, module, name, calls):
    """Wrap module.name so that every call is counted in calls[name] (the original still runs)."""
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, counted)


def test_diagnostics_do_not_change_what_is_trained(audio_dir, tmp_path, monkeypatch):
    import sage.training.engine as engine
    import sage.training.stereo_diagnostics as diag
    calls = {"adversarial_gradient_share": 0, "critic_stereo_report": 0}
    _count_calls(monkeypatch, engine, "adversarial_gradient_share", calls)
    _count_calls(monkeypatch, diag, "critic_stereo_report", calls)
    base = _arm("a2") + GATE_OPEN + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir) + RUN
    (tmp_path / "on").mkdir(), (tmp_path / "off").mkdir()
    on = _weights(run_training(base + DIAG_ON, tmp_path / "on"))
    ran = dict(calls)
    assert all(n > 0 for n in ran.values()), ran                             # the diagnostics did run
    off = _weights(run_training(base + DIAG_OFF, tmp_path / "off"))
    assert calls == ran, calls                                               # and are off by default
    assert on.keys() == off.keys()
    assert all(torch.equal(on[k], off[k]) for k in on), [k for k in on if not torch.equal(on[k], off[k])][:5]
