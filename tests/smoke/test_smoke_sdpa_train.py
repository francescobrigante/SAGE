# ===============
# Smoke test of trainer.trainer.sdpa_train + patch_linear on a tiny SAGE (CPU, eager, 2 batches = one generator
# and one discriminator update): train.py turns them on, the run takes the SDPA training path, and its losses
# and trained weights match the run with the keys off (same formula, float32 rounding apart). Exactness in
# float64: tests/test_sdpa_train.py.
# ===============
from __future__ import annotations

import pytest
import torch

from conftest import RUN, SMOKE_DISC, TINY_MODEL, _data_overrides, run_training

pytestmark = pytest.mark.smoke

ON = ["trainer.trainer.sdpa_train=true", "trainer.trainer.patch_linear=true"]
LOSS_RTOL = 1e-4          # |on − off| / |off| per backward loss (float32)
# Σ|on − off| / Σ|off| over all trained tensors. Adam's first steps move each weight by ≈ ±lr whatever |g|, so
# rounding flips the step of the weights whose gradient is ~0 (per-tensor ratios blow up on near-zero tensors);
# a wrong gradient would flip about half of them (≈ lr / mean|w| over all weights).
WEIGHT_RTOL = 1e-5


@pytest.fixture
def recorded(monkeypatch):
    """Losses passed to manual_backward and calls of the SDPA training path, per run."""
    import sage.model.swin.attention as attention_module
    from sage.training.lightning_module import SAGELightningModule
    log = {"losses": [], "sdpa_train": 0}
    backward, train_forward = SAGELightningModule.manual_backward, attention_module.sdpa_train_forward

    def manual_backward(self, loss, *args, **kwargs):
        log["losses"].append(float(loss.detach()))
        return backward(self, loss, *args, **kwargs)

    def sdpa_train_forward(*args, **kwargs):
        log["sdpa_train"] += 1
        return train_forward(*args, **kwargs)
    monkeypatch.setattr(SAGELightningModule, "manual_backward", manual_backward)
    monkeypatch.setattr(attention_module, "sdpa_train_forward", sdpa_train_forward)
    return log


def test_sdpa_train_matches_the_original_training(audio_dir, tmp_path, recorded):
    base = (["+experiment=pretrain", "data=fma", "trainer.loss_config.semantic_distill.detach_warmup_steps=0",
             "trainer.wandb.name=smoke_sdpa_train"] + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir) + RUN)
    (tmp_path / "off").mkdir(), (tmp_path / "on").mkdir()
    off_ckpt = run_training(base, tmp_path / "off")
    off_losses = recorded["losses"][:]
    assert recorded["sdpa_train"] == 0 and off_losses
    recorded["losses"].clear()
    on_ckpt = run_training(base + ON, tmp_path / "on")
    assert recorded["sdpa_train"] > 0
    assert len(recorded["losses"]) == len(off_losses)
    assert all(abs(a - b) <= LOSS_RTOL * abs(b) for a, b in zip(recorded["losses"], off_losses)), \
        (recorded["losses"], off_losses)
    on, off = (torch.load(c, map_location="cpu", weights_only=False)["state_dict"] for c in (on_ckpt, off_ckpt))
    assert on.keys() == off.keys()
    trained = [k for k in on if k.startswith(("engine.autoencoder.", "ema_autoencoder.", "engine.loss_manager."))
               and on[k].is_floating_point()]
    diff = sum(float((on[k] - off[k]).abs().sum()) for k in trained)
    total = sum(float(off[k].abs().sum()) for k in trained)
    assert diff < WEIGHT_RTOL * total, diff / total
