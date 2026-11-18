# ===============
# Smoke tests of trainer.trainer.compile_generator on a tiny SAGE (CPU, 2 batches, backend aot_eager:
# inductor on CPU is too slow for the suite, the GPU check runs it): the compiled run trains the same
# weights as the eager one, its checkpoint and EMA keys carry no `_orig_mod.`, it resumes, and a checkpoint
# of a run with the old trainer.trainer.compile (keys `_orig_mod.`) resumes with its EMA and initializes a
# decoder fine-tuning.
# ===============
from __future__ import annotations

from functools import partial

import pytest
import torch

from conftest import RUN, SMOKE_DISC, TINY_MODEL, _data_overrides, run_training

pytestmark = pytest.mark.smoke

ON = ["trainer.trainer.compile_generator=true"]
TWO_EPOCHS = [o for o in RUN if not o.startswith("trainer.trainer.epochs=")] + ["trainer.trainer.epochs=2"]
WEIGHT_RTOL = 1e-5        # Σ|on − off| / Σ|off| over every trained tensor (aot_eager: same ops, ~0 measured)


def _pretrain(name):
    return ["+experiment=pretrain", "data=fma", "trainer.loss_config.semantic_distill.detach_warmup_steps=0",
            f"trainer.wandb.name={name}"] + TINY_MODEL + SMOKE_DISC


@pytest.fixture
def compiled_calls(monkeypatch):
    """train.py compiles with aot_eager; returns the list of compiled-module calls (one entry per call)."""
    import train
    import sage.training.compile as compile_mod
    monkeypatch.setattr(train, "compile_generator", partial(compile_mod.compile_generator, backend="aot_eager"))
    calls = []
    original = compile_mod.CompiledCall.__call__

    def counted(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(compile_mod.CompiledCall, "__call__", counted)
    torch._dynamo.reset()
    yield calls
    torch._dynamo.reset()


def _state(ckpt):
    return torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]


def _rel(a, b):
    return float((a - b).abs().sum() / b.abs().sum().clamp_min(1e-30))


def test_compiled_training_matches_eager_training(audio_dir, tmp_path, compiled_calls):
    base = _pretrain("smoke_compile") + _data_overrides(audio_dir) + RUN
    (tmp_path / "off").mkdir(), (tmp_path / "on").mkdir()
    off = _state(run_training(base, tmp_path / "off"))
    assert compiled_calls == []
    torch._dynamo.reset()
    on = _state(run_training(base + ON, tmp_path / "on"))
    assert {"encoder", "decoder"} <= set(compiled_calls)
    assert on.keys() == off.keys() and not any("_orig_mod" in k for k in on)
    assert any(k.startswith("ema_autoencoder.encoder.") for k in on)
    trained = [k for k in on if k.startswith(("engine.autoencoder.", "ema_autoencoder.", "engine.loss_manager."))
               and on[k].is_floating_point()]
    worst = max(trained, key=lambda k: _rel(on[k], off[k]))
    assert _rel(on[worst], off[worst]) < WEIGHT_RTOL, (worst, _rel(on[worst], off[worst]))


def _old_style(sd):
    """The keys a run with the old trainer.trainer.compile saved; EMA weights moved by +1 (a marker)."""
    old = {}
    for k, v in sd.items():
        if k.startswith("ema_autoencoder."):
            marked = v + 1 if k.endswith((".weight", ".bias")) and v.is_floating_point() else v
            old["ema_autoencoder._orig_mod." + k[len("ema_autoencoder."):]] = marked
        elif k.startswith("engine.autoencoder."):
            old["engine.autoencoder._orig_mod." + k[len("engine.autoencoder."):]] = v
        else:
            old[k] = v
    return old


def test_resume_and_init_from_with_old_and_new_checkpoints(audio_dir, tmp_path, compiled_calls, monkeypatch):
    import sage.training.callbacks as callbacks
    ema_errors = []
    monkeypatch.setattr(callbacks, "err", lambda msg, prefix="": ema_errors.append(msg))
    data = _data_overrides(audio_dir)

    # a compiled run resumes (relaunch of its chosen name) with the compiled model
    first_path = run_training(_pretrain("smoke_compile_resume") + data + RUN + ON, tmp_path)
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    torch._dynamo.reset()
    second = torch.load(run_training(_pretrain("smoke_compile_resume") + data + TWO_EPOCHS + ON, tmp_path),
                        map_location="cpu", weights_only=False)
    assert (first["epoch"], second["epoch"]) == (0, 1)
    assert second["global_step"] == 2 * first["global_step"] > 0
    assert not any("_orig_mod" in k for k in second["state_dict"])

    # a checkpoint with the old _orig_mod. keys: full resume into an eager run, EMA kept (not re-initialized)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_ckpt = old_dir / "old_compile.ckpt"
    torch.save({**first, "state_dict": _old_style(first["state_dict"])}, old_ckpt)
    marker = {k[len("ema_autoencoder._orig_mod."):]: v for k, v in _old_style(first["state_dict"]).items()
              if k.startswith("ema_autoencoder.")}
    torch._dynamo.reset()
    resumed = _state(run_training(_pretrain("smoke_compile_old") + data + TWO_EPOCHS + [f"+ckpt_path={old_ckpt}"],
                                  old_dir))
    assert not any("_orig_mod" in k for k in resumed) and not ema_errors
    ema = {k[len("ema_autoencoder."):]: v for k, v in resumed.items() if k.startswith("ema_autoencoder.")}
    assert ema.keys() == marker.keys()
    moved = [k for k in marker if k.endswith((".weight", ".bias")) and marker[k].is_floating_point()]
    assert moved and max(float((ema[k] - marker[k]).abs().max()) for k in moved) < 1e-2   # one EMA step from it

    # ... and initializes a decoder fine-tuning (compiled): the frozen encoder holds the old EMA weights
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir()
    torch._dynamo.reset()
    ft = _state(run_training(["+experiment=decoder_ft", "data=fma", "trainer.wandb.name=smoke_compile_ft",
                              f"+init_from={old_ckpt}"] + TINY_MODEL + SMOKE_DISC + data + RUN + ON, ft_dir))
    enc = [k for k in marker if k.startswith("encoder.")]
    assert enc and all(torch.equal(ft["engine.autoencoder." + k], marker[k]) for k in enc)
