# ===============
# Smoke tests of the two training phases of the paper on a tiny SAGE (CPU, a few batches):
# phase 1 pretraining (all losses + CLAP distillation + WavTokenizer discriminator), phase 2
# decoder fine-tuning (EMA init, frozen encoder, zero-init post-net, fresh discriminator).
# ===============
from __future__ import annotations

import pytest
import torch

from conftest import PRETRAIN, RUN, SMOKE_DISC, TINY_MODEL, _data_overrides, run_training

PRETRAIN_RESUME = [o for o in PRETRAIN if not o.startswith("trainer.wandb.name=")] + ["trainer.wandb.name=smoke_resume"]

pytestmark = pytest.mark.smoke


def _state(ckpt, prefix):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def test_pretrain_writes_a_loadable_checkpoint(tiny_pretrain_ckpt):
    ck = torch.load(tiny_pretrain_ckpt, map_location="cpu", weights_only=False)
    assert "inference_config" in ck
    ema = _state(tiny_pretrain_ckpt, "ema_autoencoder.")
    assert any(k.startswith("encoder.") for k in ema) and any(k.startswith("decoder.") for k in ema)
    # the paper losses were all instantiated: discriminator, projection head of the CLAP distillation
    keys = ck["state_dict"].keys()
    assert any(k.startswith("engine.loss_manager.discriminator.") for k in keys)
    assert any("distill_proj" in k for k in keys)
    assert all(torch.isfinite(v).all() for v in ema.values() if v.is_floating_point())
    # 2 batches under the G/D alternation = 1 generator update; the generator LR schedule agrees,
    # while Lightning's global_step counts every optimizer step (generator + CLAP head + disc)
    assert ck["gen_step"] == 1 and ck["lr_schedulers"][0]["last_epoch"] == 1
    assert ck["global_step"] == 3


FINETUNE = [
    "+experiment=decoder_ft",                 # paper decoder fine-tuning recipe (Table 6)
    "data=fma",
    "trainer.wandb.name=smoke_decoder_ft",
]


def test_decoder_finetune_freezes_the_encoder(tiny_pretrain_ckpt, audio_dir, tmp_path):
    ft_ckpt = run_training(
        FINETUNE + [f"+init_from={tiny_pretrain_ckpt}"] + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir) + RUN,
        tmp_path,
    )
    src = _state(tiny_pretrain_ckpt, "ema_autoencoder.")
    live = _state(ft_ckpt, "engine.autoencoder.")
    enc = [k for k in src if k.startswith("encoder.")]
    assert enc and all(torch.equal(src[k], live[k]) for k in enc), "encoder changed during fine-tuning"
    assert any(k.startswith("decoder.postnet.") for k in live), "post-net missing"
    dec = [k for k in src if k.startswith("decoder.") and k in live]
    assert any(not torch.equal(src[k], live[k]) for k in dec), "decoder did not train"


def test_relaunching_a_run_resumes_it_in_its_folder(audio_dir, tmp_path):
    # A SLURM requeue or a resubmission relaunches the same run name: it continues the newest
    # checkpoint in the same runs/<name>/<date>/ (and the same W&B run), as the paper's scripts did.
    base = PRETRAIN_RESUME + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir)
    first = torch.load(run_training(base + RUN, tmp_path), map_location="cpu", weights_only=False)
    two_epochs = run_training(base + [o for o in RUN if not o.startswith("trainer.trainer.epochs=")]
                              + ["trainer.trainer.epochs=2"], tmp_path)   # overwrites last.ckpt
    second = torch.load(two_epochs, map_location="cpu", weights_only=False)
    assert (first["epoch"], second["epoch"]) == (0, 1)
    assert second["global_step"] == 2 * first["global_step"] > 0
    assert len(list((tmp_path / "runs" / "smoke_resume").iterdir())) == 1          # one folder per run
    run_training(base + RUN + ["auto_resume=false"], tmp_path)                        # starts over
    assert len(list((tmp_path / "runs" / "smoke_resume").iterdir())) == 2


def test_relaunching_with_a_new_init_from_or_a_derived_name(tiny_pretrain_ckpt, audio_dir, tmp_path, monkeypatch):
    # A run that has checkpoints refuses a new +init_from instead of silently resuming over it,
    # except in a SLURM requeue (same job, same command); a name derived from the config is not
    # resumed, since different experiments can share it.
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    ft = FINETUNE + [f"+init_from={tiny_pretrain_ckpt}"] + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir)
    first = torch.load(run_training(ft + RUN, tmp_path), map_location="cpu", weights_only=False)
    two_epochs = [o for o in RUN if not o.startswith("trainer.trainer.epochs=")] + ["trainer.trainer.epochs=2"]
    with pytest.raises(SystemExit, match="drop \\+init_from"):
        run_training(ft + two_epochs, tmp_path)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    requeued = torch.load(run_training(ft + two_epochs, tmp_path), map_location="cpu", weights_only=False)
    assert requeued["global_step"] == 2 * first["global_step"] > 0                   # continued, not restarted
    assert len(list((tmp_path / "runs" / "smoke_decoder_ft").iterdir())) == 1

    monkeypatch.delenv("SLURM_RESTART_COUNT")
    derived = PRETRAIN_RESUME[:-1] + ["trainer.wandb.name=null"] + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir)
    out = tmp_path / "derived"
    out.mkdir()
    run_training(derived + RUN, out)
    run_training(derived + RUN, out)
    (name,) = [d.name for d in (out / "runs").iterdir()]
    assert len(list((out / "runs" / name).iterdir())) == 2                           # started over
