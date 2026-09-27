# ===============
# Smoke tests of the two training phases of the paper on a tiny SAGE (CPU, a few batches):
# phase 1 pretraining (all losses + CLAP distillation + WavTokenizer discriminator), phase 2
# decoder fine-tuning (EMA init, frozen encoder, zero-init post-net, fresh discriminator).
# ===============
from __future__ import annotations

import pytest
import torch

from conftest import RUN, SMOKE_DISC, TINY_MODEL, _data_overrides, run_training

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
