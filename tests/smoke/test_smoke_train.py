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


FINETUNE = [
    "models=swin_real_swiglu_xsa",
    "data=fma",
    "models.model.decoder.use_postnet=true",
    "+init_from_ema=true",
    "+init_from_disc=false",
    "trainer.trainer.freeze_encoder=true",
    "trainer.trainer.warmup_steps=0",
    "trainer.loss_config.bottleneck.weights.kl=0.0",
    "trainer.loss_config.spectral.weights.stft_mse=1.0",
    "trainer.loss_config.spectral.weights.mrstft=0.0",
    "trainer.loss_config.mrmel.weights.mrmel=0.5",
    "trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0",
    "++trainer.loss_config.discriminator.config.fold_lrms=true",
    "trainer.optimizer.lr=1e-4",
    "trainer.scheduler.inv_gamma=200000",
    "++trainer.disc_optimizer._target_=torch.optim.AdamW",
    "++trainer.disc_optimizer.lr=2e-4",
    "++trainer.disc_optimizer.betas=[0.8,0.99]",
    "++trainer.disc_optimizer.weight_decay=1e-3",
    "++trainer.loss_config.discriminator.type=wavtokenizer",
    "++trainer.loss_config.discriminator.config.loss_type=rpgan",
    "++trainer.loss_config.discriminator.config.preprocess=true",
    "++trainer.loss_config.discriminator.weights.adversarial=0.1",
    "++trainer.loss_config.discriminator.weights.feature_matching=0.2",
    "trainer.trainer.precision=32",
    "++trainer.trainer.ema_decay=0.9998",
    "trainer.wandb.name=smoke_decoder_ft",   # the auto-generated run name exceeds 255 chars (bug B8)
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
