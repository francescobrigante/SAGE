# ===============
# sage.compat: every class path a pre-release checkpoint can contain maps to an importable class,
# the upgrade touches only class paths, and the real paper checkpoint config rebuilds a model whose
# state_dict keys and shapes are exactly the recorded ones (tests/smoke checks the loaded weights).
# ===============
import importlib
import json
import os
from pathlib import Path

import pytest

from sage.compat import LEGACY_TARGETS, upgrade_class_path, upgrade_model_config

REPO = Path(__file__).resolve().parents[1]
# The paper's TRAINING checkpoint (optimizer states and loss buffers, for the resume test) and the
# parameter names of a golden reference; both live outside the repository.
PAPER_CKPT = Path(os.environ.get("SAGE_TRAIN_CKPT", REPO / "sage_release_context" / "SAGE_FTe992.ckpt"))
STATE_KEYS = Path(os.environ.get("SAGE_GOLDEN_REF", REPO / "sage_release_context" / "sage_golden" / "golden_local")) \
    / "state_dict_keys.json"


def _resolve(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("old,new", sorted(LEGACY_TARGETS.items()))
def test_every_legacy_target_resolves(old, new):
    if new.startswith("sage.nn.complex"):
        pytest.importorskip("complextorch", reason="extra [complex] not installed")
    assert upgrade_class_path(old) == new
    assert isinstance(_resolve(new), type)


def test_current_paths_pass_through_and_unknown_legacy_fails():
    assert upgrade_class_path("sage.model.encoder.SAGEEncoder") == "sage.model.encoder.SAGEEncoder"
    assert upgrade_class_path("torch.nn.Linear") == "torch.nn.Linear"
    with pytest.raises(ValueError, match="not part of the released code"):
        upgrade_class_path("ar_spectra.models.implementations.SeaNET_AE.SEANetEncoder2d")


def test_upgrade_touches_only_class_paths():
    cfg = {"encoder": {"_target_": "c_vae.swin.encoder.SwinEncoder", "mlp_type": "swiglu", "depths": [2, 6, 2]},
           "bottleneck": {"class": "ar_spectra.models.bottlenecks.VAEBottleneck", "kwargs": {}},
           "autoencoder": {"pre_transform": {"type": "power_norm"}, "note": "c_vae.swin.encoder.SwinEncoder"}}
    out = upgrade_model_config(cfg)
    assert out["encoder"] == {"_target_": "sage.model.encoder.SAGEEncoder", "mlp_type": "swiglu", "depths": [2, 6, 2]}
    assert out["bottleneck"]["class"] == "sage.nn.bottleneck.VAEBottleneck"
    assert out["autoencoder"] == cfg["autoencoder"]                  # non-class strings untouched
    assert cfg["encoder"]["_target_"] == "c_vae.swin.encoder.SwinEncoder"   # input not mutated


@pytest.mark.skipif(not (PAPER_CKPT.is_file() and STATE_KEYS.is_file()), reason="paper checkpoint / golden keys not found")
def test_paper_config_rebuilds_the_recorded_state_dict_layout():
    import torch
    from sage.model.autoencoder import SAGEAutoencoder

    ckpt = torch.load(PAPER_CKPT, map_location="cpu", weights_only=True, mmap=True)
    model_cfg = ckpt["inference_config"]["model"]
    targets = {model_cfg[k]["_target_"] for k in ("encoder", "decoder", "bottleneck")}
    assert targets <= set(LEGACY_TARGETS)                             # the checkpoint really is pre-release
    model = SAGEAutoencoder.from_config(upgrade_model_config(model_cfg))
    layout = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert layout == json.loads(STATE_KEYS.read_text())


@pytest.mark.skipif(not PAPER_CKPT.is_file(), reason="paper checkpoint not found")
def test_paper_checkpoint_resumes_into_the_decoder_ft_recipe():
    """Full resume (`ckpt_path=`): Lightning calls on_load_checkpoint and then a strict
    load_state_dict. The pre-release loss key is dropped there (LEGACY_STATE_KEYS), the
    EMA weights are stashed for the EMA callback, and every other key must match."""
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import train  # noqa: F401  (resolvers)
    from sage.compat import LEGACY_STATE_KEYS
    from sage.model.autoencoder import SAGEAutoencoder
    from sage.training.lightning_module import SAGELightningModule

    ckpt = torch.load(PAPER_CKPT, map_location="cpu", weights_only=True, mmap=True)
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=decoder_ft"])
    ds = cfg.data.train_dataset
    wrapper = SAGELightningModule(
        autoencoder=SAGEAutoencoder.from_config(upgrade_model_config(ckpt["inference_config"]["model"])),
        sample_rate=int(ds.sample_rate), audio_channels=int(cfg.data.audio_channels),
        model_channels=int(cfg.data.model_channels),
        loss_config=OmegaConf.to_container(cfg.trainer.loss_config, resolve=True),
        eval_loss_config=OmegaConf.to_container(cfg.trainer.eval_loss_config, resolve=True),
        stft_params={"sample_rate": int(ds.sample_rate), "n_fft": int(ds.n_fft), "hop_length": int(ds.hop_length),
                     "win_length": int(ds.win_length), "center": bool(ds.center), "normalized": bool(ds.normalized)},
    )
    checkpoint = {**ckpt, "state_dict": dict(ckpt["state_dict"])}
    wrapper.on_load_checkpoint(checkpoint)
    assert not set(LEGACY_STATE_KEYS) & set(checkpoint["state_dict"])
    assert len(checkpoint["_ema_state"]) == sum(k.startswith("ema_autoencoder.") for k in ckpt["state_dict"])
    assert wrapper.gen_step == ckpt["lr_schedulers"][0]["last_epoch"] > 0      # pre-release: no "gen_step"
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    sd = wrapper.state_dict()
    for k in ("engine.loss_manager.discriminator.", "engine.autoencoder.decoder.postnet."):
        keys = [n for n in ckpt["state_dict"] if n.startswith(k)]
        assert keys and all(torch.equal(sd[n], ckpt["state_dict"][n]) for n in keys), k
