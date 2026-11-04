# ===============
# The stereo-discriminator proxy recipe (configs/experiment/stereo_proxy.yaml) is the pretraining
# recipe on FMA-medium at the legacy proxy scale: global batch 16, 95 epochs, a checkpoint every
# 10 epochs. `+experiment=pretrain data=fma` alone composes batch 128 (the recipe overrides the
# data group's batch), so the proxy needs its own file. Everything else must equal pretraining.
# ===============
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train  # noqa: F401  (registers the ${mul:} resolver)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# The only keys the proxy may change with respect to `+experiment=pretrain data=fma`.
PROXY_KEYS = {
    "data.train_dataloader.batch_size": 16,
    "data.eval_dataloader.batch_size": 16,
    "trainer.trainer.epochs": 95,
    "trainer.trainer.save_every_n_epochs": 10,
    "trainer.trainer.critic_diag_every_n_epochs": 5,
    "trainer.trainer.adv_grad_every": 500,
    "trainer.wandb.name": "stereo_proxy_c0",
}


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=False)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@pytest.fixture(scope="module")
def proxy():
    return _flatten(_compose("+experiment=stereo_proxy"))


@pytest.fixture(scope="module")
def pretrain_fma():
    return _flatten(_compose("+experiment=pretrain", "data=fma"))


def test_pretrain_with_fma_alone_keeps_the_global_batch_of_128(pretrain_fma):
    """The trap the proxy file exists for: the data group's batch 16 is overridden."""
    assert pretrain_fma["data.train_dataloader.batch_size"] == 128


def test_proxy_scale(proxy):
    for key, value in PROXY_KEYS.items():
        assert proxy[key] == value, key
    assert proxy["data.fma_subset"] == "medium"
    assert proxy["trainer.scheduler.inv_gamma"] == 200000
    assert not proxy.get("data.multi_corpus.enabled", False)


def test_proxy_differs_from_pretrain_only_in_the_scale(proxy, pretrain_fma):
    changed = {k for k in proxy.keys() | pretrain_fma.keys() if proxy.get(k) != pretrain_fma.get(k)}
    assert changed == set(PROXY_KEYS), sorted(changed ^ set(PROXY_KEYS))


def test_proxy_keeps_the_paper_discriminator(proxy):
    disc = "trainer.loss_config.discriminator."
    assert proxy[disc + "type"] == "wavtokenizer"
    assert proxy[disc + "config.loss_type"] == "rpgan"
    assert proxy[disc + "config.preprocess"] is True
    assert proxy[disc + "config.fold_lrms"] is True
    assert proxy[disc + "weights.adversarial"] == 0.1
    assert proxy[disc + "weights.feature_matching"] == 0.2


# ── Arms A1, A3: only the discriminator keys of each arm differ from C0 ──────

DISC = "trainer.loss_config.discriminator.config."
ARMS = {
    "a1": {"preprocess_gain": "shared"},
    "a3": {"preprocess_gain": "shared", "mpd_stereo": "lrms", "mrd_stereo": "lrms", "dac_stereo": "lrms",
           "fused_bias_act": False, "fused_bias_grad": False},     # joint critics: no Triton bias + LeakyReLU
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_differs_from_c0_only_in_its_discriminator_keys(arm, proxy):
    cfg = _flatten(_compose(f"+experiment=stereo_proxy_{arm}"))   # lists stay whole leaves
    expected = {DISC + k: v for k, v in ARMS[arm].items()}
    expected["trainer.wandb.name"] = f"stereo_proxy_{arm}"
    changed = {k for k in cfg.keys() | proxy.keys() if cfg.get(k) != proxy.get(k)}
    assert changed == set(expected), sorted(changed ^ set(expected))
    for k, v in expected.items():
        assert cfg[k] == v, k


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_builds_its_discriminator(arm):
    from sage.nn.discriminators.wavtokenizer import WavTokenizerGANLoss
    cfg = _compose(f"+experiment=stereo_proxy_{arm}")
    disc_cfg = cfg["trainer"]["loss_config"]["discriminator"]["config"]
    WavTokenizerGANLoss(channels=2, sample_rate=44100, **disc_cfg)       # as LossManager builds it
