# ===============
# The stereo-discriminator proxy recipe (configs/experiment/stereo_proxy.yaml) is the pretraining
# recipe on FMA-medium at the legacy proxy scale: global batch 16, 95 epochs, a checkpoint every
# 10 epochs. `+experiment=pretrain data=fma` alone composes batch 128 (the recipe overrides the
# data group's batch), so the proxy needs its own file. Everything else must equal pretraining,
# except the discriminator: C0 pins the paper's (folded mono critics, feature matching 0.2), while
# pretraining uses the stereo critics of arm A3+FM with the weight chosen by the sweep (10).
# ===============
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train  # noqa: F401  (registers the ${mul:} resolver)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# The only keys the proxy may change with respect to `+experiment=pretrain data=fma`: the scale ...
PROXY_KEYS = {
    "data.train_dataloader.batch_size": 16,
    "data.eval_dataloader.batch_size": 16,
    "trainer.trainer.epochs": 95,
    "trainer.trainer.save_every_n_epochs": 10,
    "trainer.trainer.critic_diag_every_n_epochs": 5,
    "trainer.wandb.name": "stereo_proxy_c0",
}
# ... and the discriminator of C0, the paper's: every branch folds L, R, M, S into the batch.
C0_DISC = {
    "trainer.loss_config.discriminator.config.preprocess_gain": "per_channel",
    "trainer.loss_config.discriminator.config.mpd_stereo": "fold",
    "trainer.loss_config.discriminator.config.mrd_stereo": "fold",
    "trainer.loss_config.discriminator.config.dac_stereo": "fold",
    "trainer.loss_config.discriminator.config.fused_bias_act": True,
    "trainer.loss_config.discriminator.config.fused_bias_grad": True,
    "trainer.loss_config.discriminator.weights.feature_matching": 0.2,
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
    assert proxy["trainer.trainer.adv_grad_every"] == 500
    assert proxy["data.fma_subset"] == "medium"
    assert proxy["trainer.scheduler.inv_gamma"] == 200000
    assert not proxy.get("data.multi_corpus.enabled", False)


def test_proxy_differs_from_pretrain_only_in_the_scale_and_the_critic(proxy, pretrain_fma):
    changed = {k for k in proxy.keys() | pretrain_fma.keys() if proxy.get(k) != pretrain_fma.get(k)}
    expected = set(PROXY_KEYS) | set(C0_DISC)
    assert changed == expected, sorted(changed ^ expected)


def test_proxy_keeps_the_paper_discriminator(proxy):
    disc = "trainer.loss_config.discriminator."
    assert proxy[disc + "type"] == "wavtokenizer"
    assert proxy[disc + "config.loss_type"] == "rpgan"
    assert proxy[disc + "config.preprocess"] is True
    assert proxy[disc + "config.fold_lrms"] is True
    assert proxy[disc + "weights.adversarial"] == 0.1
    for key, value in C0_DISC.items():
        assert proxy[key] == value, key


def test_pretrain_critic_is_the_sweep_winner(pretrain_fma):
    """Pretraining = arm A3+FM with feature_matching 10 at full scale: same discriminator block."""
    a3_fm = _flatten(_compose("+experiment=stereo_proxy_a3_fm",
                              "trainer.loss_config.discriminator.weights.feature_matching=10.0"))
    disc = "trainer.loss_config.discriminator."
    keys = {k for k in a3_fm.keys() | pretrain_fma.keys() if k.startswith(disc)}
    assert {k for k in keys if a3_fm.get(k) != pretrain_fma.get(k)} == set()
    assert pretrain_fma[disc + "weights.feature_matching"] == 10.0


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


def test_a3_fm_is_a3_with_the_feature_matching_weight_1():
    """A3+FM changes nothing of A3 but the run name and the feature-matching weight (0.2 -> 1.0)."""
    a3 = _flatten(_compose("+experiment=stereo_proxy_a3"))
    fm = _flatten(_compose("+experiment=stereo_proxy_a3_fm"))
    weight = "trainer.loss_config.discriminator.weights.feature_matching"
    changed = {k for k in fm.keys() | a3.keys() if fm.get(k) != a3.get(k)}
    assert changed == {"trainer.wandb.name", weight}, sorted(changed)
    assert (a3[weight], fm[weight]) == (0.2, 1.0)
    assert fm["trainer.wandb.name"] == "stereo_proxy_a3_fm"


# ── Decoder fine-tune on the proxy: decoder_ft (stereo critic, fm 5) with the data of C0 ──

FT_KEYS = {
    "init_from": "???",
    "trainer.wandb.name": "stereo_proxy_ft",
    "trainer.trainer.epochs": 180,
    "trainer.trainer.save_every_n_epochs": 30,
    "trainer.trainer.check_val_every_n_epoch": 1,
    "trainer.trainer.critic_diag_every_n_epochs": 5,
}


@pytest.fixture(scope="module")
def proxy_ft():
    return _compose("+experiment=stereo_proxy_ft")


def test_ft_differs_from_decoder_ft_only_in_the_proxy_keys(proxy_ft):
    ft, ref = _flatten(proxy_ft), _flatten(_compose("+experiment=decoder_ft"))
    changed = {k for k in ft.keys() | ref.keys() if ft.get(k) != ref.get(k) and not k.startswith("data.")}
    assert changed == set(FT_KEYS), sorted(changed ^ set(FT_KEYS))
    for k, v in FT_KEYS.items():
        assert ft[k] == v, k


def test_ft_init_from_is_mandatory():
    import train
    from omegaconf.errors import MissingMandatoryValue
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=stereo_proxy_ft"])
        given = compose(config_name="main", overrides=["+experiment=stereo_proxy_ft", "init_from=/runs/a3_fm.ckpt"])
    with pytest.raises(MissingMandatoryValue):
        train._init_from(cfg)
    assert train._init_from(given) == "/runs/a3_fm.ckpt"
    assert given.init_from_ema is True and given.init_from_disc is False   # EMA in, critic re-initialised


def test_ft_data_and_model_are_those_of_the_proxy(proxy_ft):
    proxy = _compose("+experiment=stereo_proxy")
    assert proxy_ft["data"] == proxy["data"]                                 # FMA-medium, global batch 16
    model, model_proxy = proxy_ft["models"]["model"], proxy["models"]["model"]
    assert model["decoder"]["use_postnet"] is True                           # the only new module, zero-init
    without_postnet = lambda m: {**m, "decoder": {**m["decoder"], "use_postnet": None}}   # noqa: E731
    assert without_postnet(model) == without_postnet(model_proxy)            # the proxy checkpoint loads


def test_ft_critic_is_that_of_a3_with_feature_matching_5(proxy_ft):
    a3_fm = _compose("+experiment=stereo_proxy_a3_fm")
    disc, disc_fm = (c["trainer"]["loss_config"]["discriminator"] for c in (proxy_ft, a3_fm))
    assert disc["config"] == disc_fm["config"]
    assert (disc["weights"]["adversarial"], disc["weights"]["feature_matching"]) == (0.1, 5.0)


def test_ft_trains_only_the_decoder(proxy_ft):
    t = proxy_ft["trainer"]
    assert t["trainer"]["freeze_encoder"] is True
    assert t["loss_config"]["bottleneck"]["weights"]["kl"] == 0.0
    assert "semantic_distill" not in t["loss_config"]
