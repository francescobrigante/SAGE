# ===============
# The two training recipes (configs/experiment/{pretrain,decoder_ft}.yaml) compose to the
# configuration of the paper, Table 6, value by value. Also checks that the semantic gate,
# now counted in generator updates, opens at exactly the same batch as in the paper run,
# where it was compared with Lightning's global_step (bug B3).
# ===============
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train  # noqa: F401  (registers the ${mul:} resolver)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(experiment, *overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=[f"+experiment={experiment}", *overrides])
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture(scope="module")
def pre():
    return _compose("pretrain")


@pytest.fixture(scope="module")
def ft():
    return _compose("decoder_ft")


def _loss_weights(cfg):
    """Weights in the order of equation 2: (STFT, mel, SD, KL, sem, adv, fm)."""
    lc = cfg["trainer"]["loss_config"]
    sem = lc.get("semantic_distill", {}).get("weights", {}).get("distill", 0.0)
    return (lc["spectral"]["weights"]["stft_mse"], lc["mrmel"]["weights"]["mrmel"],
            lc["mrstft_sd"]["weights"]["mrstft_sd"], lc["bottleneck"]["weights"]["kl"], sem,
            lc["discriminator"]["weights"]["adversarial"], lc["discriminator"]["weights"]["feature_matching"])


# ── shared by both phases ────────────────────────────────────────────────────

@pytest.mark.parametrize("phase", ["pre", "ft"])
def test_input_representation_and_architecture(phase, request):
    cfg = request.getfixturevalue(phase)
    ds, m = cfg["data"]["train_dataset"], cfg["models"]["model"]
    assert cfg["data"]["corpora"] and cfg["data"]["multi_corpus"]["enabled"]           # multi-corpus data
    assert (ds["sample_rate"], ds["stereo"], ds["cac"]) == (44100, True, True)
    assert (ds["n_fft"], ds["hop_length"], ds["target_frames"]) == (2048, 512, 128)
    pt = m["autoencoder"]["pre_transform"]
    assert pt["type"] == "power_norm" and (pt["config"]["alpha"], pt["config"]["beta"]) == (0.65, 0.35)
    enc, dec = m["encoder"], m["decoder"]
    assert (enc["patch_size"], enc["embed_dim"]) == ([64, 1], 256)
    assert (enc["depths"], enc["num_heads"], enc["window_size"]) == ([2, 6, 2], [8, 16, 32], [4, 32])
    assert (enc["mlp_type"], enc["swiglu_hidden_ratio"]) == ("swiglu", 3.0)
    assert (enc["attention_variant"], enc["norm_placement"]) == ("xsa", "res_post")
    assert m["latent_channels"] == 16 and enc["drop_path_rate"] == 0.1
    assert enc["is_complex"] is False and dec["is_complex"] is False


@pytest.mark.parametrize("phase", ["pre", "ft"])
def test_shared_optimization(phase, request):
    cfg = request.getfixturevalue(phase)
    t = cfg["trainer"]
    assert cfg["data"]["train_dataloader"]["batch_size"] == 128                          # global batch
    opt, sch, dopt = t["optimizer"], t["scheduler"], t["disc_optimizer"]
    assert opt["_target_"] == "torch.optim.AdamW" and opt["betas"] == [0.9, 0.98]
    assert opt["weight_decay"] == 1e-4 and opt["weight_decay_exclude_1d"] is True      # matrices only
    assert sch["_target_"] == "sage.nn.schedulers.InverseLR"
    assert (sch["inv_gamma"], sch["power"], sch["warmup"], sch["interval"]) == (200000, 0.5, 0.999, "step")
    assert (dopt["lr"], dopt["betas"], dopt["weight_decay"]) == (2e-4, [0.8, 0.99], 1e-3)
    tt = t["trainer"]
    assert tt["clip_grad_norm"] == 20.0 and str(tt["precision"]) == "32"
    assert tt["ema_decay"] == 0.9998 and tt["use_ema"] is True and t["seed"] == 94
    d = t["loss_config"]["discriminator"]
    assert d["type"] == "wavtokenizer" and d["config"]["loss_type"] == "rpgan"
    assert d["config"]["fold_lrms"] is True                                              # L, R, M, S
    assert (d["config"]["periods"], d["config"]["fft_sizes"]) == ([2, 3, 5, 7, 11], [2048, 1024, 512])
    sd = t["loss_config"]["mrstft_sd"]["config"]
    assert sd["fft_sizes"] == [2048, 1024, 512, 256, 128, 64] and sd["perceptual_weighting"] is True
    assert sd["hop_sizes"] == [f // 4 for f in sd["fft_sizes"]]


# ── phase 1: pretraining ─────────────────────────────────────────────────────

def test_pretrain(pre):
    t = pre["trainer"]
    assert t["trainer"]["epochs"] == 500 and t["optimizer"]["lr"] == 1e-3
    assert _loss_weights(pre) == (1.0, 0.5, 1.0, 1e-4, 1.0, 0.1, 0.2)
    aux = t["aux_optimizer"]
    assert (aux["lr"], aux["betas"], aux["weight_decay"]) == (1e-5, [0.9, 0.98], 1e-4)
    assert t["aux_scheduler"]["_target_"] == "torch.optim.lr_scheduler.ConstantLR"
    sem = t["loss_config"]["semantic_distill"]
    assert sem["teacher_type"] == "clap" and sem["config"] == {"proj_dim": 512, "latent_dim": 64}
    assert sem["detach_warmup_steps"] == 8334                                            # s0 ≈ 8.3k gen updates
    assert t["trainer"]["freeze_encoder"] is False
    assert pre["models"]["model"]["decoder"]["use_postnet"] is False


# ── phase 2: decoder fine-tuning (differences only) ──────────────────────────

def test_decoder_finetune(ft):
    t = ft["trainer"]
    assert t["trainer"]["epochs"] == 992 and t["optimizer"]["lr"] == 1e-4
    assert _loss_weights(ft) == (1.0, 0.5, 1.0, 0.0, 0.0, 0.1, 0.2)                    # λKL = λsem = 0
    assert "semantic_distill" not in t["loss_config"]                                    # no CLAP teacher
    assert t["trainer"]["freeze_encoder"] is True and t["trainer"]["warmup_steps"] == 0
    assert ft["models"]["model"]["decoder"]["use_postnet"] is True
    assert ft["init_from_ema"] is True and ft["init_from_disc"] is False              # EMA in, disc from scratch
    assert "aux_optimizer" not in t


def test_cli_can_still_override_the_dataset():
    cfg = _compose("pretrain", "data=fma")
    assert not cfg["data"].get("multi_corpus", {}).get("enabled", False)


# ── B3: the semantic gate in generator updates ──────────────────────────────

def test_gate_in_generator_updates_matches_the_paper_run():
    """Paper run: gate open when Lightning's global_step >= 25,000, where each generator batch did
    2 optimizer steps (generator + projection head) and each discriminator batch 1. Recipe: gate
    open when gen_step >= 8,334. Both must open on the same batch."""
    global_step = gen_step = 0
    disc_phase = True
    old_open = new_open = None
    for batch in range(60_000):
        disc_phase = not disc_phase                     # the wrapper's per-batch toggle, gen first
        if disc_phase:
            global_step += 1                            # opt_disc
            continue
        if old_open is None and global_step >= 25_000:
            old_open = batch
        if new_open is None and gen_step >= 8_334:
            new_open = batch
        global_step += 2                                # opt_gen + opt_aux
        gen_step += 1
    assert old_open is not None and old_open == new_open


# ── B8: the derived run name stays short ─────────────────────────────────────

@pytest.mark.parametrize("experiment", ["pretrain", "decoder_ft"])
def test_derived_run_name_is_short_and_uses_the_model_choice(experiment):
    """Without trainer.wandb.name the run name is derived from the config. It used to spell
    every loss weight and exceed the 255-character file-name limit (bug B8)."""
    from hydra.core.hydra_config import HydraConfig
    from sage.utils.run_config import resolve_run_name

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=[f"+experiment={experiment}", "trainer.wandb.name=null"],
                      return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    try:
        name, again = resolve_run_name(cfg), resolve_run_name(cfg)
    finally:
        HydraConfig.instance().cfg = None
    assert name.startswith("swin_real_swiglu_xsa-") and len(name) < 100
    assert name == again                                          # deterministic: hash of the loss weights
