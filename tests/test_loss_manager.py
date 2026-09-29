# ===============
# The loss manager builds the paper loss (eq. 2) and nothing else from its own config
# blocks; the losses explored during development (sage.nn.losses.experimental) enter
# only through `loss_config.extra`, after the paper terms. Also: the paper checkpoint's
# loss-manager state still loads (strictly) once the pre-release key is filtered.
# ===============
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

import train  # noqa: F401  (registers the ${mul:} resolver)
import sage.training.loss_manager as lm_mod
from sage.compat import LEGACY_STATE_KEYS, drop_legacy_state_keys
from sage.nn.bottleneck import VAEBottleneck
from sage.nn.losses.perceptual import MelSpectrogramLoss
from sage.nn.losses.signal import SumAndDifferenceSTFTLoss
from sage.nn.losses.spectral import ComplexMSE
from sage.training.loss_manager import LossManager
from sage.utils.run_config import build_run_name

REPO = Path(__file__).resolve().parents[1]
PAPER_KEYS = json.loads((REPO / "tests" / "data" / "paper_ckpt_loss_manager_keys.json").read_text())["keys"]
PREFIX = "engine.loss_manager."
KL = {"bottleneck": {"weights": {"kl": 1e-4}}}


class _StubAE(nn.Module):
    """The LossManager only reads `bottleneck` (for the KL term)."""
    def __init__(self):
        super().__init__()
        self.bottleneck = VAEBottleneck()


class _StubCLAP(nn.Module):
    """Stands in for the LAION-CLAP teacher (a 600 MB download)."""
    def __init__(self, model_dir, src_sr=44100):
        super().__init__()

    def forward(self, wav):
        return wav.mean(dim=1)[:, :512 * 16].reshape(wav.shape[0], 512, 16).mean(-1)


@pytest.fixture(autouse=True)
def _no_clap_download(monkeypatch):
    monkeypatch.setattr(lm_mod, "CLAPTeacher", _StubCLAP)


def _configs(*overrides):
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="main", overrides=list(overrides))
    t = cfg.trainer
    return (OmegaConf.to_container(t.loss_config, resolve=True),
            OmegaConf.to_container(t.eval_loss_config, resolve=True))


def _manager(loss_config, eval_loss_config=None):
    return LossManager(_StubAE(), sample_rate=44100, loss_config=loss_config,
                       eval_loss_config=eval_loss_config, audio_channels=2)


def _loss_info(T=8192, frames=65, bins=257):
    g = torch.Generator().manual_seed(0)
    spec = lambda: torch.randn(2, 4, bins, frames, generator=g)            # CAC, stereo
    return {
        "reals": 0.3 * torch.randn(2, 2, T, generator=g),
        "decoded": (0.3 * torch.randn(2, 2, T, generator=g)).requires_grad_(),
        "encoder_input": spec(),
        "sp_decoded": spec().requires_grad_(),
        "sp_decoded_linear": spec().requires_grad_(),
        "latents": torch.randn(2, 16, 32, generator=g).requires_grad_(),
        "feature_shape": (4, 8),
        "gen_step": 0,
        "loss_adv": torch.tensor(0.7), "feature_matching_distance": torch.tensor(1.3), "kl": torch.tensor(5.0),
    }


# ── the recipes build exactly eq. 2 ──────────────────────────────────────────

PAPER_LOSS_CLASSES = {ComplexMSE, MelSpectrogramLoss, SumAndDifferenceSTFTLoss}


@pytest.mark.parametrize("experiment,expected", [
    ("pretrain", [("loss_adv", 0.1), ("feature_matching_loss", 0.2), ("pwc_mse_loss", 1.0), ("mrmel_loss", 0.5),
                  ("mrstft_sd_loss", 1.0), ("distill_loss", 1.0), ("kl_loss", 1e-4)]),
    ("decoder_ft", [("loss_adv", 0.1), ("feature_matching_loss", 0.2), ("pwc_mse_loss", 1.0), ("mrmel_loss", 0.5),
                    ("mrstft_sd_loss", 1.0), ("kl_loss", 0.0)]),
])
def test_recipe_builds_exactly_the_paper_terms(experiment, expected):
    lm = _manager(*_configs(f"+experiment={experiment}"))
    got = [(m.name, float(m.weight)) for m in lm.losses_gen.losses]
    assert [n for n, _ in got] == [n for n, _ in expected]                   # order = checkpoint layout
    assert [w for _, w in got] == pytest.approx([w for _, w in expected])
    inner = {type(m.loss_module) for m in lm.losses_gen.losses if hasattr(m, "loss_module")}
    assert inner == PAPER_LOSS_CLASSES
    assert not [m for m in lm.modules() if type(m).__module__.startswith("sage.nn.losses.experimental")]


def test_trainer_defaults_build_the_reconstruction_terms():
    lm = _manager(*_configs())
    assert [m.name for m in lm.losses_gen.losses] == ["pwc_mse_loss", "mrmel_loss", "mrstft_sd_loss", "kl_loss"]
    assert [float(m.weight) for m in lm.losses_gen.losses] == pytest.approx([1.0, 0.5, 1.0, 1e-4])


def test_zero_weight_terms_are_not_built():
    lm = _manager({"spectral": {"weights": {"stft_mse": 0.0}}, "mrstft_sd": {"weights": {"mrstft_sd": 0.0}}, **KL})
    assert [m.name for m in lm.losses_gen.losses] == ["kl_loss"]
    assert lm.stft_mse is None and lm.mrstft_sd is None


@pytest.mark.parametrize("loss_config", [
    {"mrstft_same": {"weights": {"mrstft_same": 1.0}}},                     # pre-release top-level block
    {"hubert": {"weights": {"hubert": 0.0}}},
    {"spectral": {"mrstft": {"config": {}}, "weights": {"stft_mse": 1.0}}},  # pre-release spectral variant
    {"spectral": {"weights": {"stft_mse": 1.0, "stft_consistency": 0.0}}},
    {"mrstft_sd": {"weights": {"mrstft_sd": 1.0}, "decay": 0.5}},           # pre-release per-term decay
    {"bottleneck": {"weights": {"KL": 0.1}}},                                # typo
    {"semantic_distill": {"weights": {"distill": 1.0}, "detach_warmup_step": 10}},
    {"mrmel": {"weights": {"mrmel": 0.5}, "config": {"n_mel": [128]}}},
    {"discriminator": {"type": "wavtokenizer", "config": {}, "weights": {"adversarial": 0.1, "fm": 0.2}}},
])
def test_unknown_loss_keys_are_rejected(loss_config):
    with pytest.raises(ValueError, match="loss_config.extra"):
        _manager(loss_config)


def test_kl_weight_is_required_with_a_vae_bottleneck():
    with pytest.raises(ValueError, match="bottleneck.weights.kl"):
        _manager({"spectral": {"weights": {"stft_mse": 1.0}}})


# ── experimental losses through `extra` ──────────────────────────────────────

EXP = "sage.nn.losses.experimental."
EXTRA = {   # name: (input_key, target_key, loss config)
    "perceptual_mse": ("sp_decoded", "encoder_input",
                       {"_target_": EXP + "PerceptualComplexMSE", "n_fft": 512, "power_norm_alpha": 0.65}),
    "cosine_phase": ("sp_decoded", "encoder_input", {"_target_": EXP + "PhaseCosineDistance"}),
    "complex_sc": ("sp_decoded", "encoder_input", {"_target_": EXP + "ComplexSpectralConvergence"}),
    "spectral_contrast": ("sp_decoded", "encoder_input", {"_target_": EXP + "SpectralContrastLoss"}),
    "mse_side": ("sp_decoded", "encoder_input", {"_target_": EXP + "SideComplexMSE"}),
    "stft_consistency": ("sp_decoded_linear", None,
                         {"_target_": EXP + "STFTConsistencyLoss", "n_fft": 512, "hop_length": 128, "win_length": 512}),
    "mrstft": ("decoded", "reals", {"_target_": EXP + "MultiResolutionSpectrogramLoss",
                                    "fft_sizes": [256, 512], "log_mag": True}),
    "mrstft_sc": ("decoded", "reals", {"_target_": EXP + "MultiResSpectralConvergence", "fft_sizes": [256, 512],
                                       "hop_sizes": [64, 128], "win_lengths": [256, 512]}),
    "mrstft_stable_audio": ("decoded", "reals", {"_target_": "sage.nn.losses.signal.MultiResolutionSTFTLoss",
                                                 "fft_sizes": [256, 512], "hop_sizes": [64, 128],
                                                 "win_lengths": [256, 512]}),
    "if_gd": ("decoded", "reals", {"_target_": EXP + "InstantaneousFrequencyGroupDelayLoss",
                                   "n_fft": 512, "hop_length": 128, "win_length": 512}),
    "ncd": ("decoded", "reals", {"_target_": EXP + "NormalizedComplexDistanceLoss",
                                 "n_fft": 512, "hop_length": 128, "win_length": 512}),
    "mrstft_same": ("decoded", "reals", {"_target_": EXP + "MRSTFTSame", "fft_sizes": [128, 256, 512]}),
    "stereo_coh": ("decoded", "reals", {"_target_": EXP + "StereoCoherenceLoss", "fft_sizes": [512, 256],
                                        "hop_sizes": [128, 64], "win_lengths": [512, 256]}),
    "time_l1": ("decoded", "reals", {"_target_": "torch.nn.L1Loss"}),      # any nn.Module loss works
}


def _entry(name, weight=0.5):
    input_key, target_key, loss = EXTRA[name]
    entry = {"name": name, "weight": weight, "input_key": input_key, "loss": dict(loss)}
    if target_key is not None:
        entry["target_key"] = target_key
    return entry


def test_every_experimental_loss_has_an_extra_case():
    import inspect
    import sage.nn.losses.experimental as experimental
    classes = {n for n, o in vars(experimental).items() if inspect.isclass(o) and issubclass(o, nn.Module)}
    covered = {cfg["_target_"].rsplit(".", 1)[1] for _, _, cfg in EXTRA.values()}
    assert classes - {"HubertLoss"} <= covered                              # HubertLoss: see the weights test


@pytest.mark.parametrize("name", sorted(EXTRA))
def test_extra_loss_is_built_after_the_paper_terms_and_trains(name):
    lm = _manager({"spectral": {"weights": {"stft_mse": 1.0}}, "extra": [_entry(name)], **KL})
    assert [m.name for m in lm.losses_gen.losses] == ["pwc_mse_loss", "kl_loss", name]
    extra = lm.losses_gen.losses[-1]
    assert float(extra.weight) == 0.5
    info = _loss_info()
    total, breakdown = lm.losses_gen(info)
    assert torch.isfinite(total) and torch.isfinite(breakdown[name])
    breakdown[name].backward()
    grad = info[EXTRA[name][0]].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.weights
def test_hubert_extra_loss():
    entry = {"name": "hubert", "weight": 1.0, "input_key": "decoded", "target_key": "reals",
             "loss": {"_target_": EXP + "HubertLoss", "model_name": "HUBERT_LARGE"}}
    lm = _manager({"extra": [entry], **KL})
    info = _loss_info(T=16000)
    _, breakdown = lm.losses_gen(info)
    assert torch.isfinite(breakdown["hubert"])


def test_extra_does_not_shift_the_paper_terms():
    loss_cfg, eval_cfg = _configs("+experiment=decoder_ft")
    base = [m.name for m in _manager(loss_cfg, eval_cfg).losses_gen.losses]
    loss_cfg["extra"] = [_entry("stereo_coh")]
    with_extra = [m.name for m in _manager(loss_cfg, eval_cfg).losses_gen.losses]
    assert with_extra == base + ["stereo_coh"]


def test_extra_from_the_command_line():
    loss_cfg, _ = _configs(
        "+experiment=decoder_ft",
        "trainer.loss_config.extra=[{name:time_l1,weight:0.1,input_key:decoded,target_key:reals,"
        "loss:{_target_:torch.nn.L1Loss}}]")
    lm = _manager(loss_cfg)
    assert lm.losses_gen.losses[-1].name == "time_l1"
    assert isinstance(lm.losses_gen.losses[-1].loss_module, nn.L1Loss)


def test_extra_with_zero_weight_is_not_built():
    lm = _manager({"extra": [_entry("time_l1", weight=0.0)], **KL})
    assert [m.name for m in lm.losses_gen.losses] == ["kl_loss"]


@pytest.mark.parametrize("entry,match", [
    ({"name": "x", "weight": 1.0, "loss": {"_target_": "torch.nn.L1Loss"}}, "missing keys"),
    ({"name": "x", "weight": 1.0, "input_key": "decoded", "target": "reals",
      "loss": {"_target_": "torch.nn.L1Loss"}}, "unknown keys"),
    ({"name": "kl_loss", "weight": 1.0, "input_key": "decoded", "target_key": "reals",
      "loss": {"_target_": "torch.nn.L1Loss"}}, "duplicate"),
])
def test_bad_extra_entries_are_rejected(entry, match):
    with pytest.raises(ValueError, match=match):
        _manager({"extra": [entry], **KL})


def test_run_name_hash_includes_extra_weights():
    base = {"spectral": {"weights": {"stft_mse": 1.0}}}
    a = build_run_name("m", {**base, "extra": [_entry("time_l1", 0.1)]})
    b = build_run_name("m", {**base, "extra": [_entry("time_l1", 0.2)]})
    assert a != b != build_run_name("m", base)


# ── resuming the paper checkpoint ────────────────────────────────────────────

def _paper_loss_state():
    return {k: torch.zeros(shape) for k, shape in PAPER_KEYS.items()}


def _strip(state):
    return {k[len(PREFIX):]: v for k, v in state.items()}


def test_legacy_key_list_matches_the_paper_checkpoint():
    current = {PREFIX + k for k in _manager(*_configs("+experiment=decoder_ft")).state_dict()}
    assert set(PAPER_KEYS) - current == set(LEGACY_STATE_KEYS)             # nothing else is orphaned
    assert current - set(PAPER_KEYS) == set()                              # and nothing is missing


def test_paper_loss_state_needs_the_filter():
    lm = _manager(*_configs("+experiment=decoder_ft"))
    with pytest.raises(RuntimeError, match="consistency_loss._window"):
        lm.load_state_dict(_strip(_paper_loss_state()), strict=True)


def test_paper_loss_state_loads_strictly_after_the_filter():
    lm = _manager(*_configs("+experiment=decoder_ft"))
    state = _paper_loss_state()
    assert drop_legacy_state_keys(state) == list(LEGACY_STATE_KEYS)
    lm.load_state_dict(_strip(state), strict=True)
    # Loss weights are stored by position; after a resume the configured weight wins
    # at the first forward (master_weight), whatever the checkpoint held.
    lm.losses_gen(_loss_info())
    assert [float(m.weight) for m in lm.losses_gen.losses] == pytest.approx([0.1, 0.2, 1.0, 0.5, 1.0, 0.0])


def test_filter_leaves_current_checkpoints_alone():
    state = {PREFIX + k: v for k, v in _manager(*_configs("+experiment=decoder_ft")).state_dict().items()}
    before = set(state)
    assert drop_legacy_state_keys(state) == [] and set(state) == before


# ── frozen reference: the paper loss must not drift in later refactors ───────
# tests/data/loss_manager_reference.json was written by _reference_record() on the code of
# Phase 3, whose loss manager is bit-identical to the pre-release one (same state, losses
# and gradients for both recipes). Tolerances cover BLAS/FFT differences across machines.

REFERENCE = REPO / "tests" / "data" / "loss_manager_reference.json"


def _reference_record(experiment):
    torch.manual_seed(94)                                    # trainer seed: discriminator + CLAP head init
    lm = _manager(*_configs(f"+experiment={experiment}"))
    record = {}
    for gen_step in (0, 10_000):                             # semantic gate closed / open
        info = _loss_info()
        info["gen_step"] = gen_step
        total, breakdown = lm.losses_gen(info)
        total.backward()
        record[str(gen_step)] = {
            "total": total.item(),
            **{k: v.item() for k, v in breakdown.items()},
            **{f"grad_norm_{k}": info[k].grad.norm().item() for k in ("decoded", "sp_decoded", "latents")
               if info[k].grad is not None},
        }
    info = _loss_info()
    torch.manual_seed(0)
    loss_dis, loss_adv, feat_match = lm.discriminator.loss(reals=info["reals"], fakes=info["decoded"].detach())
    record["disc"] = {"loss_dis": loss_dis.item(), "loss_adv": loss_adv.item(), "feature_matching": feat_match.item()}
    record["disc_param_sum"] = sum(p.double().sum().item() for p in lm.discriminator.parameters())
    return record


@pytest.mark.parametrize("experiment", ["pretrain", "decoder_ft"])
def test_paper_loss_matches_the_frozen_reference(experiment):
    ref = json.loads(REFERENCE.read_text())[experiment]
    got = _reference_record(experiment)
    assert got.keys() == ref.keys()
    for section, values in ref.items():
        if isinstance(values, dict):
            assert got[section].keys() == values.keys(), section
            for k, v in values.items():
                # gradient norms move more than the losses across CPUs (vs the reference written
                # on arm64: 2.6e-4 rel on the cluster's x86, 1.1e-3 on a GitHub runner; losses
                # within 1e-7)
                rel = 1e-2 if k.startswith("grad_norm_") else 1e-5
                assert got[section][k] == pytest.approx(v, rel=rel, abs=1e-8), f"{section}.{k}"
        else:
            assert got[section] == pytest.approx(values, rel=1e-6), section
