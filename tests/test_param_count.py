# ===============
# The SAGE architecture config (configs/models/swin_real_swiglu_xsa.yaml) builds the model of the
# paper: 104,629,380 trainable parameters (Table 6), plus the 10,820 of the zero-initialised
# post-net that decoder fine-tuning adds (use_postnet=true, the released e992 checkpoint).
# ===============
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

import train  # noqa: F401  (registers the ${mul:} resolver used by the model config)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PAPER_PARAMS = 104_629_380
POSTNET_PARAMS = 10_820


def _count(overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=["models=swin_real_swiglu_xsa", *overrides])
    m = cfg.models.model
    enc, dec, bn = instantiate(m.encoder), instantiate(m.decoder), instantiate(m.bottleneck)
    return sum(p.numel() for mod in (enc, dec, bn) for p in mod.parameters())


def test_pretrained_sage_has_the_paper_parameter_count():
    assert _count([]) == PAPER_PARAMS


def test_decoder_finetune_adds_only_the_postnet():
    assert _count(["models.model.decoder.use_postnet=true"]) == PAPER_PARAMS + POSTNET_PARAMS
