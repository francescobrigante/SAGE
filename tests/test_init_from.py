# ===============
# Bug B1: `+init_from=<ckpt>` must load the phase-1 autoencoder weights even when the model is
# wrapped by torch.compile (whose state_dict keys gain an `_orig_mod.` prefix), and must fail
# loudly instead of silently loading nothing when no key matches.
# ===============
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train import _load_autoencoder_weights


class _TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)


def _ckpt(tmp_path, prefix, ae):
    path = tmp_path / "src.ckpt"
    torch.save({"state_dict": {prefix + k: v for k, v in ae.state_dict().items()}}, path)
    return path


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("use_ema,prefix", [(True, "ema_autoencoder."), (False, "engine.autoencoder.")])
def test_weights_are_loaded(tmp_path, compiled, use_ema, prefix):
    torch.manual_seed(0)
    src = _TinyAE()
    dst = _TinyAE()
    model = torch.compile(dst) if compiled else dst          # wrapping only; nothing is compiled here
    wrapper = SimpleNamespace(engine=SimpleNamespace(autoencoder=model))
    _load_autoencoder_weights(wrapper, str(_ckpt(tmp_path, prefix, src)), use_ema=use_ema)
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), k


def test_no_matching_key_is_an_error(tmp_path):
    other = nn.Sequential(nn.Linear(4, 3))                  # keys "0.weight", "0.bias"
    wrapper = SimpleNamespace(engine=SimpleNamespace(autoencoder=_TinyAE()))
    with pytest.raises(ValueError, match="match the model"):
        _load_autoencoder_weights(wrapper, str(_ckpt(tmp_path, "ema_autoencoder.", other)))
