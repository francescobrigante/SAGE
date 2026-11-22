# ===============
# Bug B1: `+init_from=<ckpt>` must load the phase-1 autoencoder weights from checkpoints of runs with the
# old `trainer.trainer.compile`, which wrapped the autoencoder in torch.compile (keys `_orig_mod.`), into
# an eager model or one compiled in place (compile_generator), and must fail loudly instead of silently
# loading nothing when no key matches.
# ===============
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sage.training.compile import compile_generator
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
@pytest.mark.parametrize("wrapped_src", [False, True])
@pytest.mark.parametrize("use_ema,prefix", [(True, "ema_autoencoder."), (False, "engine.autoencoder.")])
def test_weights_are_loaded(tmp_path, compiled, wrapped_src, use_ema, prefix):
    torch.manual_seed(0)
    src = _TinyAE()
    dst = _TinyAE()
    if compiled:
        compile_generator(dst, backend="aot_eager")          # in place, lazy: nothing is compiled here
    wrapper = SimpleNamespace(engine=SimpleNamespace(autoencoder=dst))
    path = _ckpt(tmp_path, prefix + ("_orig_mod." if wrapped_src else ""), src)
    _load_autoencoder_weights(wrapper, str(path), use_ema=use_ema)
    assert dst.state_dict().keys() == src.state_dict().keys()
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), k


def test_fallback_to_the_live_weights_strips_the_prefix_too(tmp_path):
    torch.manual_seed(0)
    src, dst = _TinyAE(), _TinyAE()
    wrapper = SimpleNamespace(engine=SimpleNamespace(autoencoder=dst))
    _load_autoencoder_weights(wrapper, str(_ckpt(tmp_path, "engine.autoencoder._orig_mod.", src)), use_ema=True)
    assert all(torch.equal(dst.state_dict()[k], v) for k, v in src.state_dict().items())


def test_no_matching_key_is_an_error(tmp_path):
    other = nn.Sequential(nn.Linear(4, 3))                  # keys "0.weight", "0.bias"
    wrapper = SimpleNamespace(engine=SimpleNamespace(autoencoder=_TinyAE()))
    with pytest.raises(ValueError, match="match the model"):
        _load_autoencoder_weights(wrapper, str(_ckpt(tmp_path, "ema_autoencoder.", other)))
