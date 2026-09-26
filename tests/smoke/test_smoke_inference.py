# ===============
# Smoke tests of inference: the paper checkpoint rebuilds from its inference_config, loads the
# EMA weights, and reproduces the local golden references bit for bit on the short inputs
# (128-frame training crop and 10 s MoisesDB mix). The full 18-array check is make_golden.py.
# ===============
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import GOLDEN_DIR, PAPER_CKPT

pytestmark = pytest.mark.smoke
GOLDEN_LOCAL = GOLDEN_DIR / "golden_local"
needs_ckpt = pytest.mark.skipif(not PAPER_CKPT.is_file(), reason=f"paper checkpoint not found: {PAPER_CKPT}")
needs_golden = pytest.mark.skipif(not GOLDEN_LOCAL.is_dir(), reason=f"golden references not found: {GOLDEN_LOCAL}")


def _pad_for_swin(wav: torch.Tensor, hop: int, n_down: int) -> tuple[torch.Tensor, int]:
    """Same padding as evaluation/evaluate_swin_10s.py::_pad_for_swin (frozen in make_golden.py)."""
    orig = wav.shape[-1]
    mult = 2 ** n_down
    frames = orig // hop + 1
    target = max(orig, (frames + (mult - frames % mult) % mult - 1) * hop)
    return (F.pad(wav, (0, target - orig)) if target > orig else wav), orig


@pytest.fixture(scope="module")
def single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)                      # golden_local was generated with 1 thread
    yield
    torch.set_num_threads(n)


def _load(varlen: str):
    from ar_spectra.models.inference import EuleroEncodeDecode
    return EuleroEncodeDecode(PAPER_CKPT, device="cpu", varlen=varlen)


@needs_ckpt
def test_checkpoint_loads_every_weight_by_name():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # missing/unexpected keys are warnings
        codec = _load("tri2")
    if GOLDEN_LOCAL.is_dir():
        ref = json.loads((GOLDEN_LOCAL / "state_dict_keys.json").read_text())
        cur = {k: list(v.shape) for k, v in codec.autoencoder.state_dict().items()}
        assert cur == ref


@needs_ckpt
def test_encode_decode_shapes():
    codec = _load("tri2")
    x = 0.1 * torch.randn(1, 2, 65024)                          # 128 STFT frames, the training grid
    z = codec.encode(x, deterministic=True)
    y = codec.decode(z, target_length=x.shape[-1])
    assert tuple(z.shape) == (1, 16, 128) and tuple(y.shape) == (1, 2, 65024)
    assert torch.isfinite(y).all()


@needs_ckpt
@needs_golden
@pytest.mark.parametrize("name", ["crop_train_len_128frames", "moises10s_mix"])
@pytest.mark.parametrize("mode", ["tri2", "off"])
def test_bit_exact_against_golden_local(name, mode, single_thread):
    codec = _load(mode)
    hop = codec.autoencoder._stft_config.hop_length
    n_down = len(codec.autoencoder.encoder.depths) - 1
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / f"{name}.npy"))
    wav_p, orig = _pad_for_swin(wav, hop, n_down)
    batch = wav_p.unsqueeze(0)

    mu = codec.encode(batch, deterministic=True)
    rec = codec.decode(mu, target_length=wav_p.shape[-1])[0, ..., :orig]
    cases = {"latent_mu": mu, "recon_mu": rec}
    if mode == "tri2":                                          # sampled path, as the paper evaluator
        torch.manual_seed(0)
        z = codec.encode(batch, deterministic=False)
        cases |= {"latent_z": z, "recon_z": codec.decode(z, target_length=wav_p.shape[-1])[0, ..., :orig]}
    for case, t in cases.items():
        ref = np.load(GOLDEN_LOCAL / f"{name}__{mode}__{case}.npy")
        assert np.array_equal(t.numpy(), ref), f"{name}/{mode}/{case}: max|Δ|={np.abs(t.numpy() - ref).max():.3e}"
