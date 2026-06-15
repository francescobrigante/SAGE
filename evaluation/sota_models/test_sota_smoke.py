# =============================================================================
# evaluation/sota_models/test_sota_smoke.py
# Light, model-free smoke tests for the sota_models scaffold: the IdentityAdapter
# round-trip and the loss sanity (identity reconstruction => ~perfect metrics).
# Run: python evaluation/sota_models/test_sota_smoke.py
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SOTA_DIR  = Path(__file__).parent.resolve()
_EVAL_DIR  = _SOTA_DIR.parent
_PROJ_ROOT = _EVAL_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "src"), str(_EVAL_DIR), str(_SOTA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import build_adapter, CodecAdapter
from losses import compute_sdr_and_sisdr, stft_loss


def test_identity_adapter_roundtrip():
    a = build_adapter("identity", device="cpu", sample_rate=44100, audio_channels=2)
    assert isinstance(a, CodecAdapter)
    assert (a.sample_rate, a.audio_channels) == (44100, 2)
    wav = torch.randn(2, 44100)
    rec = a.reconstruct(wav)
    assert rec.shape == wav.shape, f"shape {rec.shape} != {wav.shape}"
    assert torch.allclose(rec, wav), "identity must return the input unchanged"
    print("  OK  identity roundtrip: shape + exact match")


def test_identity_metrics_are_near_perfect():
    wav = torch.randn(2, 44100)
    sdr, sisdr = compute_sdr_and_sisdr(wav, wav)
    st = float(stft_loss(wav, wav))
    assert sisdr > 100.0, f"SI-SDR for identity should be huge, got {sisdr}"
    assert st < 1e-4, f"STFT loss for identity should be ~0, got {st}"
    print(f"  OK  identity metrics: si_sdr={sisdr:.1f}dB (>100), stft={st:.2e} (~0)")


def test_unknown_model_raises():
    try:
        build_adapter("does-not-exist", device="cpu")
    except ValueError:
        print("  OK  unknown model raises ValueError")
        return
    raise AssertionError("expected ValueError for unknown model")


if __name__ == "__main__":
    print("== sota_models smoke tests ==")
    test_identity_adapter_roundtrip()
    test_identity_metrics_are_near_perfect()
    test_unknown_model_raises()
    print("ALL SMOKE TESTS PASSED")
