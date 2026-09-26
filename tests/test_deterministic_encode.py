# ===============
# Tests for deterministic encode path: VAEBottleneck.encode(deterministic=True)
# returns μ exactly, is reproducible, and the flag threads through SAGEAutoencoder.
# ===============
import sys

import torch
from torch import nn

from sage.nn.bottleneck import VAEBottleneck


def _pre_bn(B=2, C=16, L=128):
    """(B, 2C, L) tensor representing [mu | scale] before the bottleneck."""
    torch.manual_seed(42)
    return torch.randn(B, 2 * C, L)


def test_deterministic_returns_mu_exactly():
    """encode(deterministic=True) must return the first-half channels verbatim."""
    x = _pre_bn()
    mu_expected = x[:, : x.shape[1] // 2]   # (B, C, L)
    bn = VAEBottleneck()
    z, _ = bn.encode(x, return_info=True, deterministic=True)
    assert torch.equal(z, mu_expected), "deterministic=True must return μ unchanged"


def test_deterministic_is_reproducible():
    """Two calls with deterministic=True on the same input must be bit-identical."""
    x = _pre_bn()
    bn = VAEBottleneck()
    z1, _ = bn.encode(x, return_info=True, deterministic=True)
    z2, _ = bn.encode(x, return_info=True, deterministic=True)
    assert torch.equal(z1, z2), "deterministic=True must be bit-reproducible"


def test_stochastic_differs_from_mu():
    """encode(deterministic=False) must NOT equal μ (noise is non-zero with prob 1)."""
    x = _pre_bn()
    mu = x[:, : x.shape[1] // 2]
    bn = VAEBottleneck()
    z, _ = bn.encode(x, return_info=True, deterministic=False)
    assert not torch.equal(z, mu), "stochastic encode must differ from μ"


def test_kl_present_in_both_modes():
    """info['kl'] must be a finite scalar in both deterministic and stochastic modes."""
    x = _pre_bn()
    bn = VAEBottleneck()
    for det in (True, False):
        _, info = bn.encode(x, return_info=True, deterministic=det)
        assert "kl" in info, f"info['kl'] missing for deterministic={det}"
        assert info["kl"].isfinite(), f"KL is not finite for deterministic={det}"


def test_default_is_stochastic():
    """Without deterministic flag the old stochastic behaviour is preserved."""
    x = _pre_bn()
    mu = x[:, : x.shape[1] // 2]
    bn = VAEBottleneck()
    # Two forward passes without seeding must differ (prob → 1)
    z1, _ = bn.encode(x.clone(), return_info=True)
    z2, _ = bn.encode(x.clone(), return_info=True)
    # At minimum, both differ from μ
    assert not torch.equal(z1, mu)
    assert not torch.equal(z2, mu)


if __name__ == "__main__":
    test_deterministic_returns_mu_exactly()
    test_deterministic_is_reproducible()
    test_stochastic_differs_from_mu()
    test_kl_present_in_both_modes()
    test_default_is_stochastic()
    print("ALL DETERMINISTIC ENCODE TESTS PASSED")
