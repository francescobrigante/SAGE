# ==============================================
# Pytest tests for the VAE bottlenecks
# (sage.nn.bottleneck, sage.nn.complex.bottleneck).
# Covers shape, correctness, KL properties, 
# gradients, and both encoding modes.
# ==============================================

import math
import sys

import pytest
import torch


from sage.nn.complex.bottleneck import reparametrize, get_kl, get_cholesky_kl, get_proper_kl, ComplexVAEBottleneck
from sage.nn.bottleneck import VAEBottleneck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(B: int = 2, C: int = 6, H: int = 8, W: int = 16) -> torch.Tensor:
    """Return a random complex tensor with C channels (must be divisible by 3)."""
    real = torch.randn(B, C, H, W)
    imag = torch.randn(B, C, H, W)
    return torch.complex(real, imag)


def _make_valid_sigma_c(shape=(2, 4, 8, 16)):
    """Return (sigma, c) such that sigma > |c| elementwise."""
    c = torch.complex(torch.randn(*shape) * 0.3, torch.randn(*shape) * 0.3)
    sigma = c.abs() + 0.5 + torch.rand(*shape) * 0.5   # strictly > |c|
    return sigma, c


def _make_mu(shape=(2, 4, 8, 16)):
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


# ---------------------------------------------------------------------------
# Test 1 — encode() output shape (Mode A)
# ---------------------------------------------------------------------------

def test_encode_shape_mode_a():
    """encode(x, return_info=True) with x=(2,6,8,16) → z=(2,2,8,16) complex, kl scalar."""
    bottleneck = ComplexVAEBottleneck(apply_cholesky_constraints=False)
    x = _make_input(B=2, C=6, H=8, W=16)
    z, info = bottleneck.encode(x, return_info=True)

    assert torch.is_complex(z), "z must be complex"
    assert z.shape == (2, 2, 8, 16), f"Expected z shape (2,2,8,16), got {z.shape}"
    assert "kl" in info, "info dict must contain 'kl'"
    assert info["kl"].ndim == 0, "kl must be a scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# Test 2 — isinstance check
# ---------------------------------------------------------------------------

def test_isinstance_vae_bottleneck():
    """ComplexVAEBottleneck must be a VAEBottleneck instance."""
    bottleneck = ComplexVAEBottleneck()
    assert isinstance(bottleneck, VAEBottleneck)


# ---------------------------------------------------------------------------
# Test 3 — KL non-negativity (Mode A, random valid inputs)
# ---------------------------------------------------------------------------

def test_get_kl_non_negative():
    """get_kl should be >= 0 for valid (sigma > |c|) inputs."""
    sigma, c = _make_valid_sigma_c()
    mu = _make_mu()
    kl, _ = get_kl(mu, sigma, c)
    assert kl.item() >= 0.0, f"KL should be non-negative, got {kl.item()}"


# ---------------------------------------------------------------------------
# Test 4 — KL at prior for Mode A: mu=0, sigma=1, c=0 → KL ≈ 0
# ---------------------------------------------------------------------------

def test_get_kl_at_prior_mode_a():
    """KL(CN(0,I,0) || CN(0,I,0)) should equal 0."""
    shape = (2, 4, 8, 16)
    mu = torch.zeros(*shape, dtype=torch.cfloat)
    sigma = torch.ones(*shape)
    c = torch.zeros(*shape, dtype=torch.cfloat)
    kl, _ = get_kl(mu, sigma, c)
    assert abs(kl.item()) < 1e-5, f"KL at prior should be ~0, got {kl.item()}"


# ---------------------------------------------------------------------------
# Test 5 — KL at prior for Mode B: mu=0, l11=l22=1/√2, l21=0 → KL ≈ 0
# ---------------------------------------------------------------------------

def test_get_cholesky_kl_at_prior():
    """KL in Cholesky space at the prior (l11=l22=1/√2, l21=0, mu=0) should be 0."""
    shape = (2, 4, 8, 16)
    mu = torch.zeros(*shape, dtype=torch.cfloat)
    val = 1.0 / math.sqrt(2)
    l11 = torch.full(shape, val)
    l21 = torch.zeros(*shape)
    l22 = torch.full(shape, val)
    kl, _ = get_cholesky_kl(mu, l11, l21, l22)
    assert abs(kl.item()) < 1e-5, f"Cholesky KL at prior should be ~0, got {kl.item()}"


# ---------------------------------------------------------------------------
# Test 6 — Cholesky auto-constraint: sigma^2 - |c|^2 = 4*l11^2*l22^2 > 0
# ---------------------------------------------------------------------------

def test_cholesky_auto_constraint():
    """
    For Mode B recovery formulas, sigma^2 - |c|^2 must equal 4*l11^2*l22^2 elementwise,
    guaranteeing the positivity constraint is auto-satisfied.
    Uses float64 so the algebraic identity can be verified at machine-precision (atol=1e-8).
    """
    shape = (2, 4, 8, 16)
    # float64: squaring accumulates ~1e-14 relative error vs ~4e-3 in float32
    l11 = (torch.rand(*shape).abs() + 0.1).double()
    l21 = torch.randn(*shape).double()
    l22 = (torch.rand(*shape).abs() + 0.1).double()

    sigma = l11**2 + l21**2 + l22**2
    c = torch.complex(
        l11**2 - l21**2 - l22**2,
        2.0 * l11 * l21,
    )

    lhs = sigma**2 - c.abs()**2
    rhs = 4.0 * l11**2 * l22**2

    assert torch.allclose(lhs, rhs, atol=1e-8), (
        "sigma^2 - |c|^2 must equal 4*l11^2*l22^2 everywhere"
    )
    assert (lhs > 0).all(), "sigma^2 - |c|^2 must be strictly positive for all elements"


# ---------------------------------------------------------------------------
# Test 7 — Gradient flow through reparametrize()
# ---------------------------------------------------------------------------

def test_reparametrize_gradient_flow():
    """z = reparametrize(mu, sigma, c) must propagate gradients back to mu, sigma, c."""
    shape = (2, 4, 8, 16)
    sigma, c = _make_valid_sigma_c(shape)
    mu = _make_mu(shape)

    mu = mu.detach().requires_grad_(True)
    sigma = sigma.detach().requires_grad_(True)
    c_r = c.real.detach().requires_grad_(True)
    c_i = c.imag.detach().requires_grad_(True)
    c_with_grad = torch.complex(c_r, c_i)

    z = reparametrize(mu, sigma, c_with_grad)
    # Scalar loss: sum of real and imag parts
    loss = z.real.sum() + z.imag.sum()
    loss.backward()

    assert mu.grad is not None, "Gradient did not flow to mu"
    assert sigma.grad is not None, "Gradient did not flow to sigma"
    assert c_r.grad is not None, "Gradient did not flow to c.real"
    assert c_i.grad is not None, "Gradient did not flow to c.imag"


# ---------------------------------------------------------------------------
# Test 8 — encode() output shape for Mode B
# ---------------------------------------------------------------------------

def test_encode_shape_mode_b():
    """encode() in Mode B (apply_cholesky_constraints=True) → z=(2,2,8,16) complex."""
    bottleneck = ComplexVAEBottleneck(apply_cholesky_constraints=True)
    x = _make_input(B=2, C=6, H=8, W=16)
    z = bottleneck.encode(x, return_info=False)

    assert torch.is_complex(z), "z must be complex"
    assert z.shape == (2, 2, 8, 16), f"Expected z shape (2,2,8,16), got {z.shape}"


# ---------------------------------------------------------------------------
# Test 9 — KL barrier: |c| ≈ sigma → large positive value, not NaN
# ---------------------------------------------------------------------------

def test_get_kl_barrier_near_constraint():
    """
    When |c| approaches sigma (constraint boundary), get_kl should return a large finite
    positive value, not NaN. The -1/2 * log(sigma^2 - |c|^2) term acts as a barrier.
    """
    shape = (1, 1, 1, 1)
    mu = torch.zeros(*shape, dtype=torch.cfloat)
    sigma = torch.ones(*shape)
    # |c| = 0.9999 * sigma: very close to boundary but still feasible
    c_r = torch.full(shape, 0.9999)
    c_i = torch.zeros(*shape)
    c = torch.complex(c_r, c_i)

    kl, _ = get_kl(mu, sigma, c)
    assert not torch.isnan(kl), "KL must not be NaN near constraint boundary"
    assert kl.item() > 0.0, f"KL near constraint boundary should be large positive, got {kl.item()}"
    # Barrier should make KL substantially large (log(sigma^2 - |c|^2) → -inf as |c| → sigma)
    assert kl.item() > 1.0, f"KL near constraint boundary should be >> 1, got {kl.item()}"


# ---------------------------------------------------------------------------
# Test 10 — decode() passthrough
# ---------------------------------------------------------------------------

def test_decode_passthrough():
    """decode(z) must return z unchanged (identity passthrough, inherited from VAEBottleneck)."""
    bottleneck = ComplexVAEBottleneck()
    shape = (2, 2, 8, 16)
    z = torch.complex(torch.randn(*shape), torch.randn(*shape))
    out = bottleneck.decode(z)
    # Must be the exact same tensor (identity, not a copy)
    assert out is z, "decode() must return the input tensor unchanged"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_encode_raises_on_real_input():
    """encode() must raise AssertionError when given a real tensor."""
    bottleneck = ComplexVAEBottleneck()
    x_real = torch.randn(2, 6, 8, 16)
    with pytest.raises(AssertionError, match="complex"):
        bottleneck.encode(x_real)


def test_encode_raises_on_wrong_channel_count():
    """encode() must raise AssertionError when channels are not divisible by 3."""
    bottleneck = ComplexVAEBottleneck()
    x = _make_input(B=2, C=4, H=8, W=16)  # 4 channels — not divisible by 3
    with pytest.raises(AssertionError):
        bottleneck.encode(x)


def test_reparametrize_output_is_complex():
    """reparametrize() must return a complex tensor."""
    sigma, c = _make_valid_sigma_c()
    mu = _make_mu()
    z = reparametrize(mu, sigma, c)
    assert torch.is_complex(z), "reparametrize() must return a complex tensor"
    assert z.shape == mu.shape, "reparametrize() output must have same shape as mu"


def test_encode_mode_a_return_info_false():
    """encode() with return_info=False must return a tensor, not a tuple."""
    bottleneck = ComplexVAEBottleneck(apply_cholesky_constraints=False)
    x = _make_input()
    result = bottleneck.encode(x, return_info=False)
    assert isinstance(result, torch.Tensor), "return_info=False should return a Tensor"


def test_encode_mode_b_return_info_true():
    """encode() Mode B with return_info=True must return (z, info) with kl key."""
    bottleneck = ComplexVAEBottleneck(apply_cholesky_constraints=True)
    x = _make_input()
    z, info = bottleneck.encode(x, return_info=True)
    assert "kl" in info
    assert info["kl"].ndim == 0


def test_kl_increases_with_mu_magnitude():
    """Larger |mu| should produce larger KL."""
    shape = (1, 4, 8, 16)
    sigma = torch.ones(*shape)
    c = torch.zeros(*shape, dtype=torch.cfloat)

    mu_small = torch.complex(torch.full(shape, 0.1), torch.zeros(*shape))
    mu_large = torch.complex(torch.full(shape, 5.0), torch.zeros(*shape))

    kl_small, _ = get_kl(mu_small, sigma, c)
    kl_large, _ = get_kl(mu_large, sigma, c)

    assert kl_large.item() > kl_small.item(), (
        "KL must increase with mu magnitude"
    )


# ---------------------------------------------------------------------------
# Test 17 — KL consistency: get_kl and get_cholesky_kl must agree
# ---------------------------------------------------------------------------

def test_kl_consistency_cholesky_vs_direct():
    """
    get_kl(mu, sigma, c) and get_cholesky_kl(mu, l11, l21, l22) must return the same
    value when (sigma, c) are derived from (l11, l21, l22) via the recovery formulas.
    This cross-checks that both KL paths are mathematically consistent.
    """
    shape = (2, 4, 8, 16)
    l11 = torch.rand(*shape).abs() + 0.1
    l21 = torch.randn(*shape)
    l22 = torch.rand(*shape).abs() + 0.1
    mu = _make_mu(shape)

    sigma = l11**2 + l21**2 + l22**2
    c = torch.complex(l11**2 - l21**2 - l22**2, 2.0 * l11 * l21)

    kl_direct, _ = get_kl(mu, sigma, c)
    kl_cholesky, _ = get_cholesky_kl(mu, l11, l21, l22)

    assert torch.isclose(kl_direct, kl_cholesky, rtol=1e-4), (
        f"get_kl={kl_direct.item():.6f} and get_cholesky_kl={kl_cholesky.item():.6f} "
        "must agree when (sigma,c) are derived from Cholesky factors"
    )


# ---------------------------------------------------------------------------
# Test 18 — get_cholesky_kl non-negativity
# ---------------------------------------------------------------------------

def test_get_cholesky_kl_non_negative():
    """get_cholesky_kl must be >= 0 for any valid l11, l21, l22 (l11, l22 > 0)."""
    shape = (2, 4, 8, 16)
    l11 = torch.rand(*shape).abs() + 0.1
    l21 = torch.randn(*shape)
    l22 = torch.rand(*shape).abs() + 0.1
    mu = _make_mu(shape)

    kl, _ = get_cholesky_kl(mu, l11, l21, l22)
    assert kl.item() >= 0.0, f"Cholesky KL should be non-negative, got {kl.item()}"


# ---------------------------------------------------------------------------
# Test 19 — Mode B encode: sigma > |c| constraint is always satisfied
# ---------------------------------------------------------------------------

def test_mode_b_constraint_always_satisfied():
    """
    In Mode B (Cholesky), the recovered (sigma, c) must satisfy sigma > |c| everywhere,
    because sigma^2 - |c|^2 = 4*l11^2*l22^2 > 0 by construction.
    Verify by running encode and checking there are no NaN/Inf in z.
    """
    torch.manual_seed(0)
    bottleneck = ComplexVAEBottleneck(apply_cholesky_constraints=True)
    # Large random inputs — if the constraint were violated, reparametrize would produce NaN
    x = _make_input(B=4, C=9, H=16, W=32)
    z, info = bottleneck.encode(x, return_info=True)

    assert not torch.isnan(z).any(), "z must not contain NaN (constraint violated)"
    assert not torch.isinf(z).any(), "z must not contain Inf (constraint violated)"
    assert not torch.isnan(info["kl"]), "KL must not be NaN in Mode B"


# ---------------------------------------------------------------------------
# Test 20 — reparametrize: E[z] ≈ mu (unbiased mean)
# ---------------------------------------------------------------------------

def test_reparametrize_unbiased_mean():
    """
    Over many samples, E[reparametrize(mu, sigma, c)] must converge to mu.
    Tests that the reparameterization is unbiased (no mean shift).
    """
    shape = (1, 1, 1, 1)
    mu = torch.complex(torch.tensor([[[[2.0]]]]), torch.tensor([[[[- 1.0]]]]))
    sigma = torch.ones(shape) * 0.5
    c = torch.zeros(shape, dtype=torch.cfloat)

    N = 10_000
    samples = torch.stack([reparametrize(mu, sigma, c) for _ in range(N)])  # (N, 1,1,1,1)
    mean = samples.mean(dim=0)

    assert abs(mean.real.item() - mu.real.item()) < 0.05, (
        f"E[z.real]={mean.real.item():.3f} should be ≈ mu.real={mu.real.item():.3f}"
    )
# ---------------------------------------------------------------------------
# Test 21 — encode() output shape (Proper mode)
# ---------------------------------------------------------------------------

def test_encode_shape_proper():
    """encode(x, return_info=True) with x=(2,4,8,16) -> z=(2,2,8,16) complex, kl scalar."""
    bottleneck = ComplexVAEBottleneck(proper=True)
    # Proper mode expects parameters_to_predict=2, so C=4 -> z has 4/2=2 channels
    x = _make_input(B=2, C=4, H=8, W=16)
    z, info = bottleneck.encode(x, return_info=True)

    assert torch.is_complex(z), "z must be complex"
    assert z.shape == (2, 2, 8, 16), f"Expected z shape (2,2,8,16), got {z.shape}"
    assert "kl" in info, "info dict must contain 'kl'"
    assert info["kl"].ndim == 0, "kl must be a scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# Test 22 — Proper KL non-negativity
# ---------------------------------------------------------------------------

def test_get_proper_kl_non_negative():
    """get_proper_kl should be >= 0 for random inputs."""
    shape = (2, 4, 8, 16)
    mu = _make_mu(shape)
    gamma = torch.rand(*shape).abs() + 0.1 # Strictly positive variance
    kl, _ = get_proper_kl(mu, gamma)
    assert kl.item() >= 0.0, f"Proper KL should be non-negative, got {kl.item()}"
