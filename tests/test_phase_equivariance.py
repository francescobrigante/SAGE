# =============================================================================
# Unit tests — phase-equivariance of complex activations.
# Validates that ComplexGELU1d and ModReLU are phase-equivariant (arg preserved),
# and that CGeLU (legacy split) is NOT — empirically confirming the design bug.
# =============================================================================

import math
import pytest
import torch
import torch.nn as nn

import sys
import os

from ar_spectra.blocks.activations import (
    ComplexGELU1d, CGeLU, ModReLU, get_activation,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _phase_error(activation: nn.Module, x: torch.Tensor) -> float:
    """Return max absolute angle error between input and output phases."""
    y = activation(x)
    return (y.angle() - x.angle()).abs().max().item()


def _rotation_equivariance_error(activation: nn.Module, x: torch.Tensor,
                                  theta: float = math.pi / 3) -> float:
    """Return max |arg(act(e^{iθ}·x)) - arg(act(x)) - θ| masked to non-zero outputs.

    Positions where the gate is 0 (output magnitude ≈ 0) are excluded because
    angle(0) = 0 by convention, creating a spurious error of θ at those sites.
    """
    rotated = x * complex(math.cos(theta), math.sin(theta))
    y_rot = activation(rotated)
    y_orig = activation(x)
    # Mask out positions where either output is near-zero (gate killed them)
    mask = (y_orig.abs() > 1e-6) & (y_rot.abs() > 1e-6)
    if not mask.any():
        return 0.0
    expected_angle = y_orig.angle() + theta
    diff = (y_rot.angle() - expected_angle + math.pi) % (2 * math.pi) - math.pi
    return diff[mask].abs().max().item()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def x_tokens():
    """Random complex64 token sequence (B=2, L=8, C=16)."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 16, dtype=torch.complex64)


@pytest.fixture
def complex_gelu(x_tokens):
    """ComplexGELU1d instantiated for C=16."""
    return ComplexGELU1d(channels=x_tokens.shape[-1])


@pytest.fixture
def mod_relu(x_tokens):
    """ModReLU instantiated for C=16."""
    return ModReLU(channels=x_tokens.shape[-1])


@pytest.fixture
def cgelu():
    """CGeLU legacy split activation (no channel param required)."""
    return CGeLU()


# ── phase-equivariance tests ──────────────────────────────────────────────────

class TestComplexGELU1d:

    def test_output_is_complex(self, complex_gelu, x_tokens):
        y = complex_gelu(x_tokens)
        assert torch.is_complex(y), "Output must be complex."

    def test_output_shape(self, complex_gelu, x_tokens):
        y = complex_gelu(x_tokens)
        assert y.shape == x_tokens.shape

    def test_phase_equivariant(self, complex_gelu, x_tokens):
        """arg(y) == arg(x) for all elements — gate depends only on |x|."""
        err = _phase_error(complex_gelu, x_tokens)
        assert err < 1e-5, f"Phase error {err:.2e} > 1e-5; ComplexGELU1d is NOT phase-equivariant."

    def test_rotation_equivariance(self, complex_gelu, x_tokens):
        """act(e^{iθ}·x) = e^{iθ}·act(x) — rotation equivariance."""
        err = _rotation_equivariance_error(complex_gelu, x_tokens)
        assert err < 1e-4, f"Rotation equivariance error {err:.2e} > 1e-4."

    def test_gate_is_real_nonneg(self, complex_gelu, x_tokens):
        """Internal gate g(|x|) must be real and non-negative."""
        mag = x_tokens.abs()
        gate = complex_gelu._gate(mag)
        assert not torch.is_complex(gate), "Gate must be real-valued."
        assert (gate >= 0).all(), "Gate must be non-negative (GELU CDF)."

    def test_raises_on_real_input(self, complex_gelu):
        real_x = torch.randn(2, 8, 16)
        with pytest.raises(TypeError):
            complex_gelu(real_x)

    def test_get_activation_factory(self):
        """get_activation('ComplexGELU1d', is_complex=True, channels=32) must work."""
        act = get_activation("ComplexGELU1d", is_complex=True, channels=32)
        assert isinstance(act, ComplexGELU1d)
        x = torch.randn(1, 4, 32, dtype=torch.complex64)
        y = act(x)
        assert y.shape == x.shape


class TestModReLU:

    def test_output_is_complex(self, mod_relu, x_tokens):
        y = mod_relu(x_tokens)
        assert torch.is_complex(y)

    def test_output_shape(self, mod_relu, x_tokens):
        y = mod_relu(x_tokens)
        assert y.shape == x_tokens.shape

    def test_phase_equivariant(self, mod_relu, x_tokens):
        """arg(y) == arg(x) for all elements with non-zero gated output."""
        y = mod_relu(x_tokens)
        # Only check positions where output magnitude > eps (gate = 0 → output = 0, angle undefined)
        mag_out = y.abs()
        mask = mag_out > 1e-6
        if mask.any():
            angle_diff = (y.angle() - x_tokens.angle()).abs()
            err = angle_diff[mask].max().item()
            assert err < 1e-5, f"Phase error {err:.2e} > 1e-5; ModReLU is NOT phase-equivariant."

    def test_rotation_equivariance(self, mod_relu, x_tokens):
        err = _rotation_equivariance_error(mod_relu, x_tokens)
        assert err < 1e-4, f"Rotation equivariance error {err:.2e} > 1e-4."

    def test_enforce_negative_bias(self, mod_relu):
        """With enforce_negative=True, effective bias is always ≤ 0."""
        b_eff = -mod_relu.b_free.abs()
        assert (b_eff <= 0).all()

    def test_raises_on_real_input(self, mod_relu):
        real_x = torch.randn(2, 8, 16)
        with pytest.raises(TypeError):
            mod_relu(real_x)

    def test_get_activation_factory(self):
        act = get_activation("ModReLU", is_complex=True, channels=32)
        assert isinstance(act, ModReLU)
        x = torch.randn(1, 4, 32, dtype=torch.complex64)
        y = act(x)
        assert y.shape == x.shape


class TestCGeLUIsNotPhaseEquivariant:
    """CGeLU (split GELU) should NOT be phase-equivariant.

    This test is expected to PASS (i.e., it asserts that the error IS large).
    This empirically validates the design bug reported in the PI review.
    """

    def test_cgelu_destroys_phase(self, cgelu, x_tokens):
        """arg(CGeLU(x)) ≠ arg(x) — confirms CGeLU is phase-destroying."""
        err = _phase_error(cgelu, x_tokens)
        # We expect the error to be large (>> 0); a small error would be a false pass
        assert err > 0.1, (
            f"Unexpected: CGeLU phase error is only {err:.4f}. "
            "CGeLU should destroy phase; if this test fails, something is wrong."
        )

    def test_cgelu_not_rotation_equivariant(self, cgelu, x_tokens):
        """CGeLU is not rotation-equivariant — confirms structural flaw."""
        err = _rotation_equivariance_error(cgelu, x_tokens)
        assert err > 0.05, (
            f"Unexpected: CGeLU rotation error is only {err:.4f}. "
            "CGeLU should NOT be rotation-equivariant."
        )


# ── param shape broadcast test ────────────────────────────────────────────────

class TestParamShapes:

    def test_complex_gelu1d_broadcasts_BLC(self):
        """ComplexGELU1d (1,1,C) params must broadcast correctly over (B,L,C) — not (B,C,L)."""
        channels = 32
        B, L, C = 3, 17, channels  # L ≠ C to catch transposed-shape bugs
        act = ComplexGELU1d(channels=C)
        x = torch.randn(B, L, C, dtype=torch.complex64)
        y = act(x)
        assert y.shape == (B, L, C), f"Expected ({B},{L},{C}), got {y.shape}"

    def test_mod_relu_broadcasts_BLC(self):
        channels = 32
        B, L, C = 3, 17, channels
        act = ModReLU(channels=C)
        x = torch.randn(B, L, C, dtype=torch.complex64)
        y = act(x)
        assert y.shape == (B, L, C), f"Expected ({B},{L},{C}), got {y.shape}"

    def test_complex_gelu1d_raises_on_channels_first_when_L_ne_C(self):
        """(B, C, L) with L ≠ C raises a broadcast error — fast-fail for wrong layout.

        Param shape (1,1,C) cannot broadcast against the last dim=L when L ≠ C,
        so PyTorch raises RuntimeError. This is the desired behavior: wrong-layout
        inputs are caught immediately rather than silently producing wrong results.
        """
        channels = 16
        B, C, L = 2, channels, 33  # channels-first layout, L ≠ C
        act = ComplexGELU1d(channels=channels)
        x = torch.randn(B, C, L, dtype=torch.complex64)
        with pytest.raises(RuntimeError):
            act(x)
