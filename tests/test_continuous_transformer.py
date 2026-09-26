# ===============
# Tests for transformer_sat.ContinuousTransformer + AdaLN global-cond in
# TransformerBlock (LATENT_ALIGNMENT_PLAN Fase 6 backbone). Verifies backward
# compatibility (no global_cond → unchanged path), shapes, AdaLN effect, and grads.
# ===============
import torch
import pytest

from sage.nn.transformer import (
    TransformerBlock, ContinuousTransformer, TransformerResamplingBlock,
)


def test_block_backward_compatible_without_global_cond():
    """A block with global_cond_dim=None must ignore any global_cond passed in."""
    torch.manual_seed(0)
    blk = TransformerBlock(dim=64, dim_heads=32, add_rope=True).eval()
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        out_a = blk(x)
        out_b = blk(x, global_cond=torch.randn(2, 6 * 64))   # must be ignored
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_resampling_block_still_runs():
    """The discriminator's TRB (uses TransformerBlock internally) is unaffected."""
    trb = TransformerResamplingBlock(in_channels=32, out_channels=48, stride=2,
                                     transformer_depth=2).eval()
    x = torch.randn(2, 32, 8)                                # (B, C_in, T)
    with torch.no_grad():
        y = trb(x)
    assert y.shape[0] == 2 and y.shape[1] == 48


def test_continuous_transformer_shape():
    ct = ContinuousTransformer(dim=64, depth=2, dim_in=16, dim_out=16,
                               global_cond_dim=32).eval()
    x = torch.randn(2, 12, 16)                               # (B, T, dim_in)
    cond = torch.randn(2, 32)
    with torch.no_grad():
        y = ct(x, global_cond=cond)
    assert y.shape == (2, 12, 16)


def test_adaln_conditioning_changes_output():
    """Different global_cond must change the output (AdaLN is active)."""
    torch.manual_seed(0)
    ct = ContinuousTransformer(dim=64, depth=2, dim_in=16, dim_out=16,
                               global_cond_dim=32, zero_init_branch_outputs=False).eval()
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        y0 = ct(x, global_cond=torch.zeros(2, 32))
        y1 = ct(x, global_cond=torch.randn(2, 32) * 5.0)
    assert not torch.allclose(y0, y1, atol=1e-4)


def test_gradients_flow_to_input_and_cond_embedder():
    ct = ContinuousTransformer(dim=64, depth=2, dim_in=16, dim_out=16,
                               global_cond_dim=32, zero_init_branch_outputs=False)
    x = torch.randn(2, 12, 16, requires_grad=True)
    cond = torch.randn(2, 32)
    ct(x, global_cond=cond).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    g = ct.global_cond_embedder[0].weight.grad
    assert g is not None and g.abs().sum() > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
