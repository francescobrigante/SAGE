# ===============================================================
# Tests for variable-length multi-phase attention (c_vae/swin/varlen.py).
# Guards the two properties the fix must have: it is inert at the training
# resolution (so no published number can move), and a single phase 0
# reproduces the baseline bit-for-bit at any length.
# ===============================================================

import sys

import pytest
import torch


from c_vae.swin.swin_block import SwinTransformerBlock
from c_vae.swin.varlen import (VarlenConfig, enable_varlen, group_bounds,
                               is_collapsed, load_config, phase_weights, resolve)

# SAGE's deepest stage: grid (4, 32) at training, window (4, 32) → collapse guard fires.
GRID = (4, 32)
DIM = 64
WINDOW = (4, 32)


def make_block(shift=(0, 0)) -> SwinTransformerBlock:
    torch.manual_seed(0)
    blk = SwinTransformerBlock(
        dim=DIM, input_resolution=GRID, num_heads=4,
        window_size=WINDOW, shift_size=shift, mlp_type="swiglu",
        swiglu_hidden_ratio=3.0, attention_variant="xsa", norm_placement="res_post",
    )
    return blk.eval()


def tokens(time_tokens: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, GRID[0] * time_tokens, DIM)


# --------------------------------------------------------------------------
# Structural guards
# --------------------------------------------------------------------------

def test_collapse_guard_zeroes_the_shift():
    """The block under test really is the pathological one: shift forced to 0."""
    blk = make_block(shift=(2, 16))
    assert blk.shift_size == (0, 0)
    assert blk.window_size == WINDOW
    assert is_collapsed(blk)


def test_enable_varlen_only_touches_collapsed_blocks():
    collapsed = make_block()
    local = SwinTransformerBlock(dim=DIM, input_resolution=(16, 128), num_heads=4,
                                 window_size=WINDOW, shift_size=(0, 0)).eval()
    assert not is_collapsed(local)
    root = torch.nn.ModuleList([collapsed, local])
    assert enable_varlen(root, VarlenConfig((0, 16), "tri")) == 1
    assert collapsed.varlen is not None and local.varlen is None


# --------------------------------------------------------------------------
# Non-regression: the fix must be invisible unless it can help
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["mean2", "tri2", "hard2", "mean4", "tri4", "hard4"])
def test_bit_identical_at_training_resolution(variant):
    """At 32 time tokens the window already spans everything: no variant may change a bit."""
    x = tokens(GRID[1])
    blk = make_block()
    with torch.no_grad():
        baseline = blk(x)
        enable_varlen(blk, resolve(variant))
        assert torch.equal(blk(x), baseline)


@pytest.mark.parametrize("combine", ["mean", "tri", "hard"])
def test_single_phase_zero_reproduces_baseline(combine):
    """One phase at offset 0 is the baseline partition — must match exactly."""
    x = tokens(4 * GRID[1])
    blk = make_block()
    with torch.no_grad():
        baseline = blk(x)
        enable_varlen(blk, VarlenConfig((0,), combine))
        assert torch.equal(blk(x), baseline)


def test_disable_restores_baseline():
    x = tokens(4 * GRID[1])
    blk = make_block()
    with torch.no_grad():
        baseline = blk(x)
        enable_varlen(blk, resolve("tri4"))
        assert not torch.equal(blk(x), baseline)
        enable_varlen(blk, None)
        assert torch.equal(blk(x), baseline)


def test_multiphase_changes_output_on_long_input():
    """Sanity: on long input the variants must actually do something."""
    x = tokens(5 * GRID[1])
    blk = make_block()
    with torch.no_grad():
        baseline = blk(x)
        enable_varlen(blk, resolve("hard4"))
        assert not torch.allclose(blk(x), baseline)


# --------------------------------------------------------------------------
# Attention-group bookkeeping (this is what the weights are built on)
# --------------------------------------------------------------------------

def test_group_bounds_match_the_window_partition():
    """Unshifted groups are exactly the windows."""
    start, end = group_bounds(160, 32, 0)
    assert (start[:32] == 0).all() and (end[:32] == 32).all()
    assert (start[32:64] == 32).all() and (end[32:64] == 64).all()


def test_group_bounds_see_the_wrapped_head():
    """torch.roll sends the first `phase` tokens into the last window, where the
    attention mask isolates them: their real context is `phase` tokens, not 32."""
    start, end = group_bounds(160, 32, 16)
    assert (int(start[0]), int(end[0])) == (0, 16)      # not (0, 32)
    assert (int(start[32]), int(end[32])) == (16, 48)   # interior token: full window


def test_group_bounds_stop_at_the_padding():
    """A group truncated by the zero padding ends at the last real token."""
    start, end = group_bounds(150, 32, 0)
    assert (int(start[149]), int(end[149])) == (128, 150)


def test_weights_are_a_partition_of_unity():
    for variant in ("mean2", "tri2", "hard2", "mean4", "tri4", "hard4"):
        w = phase_weights(200, 32, resolve(variant))
        assert torch.allclose(w.sum(dim=0), torch.ones(200), atol=1e-6)
        assert (w >= 0).all()


def test_hard_and_tri_starve_the_edge_context():
    """At a baseline block boundary, phase 0 gives the token zero margin.
    mean keeps 1/P of it, tri nearly none, hard exactly none — this ordering is
    what the measured residual energy dip follows."""
    cfg = {c: resolve(f"{c}4") for c in ("mean", "tri", "hard")}
    w = {c: phase_weights(160, 32, cfg[c])[:, 32] for c in cfg}
    assert w["mean"][0] == pytest.approx(0.25)
    assert w["tri"][0] < 0.05
    assert w["hard"][0] == 0.0
    assert w["hard"][2] == 1.0          # phase 16 centres t=32 perfectly


def test_no_phase_is_preferred_at_the_true_start_of_the_clip():
    """t=0 has no left context in any phase, so the wrapped phase must not win."""
    w = phase_weights(160, 32, resolve("hard4"))[:, 0]
    assert torch.allclose(w, torch.full((4,), 0.25))


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------

def test_resolve_off_disables():
    assert resolve("off") is None and resolve("baseline") is None


def test_resolve_rejects_unknown_variant():
    with pytest.raises(KeyError):
        resolve("tri3")


def test_config_validates_its_inputs():
    with pytest.raises(ValueError):
        VarlenConfig((0, 16), "median")
    with pytest.raises(ValueError):
        VarlenConfig((0, 16, 16), "tri")
    with pytest.raises(ValueError):
        VarlenConfig((), "tri")


# --------------------------------------------------------------------------
# The project default
# --------------------------------------------------------------------------
# From 2026-09-22 config/inference/varlen.yaml ships mode: tri2, and
# EuleroEncodeDecode applies it to every checkpoint it loads. These two tests
# exist so that default cannot drift silently: the first pins what it is, the
# second pins the property that makes shipping it safe.

def test_project_default_is_tri2():
    """The shipped default is the variant chosen in VARLEN_SEAMS.md section 8.4."""
    cfg = resolve(str(load_config().mode))
    assert cfg is not None, "default must not be 'off' — the fix is meant to ship on"
    assert cfg.phases == (0, 16)
    assert cfg.combine == "tri"


def test_project_default_is_inert_at_training_resolution():
    """Whatever the default is, it must not move a single training-length number."""
    x = tokens(GRID[1])                       # exactly the training grid: 32 time tokens
    blk = make_block()
    with torch.no_grad():
        baseline = blk(x)
        enable_varlen(blk, resolve(str(load_config().mode)))
        after = blk(x)
    assert torch.equal(baseline, after)
