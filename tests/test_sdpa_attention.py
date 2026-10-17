# ===============================================================
# Tests for the opt-in SDPA attention path (sage/model/swin/sdpa.py).
# Guards: off by default (the original forward runs untouched); in float64 the
# SDPA path equals the original to rounding on W-MSA, SW-MSA, batch > 1, XSA and
# variable-length multi-phase blocks; the cached bias follows the weights (load,
# cast, in-place edits, compiled models) and works under inference_mode; calls above
# the CUDA launch grid limit are split with the same values, calls below it are the
# single call; training and autograd always take the original forward; the state_dict
# does not change.
# ===============================================================

import math

import pytest
import torch

from sage.model.swin import sdpa as sdpa_module
from sage.model.swin.attention import WindowAttention
from sage.model.swin.block import SwinTransformerBlock
from sage.model.swin.sdpa import cpb_bias, disable_sdpa, enable_sdpa
from sage.model.swin.varlen import enable_varlen, resolve

DIM = 64
WINDOW = (4, 32)
LOCAL = (16, 128)      # window smaller than the grid: W-MSA / SW-MSA with a shift mask
COLLAPSED = (4, 32)    # SAGE's deepest stage: one window covers the training grid
ATOL = 1e-12           # float64: the two formulas differ only by rounding (~1e-15 measured)


def make_block(resolution, shift=(0, 0), variant="xsa", qkv_bias=True, seed=0) -> SwinTransformerBlock:
    torch.manual_seed(seed)
    blk = SwinTransformerBlock(
        dim=DIM, input_resolution=resolution, num_heads=4,
        window_size=WINDOW, shift_size=shift, mlp_type="swiglu", qkv_bias=qkv_bias,
        swiglu_hidden_ratio=3.0, attention_variant=variant, norm_placement="res_post",
    )
    return blk.eval().double()


def tokens(resolution, batch: int = 1, time_scale: int = 1) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, resolution[0] * resolution[1] * time_scale, DIM, dtype=torch.float64)


def attentions(module: torch.nn.Module) -> list[WindowAttention]:
    return [m for m in module.modules() if isinstance(m, WindowAttention)]


# --------------------------------------------------------------------------
# Default: nothing changes unless asked
# --------------------------------------------------------------------------

def test_off_by_default():
    blk = make_block(LOCAL, shift=(2, 16))
    assert not any(a.sdpa for a in attentions(blk))


def test_disable_restores_the_original_bit_for_bit():
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    with torch.no_grad():
        ref = blk(x)
        assert enable_sdpa(blk) == 1
        disable_sdpa(blk)
        assert torch.equal(blk(x), ref)
    assert not any(a.sdpa for a in attentions(blk))
    assert not hasattr(blk.attn, "_sdpa_cpb") and not blk.attn._load_state_dict_post_hooks


# --------------------------------------------------------------------------
# Equivalence: same formula, fused kernel
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(0, 0), (2, 16)], ids=["w-msa", "sw-msa"])
@pytest.mark.parametrize("variant", ["baseline", "xsa"])
@pytest.mark.parametrize("batch", [1, 3])
def test_matches_original_in_float64(shift, variant, batch):
    blk, x = make_block(LOCAL, shift=shift, variant=variant), tokens(LOCAL, batch)
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk)
        out = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_matches_original_with_variable_length_phases():
    """Collapsed block at 4x the training length with the default multi-phase attention (tri2)."""
    blk, x = make_block(COLLAPSED), tokens(COLLAPSED, time_scale=4)
    enable_varlen(blk, resolve("tri2"))
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk)
        out = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_matches_original_on_a_padded_grid_without_qkv_bias():
    """Time tokens not a multiple of the window (the block pads) and no Q/V bias."""
    blk = make_block(LOCAL, shift=(2, 16), qkv_bias=False)
    torch.manual_seed(2)
    x = torch.randn(2, LOCAL[0] * (LOCAL[1] + 20), DIM, dtype=torch.float64)
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk)
        out = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_matches_original_under_torch_compile():
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk)
        out = torch.compile(blk, backend="aot_eager", dynamic=False)(x)
    torch._dynamo.reset()
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


# --------------------------------------------------------------------------
# Cache: the CPB bias follows the weights, their dtype and device
# --------------------------------------------------------------------------

def test_weights_loaded_after_enabling_are_used():
    blk, other, x = make_block(LOCAL, shift=(2, 16)), make_block(LOCAL, shift=(2, 16), seed=7), tokens(LOCAL)
    with torch.no_grad():
        ref = other(x)
        enable_sdpa(blk)
        blk.load_state_dict(other.state_dict())
        out = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_dtype_change_after_enabling():
    blk, x = make_block(LOCAL, shift=(2, 16)).float(), tokens(LOCAL)
    enable_sdpa(blk)
    blk.double()
    with torch.no_grad():
        out = blk(x)
        disable_sdpa(blk)
        ref = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_weights_edited_in_place_after_an_sdpa_call_are_used():
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    enable_sdpa(blk)
    with torch.no_grad():
        blk(x)                                             # fills the cache
        blk.attn.cpb_mlp[0].weight.mul_(0.5)
        out = blk(x)
        disable_sdpa(blk)
        ref = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_weights_loaded_into_a_compiled_model_are_used():
    blk, other, x = make_block(LOCAL, shift=(2, 16)), make_block(LOCAL, shift=(2, 16), seed=7), tokens(LOCAL)
    with torch.no_grad():
        ref = other(x)
        enable_sdpa(blk)
        compiled = torch.compile(blk, backend="aot_eager", dynamic=False)
        compiled(x)
        blk.load_state_dict(other.state_dict())
        out = compiled(x)
    torch._dynamo.reset()
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def test_cached_bias_ignores_and_restores_the_matmul_precision():
    blk = make_block(LOCAL, shift=(2, 16)).float()
    before = torch.get_float32_matmul_precision()          # "high" once a test has imported train.py
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            ref = cpb_bias(blk.attn)                       # full FP32
        torch.set_float32_matmul_precision("medium")
        enable_sdpa(blk)
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(before)
    assert torch.equal(blk.attn._sdpa_cpb, ref)


def test_dtype_cast_refreshes_the_cache_without_a_call():
    blk = make_block(LOCAL, shift=(2, 16)).float()
    enable_sdpa(blk)
    blk.double()
    with torch.no_grad():
        assert torch.equal(blk.attn._sdpa_cpb, cpb_bias(blk.attn))
    assert blk.attn._sdpa_cpb.dtype == torch.float64


def test_runs_under_inference_mode():
    """Tensors made under inference_mode (here the block's shift mask) have no version counter."""
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    with torch.inference_mode():
        ref = blk(x)
        enable_sdpa(blk)
        out = blk(x)
        again = blk(x)
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()
    assert torch.equal(out, again)


def test_stamp_accepts_inference_tensors():
    with torch.inference_mode():
        t = torch.zeros(3)
    assert sdpa_module._stamp((t,)) == ((t.data_ptr(), -1),)


def test_module_pickles_after_an_sdpa_call():
    import io
    blk = make_block(LOCAL, shift=(2, 16))
    enable_sdpa(blk)
    with torch.no_grad():
        blk(tokens(LOCAL))
    torch.save(blk, io.BytesIO())


# --------------------------------------------------------------------------
# Launch grid: SDPA calls above the CUDA limit on heads / batch are split
# --------------------------------------------------------------------------

# LOCAL grid, batch 3: 16 windows of 128 tokens, 4 heads of 16 channels.
# W-MSA puts the windows on the batch axis, SW-MSA on the heads axis: (q shape, attn_mask shape).
GRID_CALLS = {
    "w-msa": ((48, 4, 128, 16), (1, 4, 128, 128)),
    "sw-msa": ((3, 64, 128, 16), (1, 64, 128, 128)),
}
SHIFTS = {"w-msa": (0, 0), "sw-msa": (2, 16)}


def record_sdpa_calls(monkeypatch) -> list:
    """(q shape, attn_mask shape) of every F.scaled_dot_product_attention call from here on."""
    calls, sdpa = [], torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kwargs):
        calls.append((tuple(q.shape), tuple(attn_mask.shape)))
        return sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    return calls


@pytest.mark.parametrize("branch", ["w-msa", "sw-msa"])
def test_at_or_below_the_grid_limit_is_the_single_call(branch, monkeypatch):
    """The default limit, and a limit equal to the largest grid axis: one call on the whole tensors."""
    blk, x = make_block(LOCAL, shift=SHIFTS[branch]), tokens(LOCAL, batch=3)
    enable_sdpa(blk)
    calls = record_sdpa_calls(monkeypatch)
    for limit in (sdpa_module._GRID_LIMIT, max(GRID_CALLS[branch][0][:2])):
        monkeypatch.setattr(sdpa_module, "_GRID_LIMIT", limit)
        calls.clear()
        with torch.no_grad():
            blk(x)
        assert calls == [GRID_CALLS[branch]]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("limit", [2, 5])
@pytest.mark.parametrize("branch", ["w-msa", "sw-msa"])
def test_split_above_the_grid_limit_matches_the_single_call(branch, limit, dtype, monkeypatch):
    """Both grid axes cut (uneven last slice with limit 5): bit-identical to the single call."""
    blk, x = make_block(LOCAL, shift=SHIFTS[branch]).to(dtype), tokens(LOCAL, batch=3).to(dtype)
    enable_sdpa(blk)
    with torch.no_grad():
        ref = blk(x)
        monkeypatch.setattr(sdpa_module, "_GRID_LIMIT", limit)
        calls = record_sdpa_calls(monkeypatch)
        out = blk(x)
    (batch, heads, _, _), _ = GRID_CALLS[branch]
    assert len(calls) == math.ceil(batch / limit) * math.ceil(heads / limit)
    assert all(q[0] <= limit and q[1] <= limit and m[:2] == (1, q[1]) for q, m in calls)
    assert torch.equal(out, ref), (out - ref).abs().max()


# --------------------------------------------------------------------------
# Scope: inference only, real-valued only, checkpoint format untouched
# --------------------------------------------------------------------------

def test_autograd_and_training_take_the_original_forward():
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    ref_grad = blk(x)                                      # autograd on, original forward
    blk.train()
    with torch.no_grad():
        ref_train = blk(x)
    blk.eval()
    enable_sdpa(blk)
    assert torch.equal(blk(x), ref_grad)
    blk.train()
    with torch.no_grad():
        assert torch.equal(blk(x), ref_train)


def test_state_dict_is_unchanged():
    blk = make_block(LOCAL, shift=(2, 16))
    keys = set(blk.state_dict())
    enable_sdpa(blk)
    with torch.no_grad():
        blk(tokens(LOCAL))
    assert set(blk.state_dict()) == keys
    fresh = make_block(LOCAL, shift=(2, 16))
    fresh.load_state_dict(blk.state_dict(), strict=True)


def test_complex_attention_is_never_touched():
    attn = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4, is_complex=True)
    assert enable_sdpa(attn) == 0
    assert not attn.sdpa
