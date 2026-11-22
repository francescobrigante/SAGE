# ===============
# Exact training speed-ups of the generator: trainer.trainer.sdpa_train (attention through SDPA with
# autograd, sage/model/swin/sdpa.py enable_sdpa_train) and trainer.trainer.patch_linear (patch layers as one
# GEMM, sage/model/swin/patches.py enable_patch_linear). Guards: off by default; in float64 the output and
# every gradient (parameters and input) equal the original's (W-MSA / SW-MSA, baseline / XSA, padded grids
# with and without Q/V bias, the whole tiny SAGE with both keys); the inference path keeps its scope with
# both flags on; attn_drop becomes SDPA's dropout_p in training mode only; enabled=False restores the
# original bit for bit; complex modules are never touched; the state_dict does not change; no graph break
# under torch.compile. Config of the two keys: tests/test_compile_generator.py, tests/test_disc_speed.py.
# Training runs with the keys on vs off: tests/smoke/test_smoke_sdpa_train.py.
# ===============
import pytest
import torch

import sage.model.swin.attention as attention_module
from sage.model.swin import sdpa as sdpa_module
from sage.model.swin.attention import WindowAttention
from sage.model.swin.patches import enable_patch_linear
from sage.model.swin.sdpa import enable_sdpa, enable_sdpa_train
from test_compile_generator import _tiny_sage
from test_sdpa_attention import DIM, LOCAL, WINDOW, attentions, make_block, tokens

GRAD_RTOL = 1e-10      # max|on − off| / max|off| per tensor, float64 (~1e-14 measured)
# Whole tiny SAGE with patch_linear too, Σ|on − off| / Σ|off| per tensor: the GEMM of PatchUnembed rounds its
# input gradient differently from ConvTranspose2d's backward (6e-16 alone, and the GEMM is the one equal to an
# einsum), and the SAGE gradient amplifies that rounding (sums with heavy cancellation): 5e-8 measured.
WHOLE_RTOL = 1e-6


def _step(module, x, call=None):
    """Forward (through ``call``, default the module) + backward of a loss with a non-trivial gradient;
    returns the output, the input gradient and every parameter gradient of ``module``."""
    module.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = (call or module)(x)
    (y.pow(2).sum() + y.sum()).backward()
    return y.detach(), x.grad, {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def _assert_same_step(on, off):
    (y1, gx1, g1), (y0, gx0, g0) = on, off
    assert g1.keys() == g0.keys() and len(g0) > 0
    assert _rel(y1, y0) < GRAD_RTOL and _rel(gx1, gx0) < GRAD_RTOL
    worst = max(g0, key=lambda n: _rel(g1[n], g0[n]))
    assert _rel(g1[worst], g0[worst]) < GRAD_RTOL, (worst, _rel(g1[worst], g0[worst]))


def test_off_by_default():
    assert not any(a.sdpa_train for a in attentions(make_block(LOCAL, shift=(2, 16))))


@pytest.mark.parametrize("shift", [(0, 0), (2, 16)], ids=["w-msa", "sw-msa"])
@pytest.mark.parametrize("variant", ["baseline", "xsa"])
def test_gradients_match_original_in_float64(shift, variant):
    blk, x = make_block(LOCAL, shift=shift, variant=variant).train(), tokens(LOCAL, batch=2)
    off = _step(blk, x)
    assert enable_sdpa_train(blk) == 1 and blk.attn.sdpa_train
    _assert_same_step(_step(blk, x), off)


@pytest.mark.parametrize("qkv_bias", [True, False], ids=["fused-bias", "no-bias"])
def test_gradients_match_original_on_a_padded_grid(qkv_bias):
    """Time tokens not a multiple of the window (the block pads), SW-MSA; with the Q/V bias in the GEMM or none."""
    blk = make_block(LOCAL, shift=(2, 16), qkv_bias=qkv_bias).train()
    assert sdpa_module.fusable_qkv_bias(blk.attn) is qkv_bias
    torch.manual_seed(2)
    x = torch.randn(2, LOCAL[0] * (LOCAL[1] + 20), DIM, dtype=torch.float64)
    off = _step(blk, x)
    enable_sdpa_train(blk)
    _assert_same_step(_step(blk, x), off)


@pytest.mark.parametrize("patch_linear", [False, True], ids=["sdpa_train", "sdpa_train+patch_linear"])
def test_whole_sage_matches_original_in_float64(patch_linear):
    """Tiny pretraining SAGE in training mode, drop path off: encode -> decode -> loss -> backward."""
    ae, cfg = _tiny_sage()
    ae.train()
    x = torch.randn(2, int(cfg["data"]["model_channels"]), 1024, 128, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1))

    def step():
        ae.zero_grad(set_to_none=True)
        y = ae.decode(ae.encode(x, deterministic=True))
        y.pow(2).mean().backward()
        return y.detach(), {n: p.grad.clone() for n, p in ae.named_parameters() if p.grad is not None}

    y0, g0 = step()
    assert enable_sdpa_train(ae) > 0 and (not patch_linear or enable_patch_linear(ae) > 0)
    y1, g1 = step()
    assert g1.keys() == g0.keys() and _rel(y1, y0) < GRAD_RTOL
    if patch_linear:
        rel = {n: float((g1[n] - g0[n]).abs().sum() / g0[n].abs().sum().clamp_min(1e-300)) for n in g0}
        worst = max(rel, key=rel.get)
        assert rel[worst] < WHOLE_RTOL, (worst, rel[worst])
    else:
        worst = max(g0, key=lambda n: _rel(g1[n], g0[n]))
        assert _rel(g1[worst], g0[worst]) < GRAD_RTOL, (worst, _rel(g1[worst], g0[worst]))


def _record_paths(monkeypatch) -> list:
    calls = []
    for name in ("sdpa_forward", "sdpa_train_forward"):
        original = getattr(attention_module, name)

        def recorded(*args, _name=name, _fn=original, **kwargs):
            calls.append(_name)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(attention_module, name, recorded)
    return calls


def test_without_the_inference_path_every_call_takes_the_training_path(monkeypatch):
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    with torch.no_grad():
        ref = blk(x)
    calls = _record_paths(monkeypatch)
    enable_sdpa_train(blk)
    with torch.no_grad():                                  # validation: eval mode, no autograd
        out = blk(x)
    blk(x)                                                 # eval mode with autograd
    assert calls == ["sdpa_train_forward"] * 2
    assert torch.allclose(out, ref, rtol=0.0, atol=1e-12), (out - ref).abs().max()


def test_the_inference_path_keeps_its_scope_with_both_flags(monkeypatch):
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    enable_sdpa(blk)
    enable_sdpa_train(blk)
    calls = _record_paths(monkeypatch)
    with torch.no_grad():
        blk(x)                                             # eval, no autograd: inference path (cached bias)
    blk(x)                                                 # eval, autograd
    blk.train()
    with torch.no_grad():
        blk(x)                                             # training mode, no autograd
    assert calls == ["sdpa_forward", "sdpa_train_forward", "sdpa_train_forward"]


def test_attn_drop_is_the_sdpa_dropout_in_training_mode_only(monkeypatch):
    attn = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4, attn_drop=0.3).double()
    x = torch.randn(3, WINDOW[0] * WINDOW[1], DIM, dtype=torch.float64)
    seen = []
    original = sdpa_module._sdpa
    monkeypatch.setattr(sdpa_module, "_sdpa", lambda *a: seen.append(a[4]) or original(*a))
    enable_sdpa_train(attn)
    attn.train()
    attn(x)
    attn.eval()
    attn(x)
    assert seen == [0.3, 0.0]


def test_disable_restores_the_original_bit_for_bit():
    blk, x = make_block(LOCAL, shift=(2, 16)).train(), tokens(LOCAL)
    y0, gx0, g0 = _step(blk, x)
    enable_sdpa_train(blk)
    _step(blk, x)
    assert enable_sdpa_train(blk, False) == 1 and not blk.attn.sdpa_train
    y1, gx1, g1 = _step(blk, x)
    assert torch.equal(y1, y0) and torch.equal(gx1, gx0) and all(torch.equal(g1[n], g0[n]) for n in g0)


def test_complex_attention_is_never_touched():
    attn = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4, is_complex=True)
    assert enable_sdpa_train(attn) == 0 and not attn.sdpa_train


def test_state_dict_is_unchanged():
    blk = make_block(LOCAL, shift=(2, 16)).train()
    keys = set(blk.state_dict())
    enable_sdpa_train(blk)
    _step(blk, tokens(LOCAL))
    assert set(blk.state_dict()) == keys


def test_no_graph_break_under_torch_compile():
    blk, x = make_block(LOCAL, shift=(2, 16)).train(), tokens(LOCAL, batch=2)
    enable_sdpa_train(blk)
    ref = _step(blk, x)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(blk, backend="aot_eager", fullgraph=True)
        _assert_same_step(_step(blk, x, call=compiled), ref)
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shift", [(0, 0), (2, 16)], ids=["w-msa", "sw-msa"])
def test_cuda_kernel_gradients_match_original_in_float32(shift):
    """The fused CUDA kernel (bias with gradient) against the original, float32 without TF32."""
    blk = make_block(LOCAL, shift=shift).train().float().cuda()
    x = tokens(LOCAL, batch=2).float().cuda()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y0, gx0, g0 = _step(blk, x)
        enable_sdpa_train(blk)
        y1, gx1, g1 = _step(blk, x)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    rel = {n: float((g1[n] - g0[n]).abs().sum() / g0[n].abs().sum()) for n in g0}
    worst = max(rel, key=rel.get)
    assert _rel(y1, y0) < 1e-4 and float((gx1 - gx0).abs().sum() / gx0.abs().sum()) < 1e-4
    assert rel[worst] < 1e-4, (worst, rel[worst])
