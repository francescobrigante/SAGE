# ===============================================================
# Tests for the FlexAttention option of the SDPA path (sage/model/swin/sdpa.py, flex):
# inside a torch.compile graph on CUDA the window attention runs through FlexAttention,
# with the Log-CPB bias and the SW-MSA mask added by a score_mod that computes the
# regions of shift_mask from the token indices.
# Guards: off by default, set and cleared by enable_sdpa, only where the head size suits
# the fused kernels; the score_mod rebuilds shift_mask exactly (every shift, sh == 0 and
# sw == 0 included, padded grids) and reads the right window after a grid split; in
# float64 the FlexAttention path equals SDPA and the original attention to rounding
# (W-MSA, SW-MSA, padded grids, multi-phase blocks); it runs only inside compiled graphs
# (eager and the fallbacks keep SDPA); no graph break. The CPU tests open the device and
# dtype gates (the compiled CPU kernels are not used in production); the CUDA tests run
# the real fused kernels under inductor.
# ===============================================================

import itertools

import pytest
import torch
import torch.nn as nn

from sage.model.decoder import SAGEDecoder
from sage.model.encoder import SAGEEncoder
from sage.model.swin import sdpa as S
from sage.model.swin.attention import WindowAttention
from sage.model.swin.block import SwinTransformerBlock, shift_mask
from sage.model.swin.sdpa import ShiftGeometry, disable_sdpa, enable_sdpa, flex_score_mod, flex_supported
from sage.model.swin.varlen import enable_varlen, resolve

pytestmark = pytest.mark.skipif(not S.FLEX_AVAILABLE, reason="torch without FlexAttention")

DIM = 64
HEADS = 4              # head size 16: the smallest the fused kernels take
WINDOW = (4, 32)
LOCAL = (16, 128)      # window smaller than the grid: W-MSA / SW-MSA with a shift mask
COLLAPSED = (4, 32)    # SAGE's deepest stage: one window covers the training grid
ATOL = 1e-12           # float64: the formulations differ only by rounding


def make_block(resolution, shift=(0, 0), variant="xsa", seed=0, dim=DIM, heads=HEADS) -> SwinTransformerBlock:
    torch.manual_seed(seed)
    blk = SwinTransformerBlock(
        dim=dim, input_resolution=resolution, num_heads=heads,
        window_size=WINDOW, shift_size=shift, mlp_type="swiglu",
        swiglu_hidden_ratio=3.0, attention_variant=variant, norm_placement="res_post",
    )
    with torch.no_grad():
        blk.attn.q_bias.normal_(0.0, 0.5)
        blk.attn.v_bias.normal_(0.0, 0.5)
    return blk.eval().double()


def tokens(resolution, batch: int = 1, time_tokens=None, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(1)
    width = resolution[1] if time_tokens is None else time_tokens
    return torch.randn(batch, resolution[0] * width, DIM, dtype=dtype)


def assert_close(out: torch.Tensor, ref: torch.Tensor, atol: float = ATOL) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.allclose(out, ref, rtol=0.0, atol=atol), (out - ref).abs().max()


@pytest.fixture
def flex_on_cpu(monkeypatch):
    """Open the device and dtype gates on the CPU (aot_eager runs flex_attention's reference implementation)."""
    monkeypatch.setattr(S, "_FLEX_DEVICES", ("cuda", "cpu"))
    monkeypatch.setattr(S, "_FLEX_DTYPES", (torch.float32, torch.float64))
    yield
    torch._dynamo.reset()


def record_flex_calls(monkeypatch) -> list:
    """(q shape, shift) of every _flex call from here on (inside a compiled graph: one per trace)."""
    calls, flex = [], S._flex

    def spy(q, k, v, cpb, shift):
        calls.append((tuple(q.shape), shift))
        return flex(q, k, v, cpb, shift)

    monkeypatch.setattr(S, "_flex", spy)
    return calls


def compiled(module: nn.Module) -> nn.Module:
    return torch.compile(module, backend="aot_eager", dynamic=False)


# --------------------------------------------------------------------------
# Switch: off by default, set and cleared by enable_sdpa where supported
# --------------------------------------------------------------------------

def test_off_by_default_and_set_by_enable_sdpa():
    blk = make_block(LOCAL)
    assert not blk.attn.flex and flex_supported(blk.attn)
    enable_sdpa(blk)
    assert blk.attn.flex
    enable_sdpa(blk, flex=False)
    assert blk.attn.sdpa and not blk.attn.flex
    enable_sdpa(blk)
    disable_sdpa(blk)
    assert not (blk.attn.sdpa or blk.attn.flex)


@pytest.mark.parametrize("dim,heads,supported", [(64, 4, True), (128, 4, True), (64, 8, False), (96, 4, False)],
                         ids=["d16", "d32", "d8", "d24"])
def test_only_head_sizes_of_the_fused_kernels(dim, heads, supported):
    """A power of two, at least 16 (SAGE: 32); elsewhere enable_sdpa leaves flex off."""
    attn = WindowAttention(dim=dim, window_size=WINDOW, num_heads=heads)
    assert flex_supported(attn) is supported
    enable_sdpa(attn)
    assert attn.sdpa and attn.flex is supported


# --------------------------------------------------------------------------
# score_mod: shift_mask rebuilt from the token indices
# --------------------------------------------------------------------------

# (Hp, Wp, wh, ww): SAGE's stage-1 grid, a grid of one window row (Hp == wh), several windows on both axes,
# square windows, and a grid that the padding made a multiple of the window.
GRIDS = [(16, 128, 4, 32), (4, 96, 4, 32), (12, 40, 4, 8), (6, 6, 3, 3), (8, 160, 4, 32)]
GEOMETRIES = [ShiftGeometry(Hp, Wp, wh, ww, sh, sw) for Hp, Wp, wh, ww in GRIDS
              for sh, sw in itertools.product(range(wh), range(0, ww, max(ww // 4, 1))) if max(sh, sw) > 0]


def bias_from_score_mod(geometry: ShiftGeometry, cpb: torch.Tensor, offset: int = 0, rows=None) -> torch.Tensor:
    """score_mod(0, b, h, q, k) for every (b, h, q, k): the bias the FlexAttention path adds, (rows, h, N, N)."""
    h, N, _ = cpb.shape
    rows = geometry.windows if rows is None else rows
    b, hh, q, k = torch.meshgrid(torch.arange(rows), torch.arange(h), torch.arange(N), torch.arange(N), indexing="ij")
    return flex_score_mod(cpb, geometry, offset)(torch.zeros(b.shape, dtype=cpb.dtype), b, hh, q, k)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"{g.Hp}x{g.Wp}-w{g.wh}x{g.ww}-s{g.sh},{g.sw}")
def test_score_mod_rebuilds_shift_mask(geometry):
    """Equal to cpb + shift_mask bit for bit: the same region pattern (an axis without shift is one region, as
    shift_mask's slice(-0, None) makes it) and the same -100."""
    g = geometry
    torch.manual_seed(0)
    cpb = 16 * torch.rand(2, g.wh * g.ww, g.wh * g.ww, dtype=torch.float64)
    mask = shift_mask(g.Hp, g.Wp, (g.wh, g.ww), (g.sh, g.sw), torch.device("cpu")).double()   # (nW, N, N)
    assert torch.equal(bias_from_score_mod(g, cpb), cpb.unsqueeze(0) + mask.unsqueeze(1))
    assert torch.equal(bias_from_score_mod(g, torch.zeros_like(cpb))[:, 0] == 0, mask == 0)


def test_score_mod_without_shift_is_the_bias_alone():
    torch.manual_seed(0)
    cpb = torch.rand(2, 8, 8, dtype=torch.float64)
    b, h, q, k = torch.meshgrid(torch.arange(3), torch.arange(2), torch.arange(8), torch.arange(8), indexing="ij")
    assert torch.equal(flex_score_mod(cpb)(torch.zeros(b.shape, dtype=torch.float64), b, h, q, k),
                       cpb.expand(3, -1, -1, -1))


@pytest.mark.parametrize("offset", [0, 1, 5, 7])
def test_score_mod_offset_reads_the_window_of_each_row(offset):
    """A slice starting at batch row `start` passes offset = start mod nW: its rows see the windows they hold."""
    g = ShiftGeometry(12, 40, 4, 8, 2, 4)                              # nW = 15
    cpb = torch.zeros(1, g.wh * g.ww, g.wh * g.ww, dtype=torch.float64)
    full = bias_from_score_mod(g, cpb, rows=3 * g.windows)            # 3 images
    for start in (offset, offset + g.windows):
        part = bias_from_score_mod(g, cpb, offset=start % g.windows, rows=10)
        assert torch.equal(part, full[start:start + 10])


# --------------------------------------------------------------------------
# _flex: FlexAttention (reference implementation, float64) equals SDPA, also split
# --------------------------------------------------------------------------

def qkv_and_bias(geometry, batch=3, heads=HEADS, d=16, seed=0):
    torch.manual_seed(seed)
    nW, N = (geometry.windows, geometry.wh * geometry.ww) if geometry else (6, 128)
    q, k, v = (torch.randn(nW * batch, heads, N, d, dtype=torch.float64) for _ in range(3))
    cpb = 16 * torch.rand(heads, N, N, dtype=torch.float64)
    return q, k, v, cpb, nW


@pytest.mark.parametrize("geometry", [None, ShiftGeometry(8, 160, 4, 32, 2, 16), ShiftGeometry(4, 96, 4, 32, 0, 16)],
                         ids=["w-msa", "sw-msa", "phase"])
def test_flex_equals_sdpa_in_float64(geometry):
    q, k, v, cpb, nW = qkv_and_bias(geometry)
    out = S._flex(q, k, v, cpb, geometry)
    if geometry is None:
        ref = S._sdpa(q, k, v, cpb.unsqueeze(0))
    else:
        g = geometry
        mask = shift_mask(g.Hp, g.Wp, (g.wh, g.ww), (g.sh, g.sw), torch.device("cpu")).double()
        bias = (cpb.unsqueeze(0) + mask.unsqueeze(1)).repeat(q.shape[0] // nW, 1, 1, 1)   # (nW·B, h, N, N)
        ref = S._sdpa(q, k, v, bias)
    assert_close(out, ref)


def record_flex_attention_calls(monkeypatch) -> list:
    """(batch rows, offset seen by row 0) of every flex_attention call."""
    import torch.nn.attention.flex_attention as fa
    calls, real = [], fa.flex_attention

    def spy(q, k, v, score_mod=None, **kwargs):
        calls.append(q.shape[0])
        return real(q, k, v, score_mod=score_mod, **kwargs)

    monkeypatch.setattr(fa, "flex_attention", spy)
    return calls


@pytest.mark.parametrize("limit,sizes", [(S._GRID_LIMIT, [45]), (4 * 30, [30, 15]), (4 * 4, [4] * 11 + [1])],
                         ids=["single", "whole-images", "inside-images"])
def test_grid_split_keeps_the_window_of_every_row(limit, sizes, monkeypatch):
    """batch·heads above the launch cap: cut along the batch axis, in whole images when one fits (2 images of nW
    = 15 windows per slice), else with each slice's window offset; the values do not change."""
    g = ShiftGeometry(12, 40, 4, 8, 2, 4)                              # nW = 15, batch 3: 45 rows of 4 heads
    q, k, v, cpb, _ = qkv_and_bias(g, d=16)
    ref = S._flex(q, k, v, cpb, g)
    monkeypatch.setattr(S, "_GRID_LIMIT", limit)
    calls = record_flex_attention_calls(monkeypatch)
    out = S._flex(q, k, v, cpb, g)
    assert calls == sizes and all(rows * 4 <= limit for rows in calls)
    assert_close(out, ref)


# --------------------------------------------------------------------------
# In the model: compiled graphs take FlexAttention (CPU: gates opened, aot_eager)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(0, 0), (2, 16)], ids=["w-msa", "sw-msa"])
@pytest.mark.parametrize("time_tokens", [LOCAL[1], LOCAL[1] + 20], ids=["exact", "padded"])
def test_compiled_block_takes_flex_and_matches_the_original(shift, time_tokens, flex_on_cpu, monkeypatch):
    blk, x = make_block(LOCAL, shift=shift), tokens(LOCAL, 3, time_tokens)
    with torch.no_grad():
        ref = blk(x)                                                     # original attention
        enable_sdpa(blk)
        calls = record_flex_calls(monkeypatch)
        out = compiled(blk)(x)
    assert len(calls) == 1 and (calls[0][1] is None) == (shift == (0, 0))
    assert_close(out, ref)


@pytest.mark.parametrize("shared_phases", [True, False], ids=["shared", "per-phase"])
@pytest.mark.parametrize("preset", ["tri2", "tri4"])
def test_compiled_multiphase_block_takes_flex_and_matches_the_original(preset, shared_phases, flex_on_cpu,
                                                                       monkeypatch):
    """Variable-length blocks: every phase's mask (shift (0, phase), sh == 0) comes from its geometry."""
    blk, x = make_block(COLLAPSED), tokens(COLLAPSED, 2, 4 * COLLAPSED[1] + 22)
    enable_varlen(blk, resolve(preset))
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk, shared_phases=shared_phases)
        calls = record_flex_calls(monkeypatch)
        out = compiled(blk)(x)
    phases = resolve(preset).phases
    assert [s for _, s in calls] == [ShiftGeometry(4, 160, 4, 32, 0, p) if p else None for p in phases]
    assert_close(out, ref)


def test_flex_runs_only_where_its_gates_hold(flex_on_cpu, monkeypatch):
    """Eager calls, flex off, a device or dtype outside the gates, and a bias the graph would have to compute
    (cache missing) keep SDPA; the compiled default float64 model does not take it either."""
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL)
    enable_sdpa(blk)
    calls = record_flex_calls(monkeypatch)
    with torch.no_grad():
        ref = blk(x)                                                     # eager
        assert calls == []
        enable_sdpa(blk, flex=False)
        compiled(blk)(x)
        torch._dynamo.reset()
        enable_sdpa(blk)
        for gate, value in (("_FLEX_DEVICES", ("cuda",)), ("_FLEX_DTYPES", (torch.float32,))):
            with monkeypatch.context() as m:
                m.setattr(S, gate, value)
                compiled(blk)(x)
                torch._dynamo.reset()
        del blk.attn._sdpa_cpb                                           # the graph recomputes the bias: SDPA
        assert_close(compiled(blk)(x), ref)
    assert calls == []


def test_flex_in_float32_matches_sdpa(flex_on_cpu, monkeypatch):
    blk = make_block(LOCAL, shift=(2, 16)).float()
    x = tokens(LOCAL, 2, LOCAL[1] + 20, dtype=torch.float32)
    enable_sdpa(blk)
    calls = record_flex_calls(monkeypatch)
    with torch.no_grad():
        ref = blk(x)
        out = compiled(blk)(x)
    assert len(calls) == 1 and torch.allclose(out, ref, rtol=0, atol=1e-4), (out - ref).abs().max()


SWIN = dict(in_channels=4, embed_dim=32, window_size=WINDOW, patch_size=(64, 1), time_frames=128,
            mlp_type="swiglu", swiglu_hidden_ratio=3.0, attention_variant="xsa", drop_path_rate=0.0)


@pytest.fixture(scope="module")
def codec():
    """SAGE's patch, window and collapsed tri2 stage, head size 16 everywhere (flex on in every block)."""
    torch.manual_seed(0)
    enc = SAGEEncoder(depths=[2, 2, 2], num_heads=[2, 4, 8], dimension=32, **SWIN).double().eval()
    dec = SAGEDecoder(channels=16, depths=[2, 2, 2], num_heads=[8, 4, 2], **SWIN).double().eval()
    for model in (enc, dec):
        enable_varlen(model, resolve("tri2"))
    return enc, dec


def test_encoder_and_decoder_compile_with_flex_without_graph_breaks(codec, flex_on_cpu, monkeypatch):
    """Whole encoder and decoder at 4x the training length plus padding: one graph each, FlexAttention in every
    attention, the original output to rounding."""
    enc, dec = codec
    torch.manual_seed(2)
    spec = torch.randn(1, 4, 1025, 520, dtype=torch.float64)
    with torch.no_grad():
        for model in (enc, dec):
            disable_sdpa(model)
        ref_latent, _ = enc(spec)
        ref_spec = dec(ref_latent[:, :16])
        for model in (enc, dec):
            assert enable_sdpa(model) > 0
            assert all(m.flex for m in model.modules() if isinstance(m, WindowAttention))
        calls = record_flex_calls(monkeypatch)
        for model, x in ((enc, spec), (dec, ref_latent[:, :16])):
            explanation = torch._dynamo.explain(model)(x)
            torch._dynamo.reset()
            assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), explanation.break_reasons
        attentions = sum(isinstance(m, WindowAttention) for model in (enc, dec) for m in model.modules())
        assert len(calls) > attentions                                   # multi-phase blocks call it per phase
        latent, _ = compiled(enc)(spec)
        out = compiled(dec)(latent[:, :16])
    assert_close(latent, ref_latent)
    assert_close(out, ref_spec)


# --------------------------------------------------------------------------
# CUDA: the fused kernels under inductor
# --------------------------------------------------------------------------

def snr_db(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (10 * torch.log10(ref.double().pow(2).sum() / (out.double() - ref.double()).pow(2).sum())).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("precision,min_snr", [("highest", 80), ("high", 40)], ids=["fp32", "tf32"])
def test_on_cuda_compiled_flex_matches_eager_sdpa(precision, min_snr, monkeypatch):
    """inductor + FlexAttention kernels against eager SDPA, FP32 without TF32 (rounding only) and with TF32 (both
    sides TF32), on W-MSA, SW-MSA with padding and a multi-phase block; also batch·heads above a (lowered) launch
    cap. The fused kernel takes FP32 or TF32 from the matmul precision at compile time."""
    saved = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        cases = [(make_block(LOCAL, shift=(0, 0)), tokens(LOCAL, 3, LOCAL[1] + 20)),
                 (make_block(LOCAL, shift=(2, 16)), tokens(LOCAL, 3, LOCAL[1] + 20)),
                 (make_block(COLLAPSED), tokens(COLLAPSED, 2, 4 * COLLAPSED[1] + 22))]
        enable_varlen(cases[2][0], resolve("tri2"))
        calls = record_flex_calls(monkeypatch)
        for limit in (S._GRID_LIMIT, 4 * 5):                              # 5 windows per launch: offsets
            monkeypatch.setattr(S, "_GRID_LIMIT", limit)
            for blk, x in cases:
                blk, x = blk.float().cuda(), x.float().cuda()
                enable_sdpa(blk)
                calls.clear()
                with torch.no_grad():
                    ref = blk(x)
                    out = torch.compile(blk, dynamic=False)(x)
                torch._dynamo.reset()
                assert calls and snr_db(out, ref) > min_snr, (limit, snr_db(out, ref))
    finally:
        torch.set_float32_matmul_precision(saved)
        torch._dynamo.reset()
