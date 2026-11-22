# ===============================================================
# Tests for the exact inference speed-ups that come with the SDPA path:
#   fused_bias     Q/V bias added by the QKV GEMM            (sage/model/swin/sdpa.py)
#   shared_phases  one QKV / output projection for all phases
#                  of a multi-phase block                    (sage/model/swin/block.py)
#   patch_linear   PatchEmbed / PatchUnembed as one GEMM     (sage/model/swin/patches.py)
# Guards: off by default, set by enable_sdpa / enable_patch_linear and cleared by
# them; in float64 each one alone, and all together on an encoder + decoder, equals
# the original to rounding (W-MSA, SW-MSA, baseline/XSA, batch 1/3, padded grids,
# every combination mode of the multi-phase blocks); bf16 autocast keeps the
# original's dtypes; training and autograd keep the original attention; complex
# layers are never touched; the state_dict does not change; no graph break under
# torch.compile.
# ===============================================================

import pytest
import torch
import torch.nn as nn

from sage.model.decoder import SAGEDecoder
from sage.model.encoder import SAGEEncoder
from sage.model.swin.attention import WindowAttention
from sage.model.swin.block import SwinTransformerBlock
from sage.model.swin.patches import PatchEmbed, PatchUnembed, enable_patch_linear
from sage.model.swin.sdpa import disable_sdpa, enable_sdpa, fusable_qkv_bias, sdpa_qkv
from sage.model.swin.varlen import enable_varlen, load_presets, phase_weights, resolve
from sage.nn.conv.normed import NormLinear

DIM = 64
WINDOW = (4, 32)
LOCAL = (16, 128)      # window smaller than the grid: W-MSA / SW-MSA with a shift mask
COLLAPSED = (4, 32)    # SAGE's deepest stage: one window covers the training grid
ATOL = 1e-12           # float64: the reformulations differ from the original only by rounding (~1e-14 measured)
PATCHES = [(64, 1), (4, 4), (4, 2)]   # SAGE's (freq, time) patch first


def randomize_qv_bias(module: nn.Module) -> None:
    """Non-zero Q/V biases (they start at 0), so the bias terms and the padding tokens are tested too."""
    with torch.no_grad():
        for attn in module.modules():
            if isinstance(attn, WindowAttention) and attn.q_bias is not None:
                attn.q_bias.normal_(0.0, 0.5)
                attn.v_bias.normal_(0.0, 0.5)


def make_block(resolution, shift=(0, 0), variant="xsa", qkv_bias=True, seed=0) -> SwinTransformerBlock:
    torch.manual_seed(seed)
    blk = SwinTransformerBlock(
        dim=DIM, input_resolution=resolution, num_heads=4,
        window_size=WINDOW, shift_size=shift, mlp_type="swiglu", qkv_bias=qkv_bias,
        swiglu_hidden_ratio=3.0, attention_variant=variant, norm_placement="res_post",
    )
    randomize_qv_bias(blk)
    return blk.eval().double()


def tokens(resolution, batch: int = 1, time_tokens=None) -> torch.Tensor:
    torch.manual_seed(1)
    width = resolution[1] if time_tokens is None else time_tokens
    return torch.randn(batch, resolution[0] * width, DIM, dtype=torch.float64)


def assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.allclose(out, ref, rtol=0.0, atol=ATOL), (out - ref).abs().max()


def snr_db(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (10 * torch.log10(ref.double().pow(2).sum() / (out.double() - ref.double()).pow(2).sum())).item()


# --------------------------------------------------------------------------
# Switches: off by default, set and cleared by enable_sdpa / enable_patch_linear
# --------------------------------------------------------------------------

def test_off_by_default():
    blk = make_block(COLLAPSED)
    pe = PatchEmbed(img_size=(256, 32), patch_size=(64, 1), in_chans=4, embed_dim=8)
    pu = PatchUnembed(input_resolution=(4, 32), embed_dim=8, out_channels=4, patch_size=(64, 1))
    assert not (blk.attn.fused_bias or blk.shared_phases or pe.patch_linear or pu.patch_linear)


def test_enable_sdpa_sets_and_clears_the_options():
    blk = make_block(COLLAPSED)
    enable_sdpa(blk)
    assert blk.attn.sdpa and blk.attn.fused_bias and blk.shared_phases
    enable_sdpa(blk, fused_bias=False, shared_phases=False)
    assert blk.attn.sdpa and not blk.attn.fused_bias and not blk.shared_phases
    enable_sdpa(blk)
    disable_sdpa(blk)
    assert not (blk.attn.sdpa or blk.attn.fused_bias or blk.shared_phases)


def test_fused_bias_only_on_a_plain_bias_free_qkv_linear():
    assert fusable_qkv_bias(make_block(LOCAL).attn)
    no_bias = make_block(LOCAL, qkv_bias=False)
    assert not fusable_qkv_bias(no_bias.attn)
    enable_sdpa(no_bias)
    assert no_bias.attn.sdpa and not no_bias.attn.fused_bias
    normed = make_block(LOCAL).attn
    normed.qkv = NormLinear(DIM, 3 * DIM, bias=False, norm="weight_norm").double()
    assert not fusable_qkv_bias(normed)                   # the bias would have to follow the norm
    with_bias = make_block(LOCAL).attn
    with_bias.qkv = NormLinear(DIM, 3 * DIM, bias=True).double()
    assert not fusable_qkv_bias(with_bias)


def test_enable_patch_linear_sets_and_clears_the_flag():
    pe = PatchEmbed(img_size=(256, 32), patch_size=(64, 1), in_chans=4, embed_dim=8)
    pu = PatchUnembed(input_resolution=(4, 32), embed_dim=8, out_channels=4, patch_size=(64, 1))
    model = nn.Sequential(pe, pu)
    assert enable_patch_linear(model) == 2 and pe.patch_linear and pu.patch_linear
    assert enable_patch_linear(model, False) == 2 and not (pe.patch_linear or pu.patch_linear)


# --------------------------------------------------------------------------
# fused_bias: same formula, the bias added by the GEMM
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(0, 0), (2, 16)], ids=["w-msa", "sw-msa"])
@pytest.mark.parametrize("variant", ["baseline", "xsa"])
@pytest.mark.parametrize("batch", [1, 3])
def test_fused_bias_matches_the_original_in_float64(shift, variant, batch):
    blk, x = make_block(LOCAL, shift=shift, variant=variant), tokens(LOCAL, batch)
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk, fused_bias=True, shared_phases=False)
        assert blk.attn.fused_bias
        out = blk(x)
    assert_close(out, ref)


def test_fused_bias_matches_the_original_on_a_padded_grid():
    """Time tokens not a multiple of the window: the padding tokens' qkv is the bias alone."""
    blk, x = make_block(LOCAL, shift=(2, 16)), tokens(LOCAL, 2, time_tokens=LOCAL[1] + 20)
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk, shared_phases=False)
        out = blk(x)
    assert_close(out, ref)


def test_fused_bias_keeps_the_original_dtypes_under_bf16_autocast():
    """q/k/v in FP32 as in the original (bf16 GEMM + FP32 bias), attention in bf16; one bf16 rounding apart."""
    attn = make_block(LOCAL).attn.float()
    torch.manual_seed(3)
    x = torch.randn(48, 128, DIM)
    bias = torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias)).detach()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        enable_sdpa(attn, fused_bias=False)
        ref, ref_dt = sdpa_qkv(attn, x)
        enable_sdpa(attn)
        out, dt = sdpa_qkv(attn, x)
    assert (out.dtype, dt) == (ref.dtype, ref_dt) == (torch.float32, torch.bfloat16)
    assert ((out - ref).abs() <= 2.0 ** -6 * (ref.abs() + bias.abs())).all()


# --------------------------------------------------------------------------
# shared_phases: one QKV / output projection for all phases of a multi-phase block
# --------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["tri2", "mean2", "hard2", "tri4"])
@pytest.mark.parametrize("variant", ["baseline", "xsa"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("time_tokens", [4 * COLLAPSED[1], 4 * COLLAPSED[1] + 22], ids=["4x", "padded"])
def test_shared_phases_match_the_original_in_float64(preset, variant, batch, time_tokens):
    blk, x = make_block(COLLAPSED, variant=variant), tokens(COLLAPSED, batch, time_tokens)
    enable_varlen(blk, resolve(preset))
    with torch.no_grad():
        ref = blk(x)
        enable_sdpa(blk, fused_bias=False, shared_phases=True)
        out = blk(x)
    assert_close(out, ref)


def count_shared_calls(monkeypatch) -> list:
    calls, shared = [], SwinTransformerBlock._shared_phase_attention

    def spy(self, x, weights):
        calls.append(x.shape)
        return shared(self, x, weights)

    monkeypatch.setattr(SwinTransformerBlock, "_shared_phase_attention", spy)
    return calls


def test_shared_phases_run_only_on_the_inference_sdpa_path(monkeypatch):
    blk, x = make_block(COLLAPSED), tokens(COLLAPSED, time_tokens=4 * COLLAPSED[1])
    enable_varlen(blk, resolve("tri2"))
    calls = count_shared_calls(monkeypatch)
    enable_sdpa(blk)
    with torch.no_grad():
        blk(x)
        blk(tokens(COLLAPSED))                             # training length: single phase, nothing to share
    assert len(calls) == 1
    blk(x)                                                 # autograd on
    blk.train()
    with torch.no_grad():
        blk(x)
    blk.eval()
    enable_sdpa(blk, shared_phases=False)
    with torch.no_grad():
        blk(x)
    disable_sdpa(blk)
    blk.shared_phases = True                               # without the SDPA path the option is inert
    with torch.no_grad():
        blk(x)
    assert len(calls) == 1


def test_shared_phases_keep_the_output_dtype_under_bf16_autocast():
    blk = make_block(COLLAPSED).float()
    enable_varlen(blk, resolve("tri2"))
    x = tokens(COLLAPSED, 2, time_tokens=4 * COLLAPSED[1] + 22).float()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        enable_sdpa(blk, shared_phases=False)
        ref = blk._multiphase_attention(x)
        enable_sdpa(blk)
        out = blk._multiphase_attention(x)
    assert out.dtype == ref.dtype == torch.float32
    assert snr_db(out, ref) > 35                           # bf16 rounding apart (~46 dB measured)


def test_phase_weights_in_float64_sum_to_one_and_round_to_the_float32_weights():
    for name in load_presets():
        cfg = resolve(name)
        for time_tokens in range(1, 260):
            w32 = phase_weights(time_tokens, 32, cfg)
            w64 = phase_weights(time_tokens, 32, cfg, dtype=torch.float64)
            assert w32.dtype == torch.float32 and w64.dtype == torch.float64
            assert (w64.sum(dim=0) - 1).abs().max() <= 4 * torch.finfo(torch.float64).eps, (name, time_tokens)
            assert torch.equal(w64.float(), w32), (name, time_tokens)
            assert torch.equal(phase_weights(time_tokens, 32, cfg, dtype=torch.bfloat16), w32.bfloat16())


# --------------------------------------------------------------------------
# patch_linear: the patch convolutions as one GEMM
# --------------------------------------------------------------------------

def make_patch_embed(patch, img_h=256, norm=False) -> PatchEmbed:
    torch.manual_seed(0)
    pe = PatchEmbed(img_size=(img_h, 32), patch_size=patch, in_chans=4, embed_dim=48,
                    norm_layer=nn.LayerNorm if norm else None)
    return pe.double().eval()


def make_patch_unembed(patch, bias=True) -> PatchUnembed:
    torch.manual_seed(0)
    pu = PatchUnembed(input_resolution=(4, 32), embed_dim=48, out_channels=4, patch_size=patch)
    if not bias:
        pu.conv.register_parameter("bias", None)
    return pu.double().eval()


@pytest.mark.parametrize("patch", PATCHES)
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("img_h,norm", [(256, False), (256, True), (250, False)],
                         ids=["plain", "norm", "partial-row"])
def test_patch_embed_as_gemm_matches_the_conv_in_float64(patch, batch, img_h, norm):
    """Longer than the training length; img_h=250 leaves a partial row of patches, which the conv drops."""
    pe = make_patch_embed(patch, img_h, norm)
    torch.manual_seed(1)
    x = torch.randn(batch, 4, img_h, 96, dtype=torch.float64)
    with torch.no_grad():
        ref = pe(x)
        assert enable_patch_linear(pe) == 1
        out = pe(x)
    assert_close(out, ref)


@pytest.mark.parametrize("patch", PATCHES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch", [1, 3])
def test_patch_unembed_as_gemm_matches_the_conv_in_float64(patch, bias, batch):
    pu = make_patch_unembed(patch, bias)
    torch.manual_seed(1)
    x = torch.randn(batch, 4 * 96, 48, dtype=torch.float64)
    with torch.no_grad():
        ref = pu(x)
        assert enable_patch_linear(pu) == 1
        out = pu(x)
    assert_close(out, ref)
    assert out.is_contiguous()                             # NCHW, as the conv returns it


def test_patch_layers_as_gemm_have_the_conv_gradients():
    """Differentiable reformulation: same gradients for weights, biases and input (training may use it)."""
    pe, pu = make_patch_embed((64, 1)), make_patch_unembed((64, 1))
    model = nn.Sequential(pe, pu)
    torch.manual_seed(1)
    x = torch.randn(2, 4, 256, 96, dtype=torch.float64)
    grads = []
    for enabled in (False, True):
        enable_patch_linear(model, enabled)
        model.zero_grad()
        xi = x.clone().requires_grad_(True)
        model(xi).pow(2).sum().backward()
        grads.append([xi.grad] + [p.grad for p in model.parameters()])
    for ref, out in zip(*grads):
        assert_close(out, ref)


# --------------------------------------------------------------------------
# All together: encoder + decoder of SAGE's layout, variable length
# --------------------------------------------------------------------------

SWIN = dict(in_channels=4, embed_dim=16, window_size=WINDOW, patch_size=(64, 1), time_frames=128,
            mlp_type="swiglu", swiglu_hidden_ratio=3.0, attention_variant="xsa", drop_path_rate=0.0)


@pytest.fixture(scope="module")
def codec():
    """Encoder and decoder with SAGE's patch, window and collapsed last stage (multi-phase tri2), small widths."""
    torch.manual_seed(0)
    enc = SAGEEncoder(depths=[2, 2, 2], num_heads=[2, 4, 8], dimension=32, **SWIN).double().eval()
    dec = SAGEDecoder(channels=16, depths=[2, 2, 2], num_heads=[8, 4, 2], **SWIN).double().eval()
    for model in (enc, dec):
        randomize_qv_bias(model)
        assert enable_varlen(model, resolve("tri2")) == 2
    return enc, dec


def set_fast(enc, dec, fast: bool) -> None:
    for model in (enc, dec):
        enable_sdpa(model, fast)
        enable_patch_linear(model, fast)


def run_codec(enc, dec, spec, fast: bool):
    set_fast(enc, dec, fast)
    with torch.no_grad():
        latent, _ = enc(spec)                              # (B, 32, T)
        return latent, dec(latent[:, :16])                 # (B, 4, 1024, T)


@pytest.mark.parametrize("frames", [128, 512, 520], ids=["train-len", "4x", "padded"])
def test_encoder_and_decoder_match_the_original_in_float64(codec, frames):
    enc, dec = codec
    torch.manual_seed(2)
    spec = torch.randn(2, 4, 1025, frames, dtype=torch.float64)
    ref_latent, ref_spec = run_codec(enc, dec, spec, fast=False)
    latent, out = run_codec(enc, dec, spec, fast=True)
    assert enc.patch_embed.patch_linear and dec.patch_unembed.patch_linear
    assert_close(latent, ref_latent)
    assert_close(out, ref_spec)


def test_encoder_and_decoder_compile_without_graph_breaks_and_match(codec):
    enc, dec = codec
    torch.manual_seed(2)
    spec = torch.randn(1, 4, 1025, 520, dtype=torch.float64)
    ref_latent, ref_spec = run_codec(enc, dec, spec, fast=False)
    set_fast(enc, dec, True)
    with torch.no_grad():
        for model, x in ((enc, spec), (dec, ref_latent[:, :16])):
            explanation = torch._dynamo.explain(model)(x)
            torch._dynamo.reset()
            assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), explanation.break_reasons
        latent, _ = torch.compile(enc, backend="aot_eager", dynamic=False)(spec)
        out = torch.compile(dec, backend="aot_eager", dynamic=False)(latent[:, :16])
    torch._dynamo.reset()
    assert_close(latent, ref_latent)
    assert_close(out, ref_spec)


def test_state_dict_is_unchanged(codec):
    enc, dec = codec
    before = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in (("enc", enc), ("dec", dec))}
    run_codec(enc, dec, torch.randn(1, 4, 1025, 520, dtype=torch.float64), fast=True)
    for name, model in (("enc", enc), ("dec", dec)):
        state = model.state_dict()
        assert state.keys() == before[name].keys()
        assert all(torch.equal(state[k], v) for k, v in before[name].items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_on_cuda_match_the_original():
    """CUDA kernels: FP64 (math SDPA, cuBLAS) to rounding; FP32 (fused SDPA kernels, cuBLAS bias epilogue,
    fused window kernels when built) and bf16 autocast (same dtypes) up to their rounding."""
    torch.manual_seed(0)
    enc = SAGEEncoder(depths=[2, 2, 2], num_heads=[2, 4, 8], dimension=32, fused_window_process=True, **SWIN)
    dec = SAGEDecoder(channels=16, depths=[2, 2, 2], num_heads=[8, 4, 2], **SWIN)
    for model in (enc, dec):
        randomize_qv_bias(model)
        enable_varlen(model.cuda().eval(), resolve("tri2"))
    spec = torch.randn(2, 4, 1025, 520, device="cuda")
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")                 # no TF32: FP32 rounding only
        for dtype, autocast in ((torch.float64, False), (torch.float32, False), (torch.float32, True)):
            enc.to(dtype), dec.to(dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                ref, out = (run_codec(enc, dec, spec.to(dtype), fast) for fast in (False, True))
            for o, r in zip(out, ref):
                assert o.dtype == r.dtype
                if dtype == torch.float64:
                    assert_close(o, r)
                else:
                    assert snr_db(o, r) > (25 if autocast else 80), (dtype, autocast, snr_db(o, r))
    finally:
        torch.set_float32_matmul_precision(precision)


# --------------------------------------------------------------------------
# Scope: complex-valued layers keep their own path
# --------------------------------------------------------------------------

def test_complex_patch_layers_are_never_touched():
    pe = PatchEmbed(img_size=(256, 32), patch_size=(64, 1), in_chans=4, embed_dim=8, is_complex=True)
    pu = PatchUnembed(input_resolution=(4, 32), embed_dim=8, out_channels=4, patch_size=(64, 1), is_complex=True)
    assert enable_patch_linear(nn.Sequential(pe, pu)) == 0
    assert not (pe.patch_linear or pu.patch_linear)


def test_complex_blocks_are_never_touched():
    pytest.importorskip("complextorch", reason="extra [complex] not installed")
    blk = SwinTransformerBlock(dim=8, input_resolution=COLLAPSED, num_heads=2, window_size=WINDOW, is_complex=True)
    assert enable_sdpa(blk) == 0
    assert not (blk.attn.sdpa or blk.attn.fused_bias or blk.shared_phases)
