# ===============================================================
# SAGEEncoder / SAGEDecoder (real-valued): unit tests U1–U12 and integration tests I1–I5.
#
# Markers
#   (none)        fast: individual building blocks, small tensors, < 1 s each
#   @pytest.mark.slow  heavy: full 1024×128 encoder/decoder, may take > 30 s
#
# Run all fast tests:   pytest tests/test_swin_real.py
# Run slow tests too:   pytest tests/test_swin_real.py -m slow
# ===============================================================

import sys
import pytest
import torch
import torch.nn as nn


from sage.model.swin.windowing import window_partition, window_reverse
from sage.model.swin.attention import WindowAttention
from sage.model.swin.block import SwinTransformerBlock
from sage.model.swin.patches import PatchEmbed, PatchMerging, PatchExpand
from sage.model.swin.stage import SwinStage as BasicLayer
from sage.model.encoder import SAGEEncoder
from sage.model.decoder import SAGEDecoder
from sage.model.autoencoder import SAGEAutoencoder
from sage.nn.bottleneck import VAEBottleneck


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B = 2                  # batch size used throughout
LATENT_CH = 64         # latent_channels for Exp 0
DIMENSION = 128        # parameters_to_predict(2) × latent_channels(64)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — instantiated once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def enc_full():
    """Production SAGEEncoder (embed_dim=48).  Slow to construct + run."""
    return SAGEEncoder(
        in_channels=4,
        embed_dim=48,
        depths=[2, 2, 4, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        patch_size=4,
        dimension=DIMENSION,
    ).eval()


@pytest.fixture(scope="module")
def dec_full():
    """Production SAGEDecoder (embed_dim=48).  Slow to construct + run."""
    return SAGEDecoder(
        channels=LATENT_CH,
        in_channels=4,
        embed_dim=48,
        depths=[2, 4, 2, 2],
        num_heads=[24, 12, 6, 3],
        window_size=8,
        patch_size=4,
    ).eval()


@pytest.fixture(scope="module")
def enc_mini():
    """Mini SAGEEncoder (embed_dim=8, 1 block per stage) — fast for integration tests."""
    return SAGEEncoder(
        in_channels=4,
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        window_size=4,
        patch_size=4,
        dimension=16,        # 2 × 8
    ).eval()


@pytest.fixture(scope="module")
def dec_mini():
    """Mini SAGEDecoder (embed_dim=8, 1 block per stage) — fast for integration tests."""
    return SAGEDecoder(
        channels=8,          # latent_channels = embed_dim for mini
        in_channels=4,
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[8, 4, 2, 1],
        window_size=4,
        patch_size=4,
    ).eval()


@pytest.fixture(scope="module")
def ae_mini(enc_mini, dec_mini):
    """SAGEAutoencoder with mini encoder + decoder + real VAEBottleneck."""
    bn = VAEBottleneck()
    return SAGEAutoencoder(encoder=enc_mini, decoder=dec_mini, bottleneck=bn)


# ===========================================================================
# Unit tests  U1 – U12
# ===========================================================================


# ---------------------------------------------------------------------------
# U1 — window_partition → window_reverse roundtrip
# ---------------------------------------------------------------------------

def test_u1_window_partition_roundtrip():
    """window_partition then window_reverse is the identity."""
    B_loc, H, W, C = 2, 32, 32, 48
    ws = 8
    x = torch.randn(B_loc, H, W, C)
    windows = window_partition(x, ws)               # (B*nW, ws, ws, C)
    x_rec   = window_reverse(windows, ws, H, W)     # (B, H, W, C)
    assert x_rec.shape == x.shape, f"Shape mismatch: {x_rec.shape} vs {x.shape}"
    assert torch.allclose(x_rec, x, atol=1e-6), "Roundtrip must be bit-exact identity"


# ---------------------------------------------------------------------------
# U2 — PatchEmbed output shape
# ---------------------------------------------------------------------------

def test_u2_patch_embed_shape():
    """PatchEmbed(4→48, k=4, s=4): (B, 4, 1024, 128) → (B, 8192, 48)."""
    embed = PatchEmbed(
        img_size=(1024, 128), patch_size=4, in_chans=4,
        embed_dim=48, norm_layer=nn.LayerNorm,
    ).eval()
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        out = embed(x)
    assert out.shape == (B, 8192, 48), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# U3 — WindowAttention: shape preserved + no NaN
# ---------------------------------------------------------------------------

def test_u3_window_attention_shape():
    """WindowAttention output matches input shape and has no NaN."""
    ws, dim, nheads = 8, 48, 3
    nW = 16   # arbitrary number of windows
    attn = WindowAttention(
        dim=dim, window_size=(ws, ws), num_heads=nheads,
        qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
        pretrained_window_size=(0, 0),
    ).eval()
    x = torch.randn(nW, ws * ws, dim)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape, f"Shape: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "NaN in WindowAttention output"


# ---------------------------------------------------------------------------
# U4 — WindowAttention: gradient flow
# ---------------------------------------------------------------------------

def test_u4_window_attention_gradients():
    """All WindowAttention parameters receive non-None gradients after backward."""
    ws, dim, nheads = 8, 48, 3
    nW = 16
    attn = WindowAttention(
        dim=dim, window_size=(ws, ws), num_heads=nheads,
        qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
        pretrained_window_size=(0, 0),
    )
    x = torch.randn(nW, ws * ws, dim, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"


# ---------------------------------------------------------------------------
# U5 — SwinTransformerBlock: shape preserved
# ---------------------------------------------------------------------------

def test_u5_swin_block_shape():
    """SwinTransformerBlock output has same shape as input."""
    H, W, C = 32, 32, 48
    block = SwinTransformerBlock(
        dim=C, input_resolution=(H, W), num_heads=3,
        window_size=8, shift_size=0, mlp_ratio=4.0,
    ).eval()
    x = torch.randn(B, H * W, C)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape, f"Shape: {out.shape} vs {x.shape}"


# ---------------------------------------------------------------------------
# U6 — SwinTransformerBlock: gradient flow
# ---------------------------------------------------------------------------

def test_u6_swin_block_gradients():
    """No SwinTransformerBlock parameter has None gradient after backward."""
    H, W, C = 32, 32, 48
    block = SwinTransformerBlock(
        dim=C, input_resolution=(H, W), num_heads=3,
        window_size=8, shift_size=0, mlp_ratio=4.0,
    )
    x = torch.randn(B, H * W, C, requires_grad=True)
    out = block(x)
    out.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"


def test_fused_kernels_are_skipped_for_cpu_tensors(monkeypatch):
    """A block built where the fused CUDA kernels are available (GPU machine, extension
    installed) still runs on CPU tensors, through the plain roll + partition path."""
    import sage.model.swin.block as block_mod

    def cuda_only(*args, **kwargs):
        raise RuntimeError("input must be a CUDA tensor")

    kernel = type("CudaOnlyKernel", (), {"apply": staticmethod(cuda_only)})
    monkeypatch.setattr(block_mod, "FUSED_WINDOW_AVAILABLE", True)
    monkeypatch.setattr(block_mod, "WindowProcess", kernel)
    monkeypatch.setattr(block_mod, "WindowProcessReverse", kernel)
    H, W, C = 16, 16, 48
    kw = dict(dim=C, input_resolution=(H, W), num_heads=3, window_size=8, shift_size=4)
    fused = SwinTransformerBlock(**kw, fused_window_process=True).eval()
    plain = SwinTransformerBlock(**kw, fused_window_process=False).eval()
    plain.load_state_dict(fused.state_dict())
    assert fused.fused_window_process and not plain.fused_window_process
    x = torch.randn(B, H * W, C)
    with torch.no_grad():
        assert torch.equal(fused(x), plain(x))


def _fused_available() -> bool:
    from sage.model.swin.cuda_kernels import FUSED_WINDOW_AVAILABLE
    return FUSED_WINDOW_AVAILABLE


def _recording_kernels(calls: list):
    """Stand-ins for the fused CUDA window kernels: record each call, compute roll + partition."""
    def partition(x, B, H, W, C, shift_h, shift_w, wh, ww):
        calls.append("partition")
        return window_partition(torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2)), (wh, ww))

    def reverse(windows, B, H, W, C, shift_h, shift_w, wh, ww):
        calls.append("reverse")
        return torch.roll(window_reverse(windows, (wh, ww), H, W), shifts=(shift_h, shift_w), dims=(1, 2))

    return (type("Partition", (), {"apply": staticmethod(partition)}),
            type("Reverse", (), {"apply": staticmethod(reverse)}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused path runs on CUDA tensors only")
def test_fused_kernels_stay_out_of_compiled_graphs(monkeypatch):
    """torch.compile cannot trace the fused kernels (one graph break per call): a compiled graph takes
    the plain roll + partition path. Eager calls keep using the kernels."""
    import sage.model.swin.block as block_mod

    calls: list = []
    partition, reverse = _recording_kernels(calls)
    monkeypatch.setattr(block_mod, "FUSED_WINDOW_AVAILABLE", True)
    monkeypatch.setattr(block_mod, "WindowProcess", partition)
    monkeypatch.setattr(block_mod, "WindowProcessReverse", reverse)
    H, W, C = 16, 16, 48
    blk = SwinTransformerBlock(dim=C, input_resolution=(H, W), num_heads=3, window_size=8, shift_size=4,
                               fused_window_process=True).eval().cuda()
    x = torch.randn(B, H * W, C, device="cuda")
    with torch.no_grad():
        ref = blk(x)
        assert calls == ["partition", "reverse"]
        calls.clear()
        out = torch.compile(blk, backend="aot_eager", dynamic=False)(x)
    torch._dynamo.reset()
    assert calls == []
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares CPU and CUDA")
def test_per_grid_constants_are_the_same_on_cpu_and_cuda():
    """The shift masks and the multi-phase weights are built on the activations' device: same values."""
    from sage.model.swin.block import shift_mask
    from sage.model.swin.varlen import phase_weights, resolve
    for grid, window, shift in [((8, 32), (4, 8), (2, 4)), ((4, 96), (4, 32), (0, 16)), ((16, 40), (8, 8), (4, 4))]:
        cpu = shift_mask(*grid, window, shift, torch.device("cpu"))
        assert torch.equal(shift_mask(*grid, window, shift, torch.device("cuda")).cpu(), cpu)
    for variant in ("mean2", "tri2", "hard2", "mean4", "tri4", "hard4"):
        for time_tokens in (33, 160, 2587):
            cpu = phase_weights(time_tokens, 32, resolve(variant))
            assert torch.equal(phase_weights(time_tokens, 32, resolve(variant), torch.device("cuda")).cpu(), cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled graph builds the constants on the GPU")
def test_compiled_block_on_cuda_before_any_eager_call_has_no_graph_break():
    H, W, C = 16, 16, 48
    blk = SwinTransformerBlock(dim=C, input_resolution=(H, W), num_heads=3, window_size=8, shift_size=4,
                               fused_window_process=True).eval().cuda()
    x = torch.randn(B, H * W, C, device="cuda")
    with torch.no_grad():
        explanation = torch._dynamo.explain(blk)(x)
        out = torch.compile(blk, backend="aot_eager", dynamic=False)(x)
        torch._dynamo.reset()
        assert not blk._attn_mask_cache
        ref = blk(x)
    assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), explanation.break_reasons
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not (torch.cuda.is_available() and _fused_available()),
                    reason="needs the compiled swin_window_process extension and a GPU")
def test_compiled_block_with_real_fused_kernels_has_no_graph_break():
    H, W, C = 16, 16, 48
    blk = SwinTransformerBlock(dim=C, input_resolution=(H, W), num_heads=3, window_size=8, shift_size=4,
                               fused_window_process=True).eval().cuda()
    assert blk.fused_window_process
    x = torch.randn(B, H * W, C, device="cuda")
    with torch.no_grad():
        ref = blk(x)                                       # eager: the fused kernels
        explanation = torch._dynamo.explain(blk)(x)
        out = torch.compile(blk, backend="aot_eager", dynamic=False)(x)
    torch._dynamo.reset()
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# U7 — BasicLayer without PatchMerging: shape preserved
# ---------------------------------------------------------------------------

def test_u7_basic_layer_no_downsample():
    """BasicLayer (downsample=None) preserves token count and channel dim."""
    H, W, C = 64, 8, 192
    layer = BasicLayer(
        dim=C, input_resolution=(H, W), depth=2, num_heads=12,
        window_size=8, downsample=None,
    ).eval()
    x = torch.randn(B, H * W, C)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (B, H * W, C), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# U8 — BasicLayer with PatchMerging: spatial halved, channels doubled
# ---------------------------------------------------------------------------

def test_u8_basic_layer_with_patch_merging():
    """BasicLayer + PatchMerging: (B, H*W, C) → (B, H/2*W/2, 2C)."""
    H, W, C = 256, 32, 48
    layer = BasicLayer(
        dim=C, input_resolution=(H, W), depth=2, num_heads=3,
        window_size=8, downsample=PatchMerging,
    ).eval()
    x = torch.randn(B, H * W, C)
    with torch.no_grad():
        out = layer(x)
    expected = (B, (H // 2) * (W // 2), 2 * C)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# U9 — PatchExpand: (B, H*W, 2C) → (B, 4H*W, C)
# ---------------------------------------------------------------------------

def test_u9_patch_expand_shape():
    """PatchExpand doubles spatial dims and halves channels."""
    H, W = 32, 4
    dim_in = 384   # 2C = 384, C = 192
    expand = PatchExpand(input_resolution=(H, W), dim=dim_in).eval()
    x = torch.randn(B, H * W, dim_in)
    with torch.no_grad():
        out = expand(x)
    assert out.shape == (B, 4 * H * W, dim_in // 2), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# U10 — PatchMerging → PatchExpand: grid shape restored
# ---------------------------------------------------------------------------

def test_u10_merge_expand_grid_roundtrip():
    """After PatchMerging halves the grid, PatchExpand restores it."""
    H, W, C = 128, 16, 96      # a mid-encoder resolution / channel width
    merge = PatchMerging(input_resolution=(H, W), dim=C).eval()
    # After merge: grid (H/2, W/2), channels 2C
    x_pre = torch.randn(B, H * W, C)
    with torch.no_grad():
        x_merged = merge(x_pre)
    assert x_merged.shape == (B, (H // 2) * (W // 2), 2 * C), (
        f"PatchMerging shape wrong: {x_merged.shape}"
    )

    expand = PatchExpand(input_resolution=(H // 2, W // 2), dim=2 * C).eval()
    with torch.no_grad():
        x_expanded = expand(x_merged)
    assert x_expanded.shape == (B, H * W, C), (
        f"PatchExpand did not restore grid: {x_expanded.shape}"
    )


# ---------------------------------------------------------------------------
# U11 — Full encoder: (B, 4, 1024, 128) → (B, 128, 128)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_u11_encoder_shape(enc_full):
    """SAGEEncoder maps (B, 4, 1024, 128) → (B, 128, 128) with feature_shape info."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        latents, info = enc_full(x)
    assert latents.shape == (B, DIMENSION, 128), f"Got {latents.shape}"
    assert info.get("feature_shape") == (32, 4), f"feature_shape: {info.get('feature_shape')}"
    assert not torch.isnan(latents).any(), "NaN in encoder output"


# ---------------------------------------------------------------------------
# U12 — Full decoder: (B, 64, 128) → (B, 4, 1024, 128)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_u12_decoder_shape(dec_full):
    """SAGEDecoder maps (B, 64, 128) → (B, 4, 1024, 128)."""
    z = torch.randn(B, LATENT_CH, 128)
    with torch.no_grad():
        out = dec_full(z)
    assert out.shape == (B, 4, 1024, 128), f"Got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in decoder output"


# ===========================================================================
# Integration tests  I1 – I5
# (Use mini encoder/decoder — embed_dim=8, depths=[1,1,1,1] — for speed)
# ===========================================================================

# Mini-model expected shapes
_MINI_DIM = 16       # dimension = 2 × 8
_MINI_LC  = 8        # latent_channels = embed_dim


# ---------------------------------------------------------------------------
# I1 — SAGEAutoencoder full forward: shapes correct, no NaN
# ---------------------------------------------------------------------------

def test_i1_autoencoder_forward(ae_mini, enc_mini):
    """encode → bottleneck → decode: shapes correct, output has no NaN."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        latents, enc_info, bn_info = ae_mini.encode(x, return_info=True)
        recon = ae_mini.decode(latents, encoder_info=enc_info)

    # After bottleneck, sequence unchanged; channels halved (ptp=2 → /2)
    assert latents.shape == (B, _MINI_LC, 128), (
        f"Latent shape wrong: {latents.shape}"
    )
    assert recon.shape == (B, 4, 1024, 128), f"Recon shape wrong: {recon.shape}"
    assert not torch.isnan(recon).any(), "NaN in reconstruction"
    assert "kl" in bn_info, "bottleneck info must contain 'kl'"


# ---------------------------------------------------------------------------
# I2 — Gradient flow end-to-end
# ---------------------------------------------------------------------------

def test_i2_gradient_flow():
    """loss.backward() propagates non-None gradients to every encoder+decoder param."""
    enc = SAGEEncoder(
        in_channels=4, embed_dim=8, depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8], window_size=4, patch_size=4, dimension=16,
    )
    dec = SAGEDecoder(
        channels=8, in_channels=4, embed_dim=8,
        depths=[1, 1, 1, 1], num_heads=[8, 4, 2, 1],
        window_size=4, patch_size=4,
    )
    ae = SAGEAutoencoder(encoder=enc, decoder=dec, bottleneck=VAEBottleneck())

    x = torch.randn(B, 4, 1024, 128)
    latents, enc_info, bn_info = ae.encode(x, return_info=True)
    recon = ae.decode(latents, encoder_info=enc_info)
    loss = recon.sum() + bn_info["kl"]
    loss.backward()

    for name, param in ae.named_parameters():
        assert param.grad is not None, f"No gradient for param '{name}'"


# ---------------------------------------------------------------------------
# I3 — KL sanity: finite and non-negative at random init
# ---------------------------------------------------------------------------

def test_i3_kl_sanity(ae_mini):
    """KL from the bottleneck is finite and non-negative at random init."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        _, _, bn_info = ae_mini.encode(x, return_info=True)
    kl = bn_info["kl"].item()
    assert kl == kl,               "KL is NaN"
    assert kl != float("inf"),     "KL is +Inf"
    assert kl >= 0.0,              f"KL must be ≥ 0, got {kl:.6f}"


# ---------------------------------------------------------------------------
# I4 — Reconstruction loss: finite at random init
# ---------------------------------------------------------------------------

def test_i4_recon_loss_finite(ae_mini):
    """MSE reconstruction loss is finite on a random-init forward pass."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        latents, enc_info, _ = ae_mini.encode(x, return_info=True)
        recon = ae_mini.decode(latents, encoder_info=enc_info)
    loss = torch.nn.functional.mse_loss(recon, x)
    assert torch.isfinite(loss), f"MSE loss not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# I5 — Freq crop: encoder accepts 1025 bins (raw STFT) and crops to 1024
# ---------------------------------------------------------------------------

def test_i5_freq_crop(enc_mini):
    """Encoder accepts (B, 4, 1025, 128) — raw CAC STFT with Nyquist bin — and crops internally."""
    x = torch.randn(B, 4, 1025, 128)    # raw CAC STFT output (cac=true)
    with torch.no_grad():
        latents, info = enc_mini(x)
    assert latents.shape == (B, _MINI_DIM, 128), (
        f"Expected (B,{_MINI_DIM},128) after freq crop, got {latents.shape}"
    )
    assert not torch.isnan(latents).any(), "NaN in latents after freq crop"
