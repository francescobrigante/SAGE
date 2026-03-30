# ===============================================================
# Phase 3 test suite — SwinEncoder / SwinDecoder (Exp 0, real-valued).
# Covers U1–U12 (unit) and I1–I5 (integration) from EXPERIMENTS.md §5.
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

_SRC = "/Users/francesco/Desktop/C-VAE/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from c_vae.swin.swin_transformer_v2 import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
    SwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_reverse,
)
from c_vae.swin.encoder import SwinEncoder
from c_vae.swin.decoder import SwinDecoder, PatchExpand
from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.models.bottlenecks import VAEBottleneck


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
    """Production SwinEncoder (embed_dim=48).  Slow to construct + run."""
    return SwinEncoder(
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
    """Production SwinDecoder (embed_dim=48).  Slow to construct + run."""
    return SwinDecoder(
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
    """Mini SwinEncoder (embed_dim=8, 1 block per stage) — fast for integration tests."""
    return SwinEncoder(
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
    """Mini SwinDecoder (embed_dim=8, 1 block per stage) — fast for integration tests."""
    return SwinDecoder(
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
    """AutoEncoder with mini encoder + decoder + real VAEBottleneck."""
    bn = VAEBottleneck()
    return AutoEncoder(encoder=enc_mini, decoder=dec_mini, bottleneck=bn)


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
# U11 — Full encoder: (B, 2, 1024, 128) → (B, 128, 32, 4)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_u11_encoder_shape(enc_full):
    """SwinEncoder maps (B, 4, 1024, 128) → (B, 128, 32, 4) with feature_shape info."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        latents, info = enc_full(x)
    assert latents.shape == (B, DIMENSION, 32, 4), f"Got {latents.shape}"
    assert info.get("feature_shape") == (32, 4), f"feature_shape: {info.get('feature_shape')}"
    assert not torch.isnan(latents).any(), "NaN in encoder output"


# ---------------------------------------------------------------------------
# U12 — Full decoder: (B, 64, 32, 4) → (B, 2, 1024, 128)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_u12_decoder_shape(dec_full):
    """SwinDecoder maps (B, 64, 32, 4) → (B, 4, 1024, 128)."""
    z = torch.randn(B, LATENT_CH, 32, 4)
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
# I1 — AutoEncoder full forward: shapes correct, no NaN
# ---------------------------------------------------------------------------

def test_i1_autoencoder_forward(ae_mini, enc_mini):
    """encode → bottleneck → decode: shapes correct, output has no NaN."""
    x = torch.randn(B, 4, 1024, 128)
    with torch.no_grad():
        latents, enc_info, bn_info = ae_mini.encode(x, return_info=True)
        recon = ae_mini.decode(latents, encoder_info=enc_info)

    # After bottleneck, spatial grid unchanged; channels halved (ptp=2 → /2)
    assert latents.shape == (B, _MINI_LC, 32, 4), (
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
    enc = SwinEncoder(
        in_channels=4, embed_dim=8, depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8], window_size=4, patch_size=4, dimension=16,
    )
    dec = SwinDecoder(
        channels=8, in_channels=4, embed_dim=8,
        depths=[1, 1, 1, 1], num_heads=[8, 4, 2, 1],
        window_size=4, patch_size=4,
    )
    ae = AutoEncoder(encoder=enc, decoder=dec, bottleneck=VAEBottleneck())

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
    assert latents.shape == (B, _MINI_DIM, 32, 4), (
        f"Expected (B,{_MINI_DIM},32,4) after freq crop, got {latents.shape}"
    )
    assert not torch.isnan(latents).any(), "NaN in latents after freq crop"
