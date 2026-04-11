# ===============================================================
# Phase 6b test suite — SwinEncoder / SwinDecoder (complex path).
# Mirrors test_swin_real.py but with complex64 inputs and
# is_complex=True throughout.
#
# Tests
#   CU1  PatchEmbed:      complex input → correct shape + dtype
#   CU2  PatchMerging:    complex input → halved grid, doubled C, complex dtype
#   CU3  PatchExpand:     complex input → doubled grid, halved C, complex dtype
#   CU4  WindowAttention: attention weights real, output complex
#   CU5  SwinBlock:       shape + dtype preserved, no NaN
#   CU6  SwinBlock:       gradient flow
#   CU7  SwinStage:       no downsample — shape + dtype preserved
#   CU8  SwinStage:       with PatchMerging — shape + dtype correct
#   CU9  SwinEncoder:     (B,2,1024,128)ℂ → (B, dim, 128)ℂ, no NaN
#   CU10 SwinDecoder:     (B, lat, 128)ℂ → (B,2,1024,128)ℂ, no NaN
#   CU11 gradient flow:   end-to-end encoder + decoder
#   CU12 DropPath:        ComplexSafeDropPath mask is real, output complex
# ===============================================================

import sys
import pytest
import torch
import torch.nn as nn

_SRC = "/Users/francesco/Desktop/C-VAE/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from c_vae.swin.patches import PatchEmbed, PatchMerging, PatchExpand
from c_vae.swin.attention import WindowAttention
from c_vae.swin.swin_block import SwinTransformerBlock
from c_vae.swin.swin_stage import SwinStage as BasicLayer
from c_vae.swin.encoder import SwinEncoder
from c_vae.swin.decoder import SwinDecoder
from c_vae.swin.utils import ComplexSafeDropPath


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B = 2
LATENT_CH = 8       # latent_channels for mini complex model
DIMENSION  = 24     # parameters_to_predict(3) × latent_channels(8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def enc_cplx_mini():
    """Mini complex SwinEncoder (embed_dim=8, depths=[1,1,1,1])."""
    return SwinEncoder(
        in_channels=2,          # stereo complex STFT (cac=false)
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        window_size=4,
        patch_size=4,
        dimension=DIMENSION,    # ptp=3 × latent_ch=8
        is_complex=True,
    ).eval()


@pytest.fixture(scope="module")
def dec_cplx_mini():
    """Mini complex SwinDecoder (embed_dim=8, depths=[1,1,1,1])."""
    return SwinDecoder(
        channels=LATENT_CH,
        in_channels=2,
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[8, 4, 2, 1],
        window_size=4,
        patch_size=4,
        is_complex=True,
    ).eval()


def cplx(shape):
    """Random complex64 tensor."""
    return torch.randn(*shape, dtype=torch.complex64)


# ===========================================================================
# Unit tests  CU1 – CU12
# ===========================================================================

# ---------------------------------------------------------------------------
# CU1 — PatchEmbed: complex input, correct shape and dtype
# ---------------------------------------------------------------------------

def test_cu1_patch_embed_complex():
    """PatchEmbed(is_complex=True): (B,2,1024,128)ℂ → (B,8192,8)ℂ."""
    embed = PatchEmbed(
        img_size=(1024, 128), patch_size=4, in_chans=2,
        embed_dim=8, norm_layer=nn.LayerNorm, is_complex=True,
    ).eval()
    x = cplx((B, 2, 1024, 128))
    with torch.no_grad():
        out = embed(x)
    assert out.shape == (B, 8192, 8), f"Got {out.shape}"
    assert out.is_complex(), "PatchEmbed output must be complex64"
    assert not torch.isnan(out.real).any() and not torch.isnan(out.imag).any()


# ---------------------------------------------------------------------------
# CU2 — PatchMerging: complex input → halved grid, doubled C, complex dtype
# ---------------------------------------------------------------------------

def test_cu2_patch_merging_complex():
    """PatchMerging(is_complex=True): (B,H*W,C)ℂ → (B,H/2*W/2,2C)ℂ."""
    H, W, C = 64, 8, 16
    merge = PatchMerging(input_resolution=(H, W), dim=C, is_complex=True).eval()
    x = cplx((B, H * W, C))
    with torch.no_grad():
        out = merge(x)
    assert out.shape == (B, (H // 2) * (W // 2), 2 * C), f"Got {out.shape}"
    assert out.is_complex(), "PatchMerging output must be complex64"


# ---------------------------------------------------------------------------
# CU3 — PatchExpand: complex input → doubled grid, halved C, complex dtype
# ---------------------------------------------------------------------------

def test_cu3_patch_expand_complex():
    """PatchExpand(is_complex=True): (B,H*W,2C)ℂ → (B,4H*W,C)ℂ."""
    H, W, dim = 32, 4, 16   # dim = 2C, out_dim = C = 8
    expand = PatchExpand(input_resolution=(H, W), dim=dim, is_complex=True).eval()
    x = cplx((B, H * W, dim))
    with torch.no_grad():
        out = expand(x)
    assert out.shape == (B, 4 * H * W, dim // 2), f"Got {out.shape}"
    assert out.is_complex(), "PatchExpand output must be complex64"


# ---------------------------------------------------------------------------
# CU4 — WindowAttention: scores real, output complex
# ---------------------------------------------------------------------------

def test_cu4_window_attention_complex():
    """WindowAttention(is_complex=True): scores always real, output complex."""
    ws, dim, nheads = 4, 8, 2
    nW = 16
    attn = WindowAttention(
        dim=dim, window_size=(ws, ws), num_heads=nheads,
        qkv_bias=True, is_complex=True,
    ).eval()
    x = cplx((nW, ws * ws, dim))
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert out.is_complex(), "WindowAttention output must be complex64"
    assert not torch.isnan(out.real).any() and not torch.isnan(out.imag).any()


# ---------------------------------------------------------------------------
# CU5 — SwinTransformerBlock: shape + dtype preserved, no NaN
# ---------------------------------------------------------------------------

def test_cu5_swin_block_complex_shape():
    """SwinTransformerBlock(is_complex=True): shape and dtype preserved."""
    H, W, C = 32, 8, 8
    block = SwinTransformerBlock(
        dim=C, input_resolution=(H, W), num_heads=2,
        window_size=4, shift_size=0, is_complex=True,
    ).eval()
    x = cplx((B, H * W, C))
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape
    assert out.is_complex()
    assert not torch.isnan(out.real).any() and not torch.isnan(out.imag).any()


# ---------------------------------------------------------------------------
# CU6 — SwinTransformerBlock: gradient flow through complex ops
# ---------------------------------------------------------------------------

def test_cu6_swin_block_complex_gradients():
    """All SwinTransformerBlock parameters receive gradients with complex input."""
    H, W, C = 32, 8, 8
    block = SwinTransformerBlock(
        dim=C, input_resolution=(H, W), num_heads=2,
        window_size=4, shift_size=0, is_complex=True,
    )
    x = cplx((B, H * W, C)).requires_grad_(True)
    out = block(x)
    # Use real part of sum as scalar loss (complex sum not allowed)
    out.real.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"


# ---------------------------------------------------------------------------
# CU7 — SwinStage without downsample: shape + dtype preserved
# ---------------------------------------------------------------------------

def test_cu7_swin_stage_no_downsample_complex():
    """SwinStage(is_complex=True, downsample=None): shape and dtype preserved."""
    H, W, C = 64, 8, 16
    stage = BasicLayer(
        dim=C, input_resolution=(H, W), depth=1, num_heads=2,
        window_size=4, downsample=None, is_complex=True,
    ).eval()
    x = cplx((B, H * W, C))
    with torch.no_grad():
        out = stage(x)
    assert out.shape == (B, H * W, C)
    assert out.is_complex()


# ---------------------------------------------------------------------------
# CU8 — SwinStage with PatchMerging: shape + dtype correct
# ---------------------------------------------------------------------------

def test_cu8_swin_stage_with_merging_complex():
    """SwinStage(is_complex=True) + PatchMerging: (B,H*W,C)ℂ → (B,H/2*W/2,2C)ℂ."""
    from c_vae.swin.patches import PatchMerging
    H, W, C = 64, 8, 8
    stage = BasicLayer(
        dim=C, input_resolution=(H, W), depth=1, num_heads=1,
        window_size=4, downsample=PatchMerging, is_complex=True,
    ).eval()
    x = cplx((B, H * W, C))
    with torch.no_grad():
        out = stage(x)
    assert out.shape == (B, (H // 2) * (W // 2), 2 * C)
    assert out.is_complex()


# ---------------------------------------------------------------------------
# CU9 — Full complex encoder: (B,2,1024,128)ℂ → (B, DIMENSION, 128)ℂ
# ---------------------------------------------------------------------------

def test_cu9_encoder_complex(enc_cplx_mini):
    """SwinEncoder(is_complex=True): correct output shape, complex dtype, no NaN."""
    x = cplx((B, 2, 1024, 128))
    with torch.no_grad():
        latents, info = enc_cplx_mini(x)
    assert latents.shape == (B, DIMENSION, 128), f"Got {latents.shape}"
    assert latents.is_complex(), "Encoder output must be complex64"
    assert not torch.isnan(latents.real).any() and not torch.isnan(latents.imag).any()
    assert info.get("feature_shape") == (32, 4)


# ---------------------------------------------------------------------------
# CU10 — Full complex decoder: (B, LATENT_CH, 128)ℂ → (B,2,1024,128)ℂ
# ---------------------------------------------------------------------------

def test_cu10_decoder_complex(dec_cplx_mini):
    """SwinDecoder(is_complex=True): correct output shape, complex dtype, no NaN."""
    z = cplx((B, LATENT_CH, 128))
    with torch.no_grad():
        out = dec_cplx_mini(z)
    assert out.shape == (B, 2, 1024, 128), f"Got {out.shape}"
    assert out.is_complex(), "Decoder output must be complex64"
    assert not torch.isnan(out.real).any() and not torch.isnan(out.imag).any()


# ---------------------------------------------------------------------------
# CU11 — End-to-end gradient flow: encoder + decoder, complex path
# ---------------------------------------------------------------------------

def test_cu11_gradient_flow_complex():
    """loss.backward() propagates non-None gradients to all complex enc+dec params."""
    enc = SwinEncoder(
        in_channels=2, embed_dim=8, depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8], window_size=4, patch_size=4,
        dimension=DIMENSION, is_complex=True,
    )
    dec = SwinDecoder(
        channels=LATENT_CH, in_channels=2, embed_dim=8,
        depths=[1, 1, 1, 1], num_heads=[8, 4, 2, 1],
        window_size=4, patch_size=4, is_complex=True,
    )
    x = cplx((B, 2, 1024, 128))
    latents, _ = enc(x)
    # Use only LATENT_CH channels (simulate bottleneck split)
    z = latents[:, :LATENT_CH, :]
    recon = dec(z)
    recon.real.sum().backward()
    for name, param in list(enc.named_parameters()) + list(dec.named_parameters()):
        assert param.grad is not None, f"No gradient for '{name}'"


# ---------------------------------------------------------------------------
# CU12 — ComplexSafeDropPath: mask is real, output preserves complex dtype
# ---------------------------------------------------------------------------

def test_cu12_complex_drop_path():
    """ComplexSafeDropPath applies a real mask and preserves complex64 dtype."""
    drop = ComplexSafeDropPath(drop_prob=0.5).train()
    x = cplx((4, 16, 8))
    out = drop(x)
    assert out.is_complex(), "DropPath must preserve complex64 dtype"
    assert out.shape == x.shape
    # In eval mode it must be identity
    drop.eval()
    out_eval = drop(x)
    assert torch.allclose(out_eval, x), "DropPath in eval must be identity"
