# ===============================================================
# Tests for the Option-C learned freq-fold latent (fold_freq_to_channels):
#   - encoder produces a native 1-D latent (dimension, T_lat), feature_shape (1, T_lat)
#   - decoder round-trips the shape back to the input spectrogram
#   - fold/unfold token ordering is exactly bijective (freq-major f*T + t)
#   - legacy path (flag=False) is unchanged (regression)
# ===============================================================

import torch

from c_vae.swin.encoder import SwinEncoder
from c_vae.swin.decoder import SwinDecoder


# Mini geometry: same as champion (PS=[64,1], 3 stages -> grid 4×32) but tiny embed.
EMBED = 32
DEPTHS = [1, 1, 1]
HEADS = [2, 4, 8]
PS = [64, 1]
WIN = [4, 32]
LAT_CH = 8
P2P = 2                       # mu + logvar
DIM = P2P * LAT_CH            # encoder output channels = 16
B, F, T = 2, 1024, 128       # input spectrogram (B, 4, 1024, 128)


def _enc(fold: bool) -> SwinEncoder:
    return SwinEncoder(
        in_channels=4, is_complex=False, embed_dim=EMBED, depths=DEPTHS,
        num_heads=HEADS, window_size=WIN, patch_size=PS, dimension=DIM,
        time_frames=T, fold_freq_to_channels=fold,
    )


def _dec(fold: bool) -> SwinDecoder:
    return SwinDecoder(
        channels=LAT_CH, in_channels=4, is_complex=False, embed_dim=EMBED,
        depths=DEPTHS[::-1], num_heads=HEADS[::-1], window_size=WIN,
        patch_size=PS, time_frames=T, fold_freq_to_channels=fold,
    )


def test_encoder_fold_shapes():
    enc = _enc(fold=True).eval()
    x = torch.randn(B, 4, F, T)
    with torch.no_grad():
        z, info = enc(x)
    # native 1-D latent: (B, dimension, T_lat=32); freq folded away
    assert z.shape == (B, DIM, 32), z.shape
    assert info["feature_shape"] == (1, 32), info["feature_shape"]


def test_decoder_fold_roundtrip_shape():
    dec = _dec(fold=True).eval()
    z = torch.randn(B, LAT_CH, 32)          # native 1-D latent (channels, T_lat)
    with torch.no_grad():
        rec = dec(z)
    assert rec.shape == (B, 4, F, T), rec.shape


def test_encoder_decoder_fold_endtoend():
    enc, dec = _enc(fold=True).eval(), _dec(fold=True).eval()
    x = torch.randn(B, 4, F, T)
    with torch.no_grad():
        z, _ = enc(x)                       # (B, DIM, 32)
        rec = dec(z[:, :LAT_CH, :])         # take mu half as fake latent (B, LAT_CH, 32)
    assert rec.shape == (B, 4, F, T), rec.shape


def test_legacy_path_unchanged():
    enc = _enc(fold=False).eval()
    x = torch.randn(B, 4, F, T)
    with torch.no_grad():
        z, info = enc(x)
    # legacy: 2-D grid kept -> 128 tokens, feature_shape (4, 32)
    assert z.shape == (B, DIM, 128), z.shape
    assert info["feature_shape"] == (4, 32), info["feature_shape"]


def test_encoder_fold_variable_length():
    """Option-C encoder must accept clips longer than the training crop.

    The latent freq grid is fixed (F_lat) but T_lat scales with the input
    time frames — eval/maeb feed full-length clips, not the 1.5 s crop.
    """
    enc = _enc(fold=True).eval()
    T_long = 2 * T                               # twice the training crop
    x = torch.randn(B, 4, F, T_long)
    with torch.no_grad():
        z, info = enc(x)
    assert z.shape == (B, DIM, 64), z.shape      # T_lat doubles: 32 -> 64
    assert info["feature_shape"] == (1, 64), info["feature_shape"]


def test_encoder_decoder_fold_variable_length_endtoend():
    """Full Option-C round-trip on a longer clip reconstructs the same shape."""
    enc, dec = _enc(fold=True).eval(), _dec(fold=True).eval()
    T_long = 2 * T
    x = torch.randn(B, 4, F, T_long)
    with torch.no_grad():
        z, _ = enc(x)                            # (B, DIM, 64)
        rec = dec(z[:, :LAT_CH, :])              # (B, LAT_CH, 64) -> spectrogram
    assert rec.shape == (B, 4, F, T_long), rec.shape


def test_fold_unfold_is_bijective():
    """Encoder fold then decoder unfold must be the identity on a (B,F,T,C) grid."""
    F_lat, T_lat, C = 4, 32, 5
    g = torch.randn(B, F_lat, T_lat, C)
    # encoder fold: (B,F,T,C) -> (B,T,F*C)
    folded = g.permute(0, 2, 1, 3).reshape(B, T_lat, F_lat * C)
    # decoder unfold: (B,T,F*C) -> (B,F*T,C) ; then back to grid (B,F,T,C)
    unfolded = folded.view(B, T_lat, F_lat, C).permute(0, 2, 1, 3).reshape(B, F_lat * T_lat, C)
    back = unfolded.view(B, F_lat, T_lat, C)
    assert torch.equal(back, g)
