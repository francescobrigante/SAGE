# ===============
# ResidualPostNet tests: exact identity at init, gradient flow into the
# zero conv, and SwinDecoder integration (use_postnet flag + strict=False
# checkpoint loading leaves the pretrained function bit-identical).
# ===============
import sys

import torch

from c_vae.swin.postnet import ResidualPostNet
from c_vae.swin.decoder import SwinDecoder


def _tiny_decoder(use_postnet: bool) -> SwinDecoder:
    """2-stage real decoder on the production (1024, 128) STFT geometry."""
    return SwinDecoder(
        channels=16, in_channels=4, embed_dim=8,
        depths=[1, 1], num_heads=[2, 2],
        window_size=(4, 32), patch_size=(64, 1),
        time_frames=128, drop_path_rate=0.0,
        is_complex=False, use_postnet=use_postnet,
    )


def test_postnet_identity_at_init():
    net = ResidualPostNet(channels=4, hidden=8)
    S = torch.randn(2, 4, 64, 32)                       # (B, C, F, T)
    with torch.no_grad():
        out = net(S)
    assert torch.equal(out, S)                          # exact, not allclose


def test_postnet_zero_conv_receives_gradient():
    net = ResidualPostNet(channels=4, hidden=8)
    S = torch.randn(2, 4, 64, 32, requires_grad=True)
    (net(S) ** 2).mean().backward()
    last = net.net[-1]
    # the zero conv itself must receive a nonzero gradient (it can move away from 0)
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0


def test_decoder_postnet_flag_and_identity():
    torch.manual_seed(0)
    dec = _tiny_decoder(use_postnet=True)
    assert hasattr(dec, "postnet")
    z = torch.randn(1, 16, 8 * 64)                      # (B, latent_ch, H_lat*W_lat)
    with torch.no_grad():
        out = dec(z)
    assert out.shape == (1, 4, 1024, 128)
    # postnet is identity at init → bypassing it gives the same output
    with torch.no_grad():
        out_no_post = dec.patch_unembed(dec.norm(_forward_to_norm(dec, z)))
    assert torch.equal(out, out_no_post)


def _forward_to_norm(dec: SwinDecoder, z: torch.Tensor) -> torch.Tensor:
    """Replicate SwinDecoder.forward up to (but excluding) norm/unembed."""
    x = z.transpose(1, 2)
    x = dec.input_proj(x)
    for i, stage in enumerate(dec.stages):
        x = stage(x)
        if i < dec.num_stages - 1:
            x = dec.patch_expands[i](x)
    return x


def test_decoder_strict_false_load_preserves_pretrained_function():
    """Loading a no-postnet checkpoint into a postnet decoder (strict=False,
    same protocol as +init_from) must leave the function bit-identical to the
    pretrained decoder: only postnet keys may be missing, and they stay zero."""
    torch.manual_seed(0)
    dec_pre = _tiny_decoder(use_postnet=False)          # "pretrained" decoder
    torch.manual_seed(123)
    dec_ft = _tiny_decoder(use_postnet=True)            # finetune decoder (different init)

    missing, unexpected = dec_ft.load_state_dict(dec_pre.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("postnet.") for k in missing)

    z = torch.randn(1, 16, 8 * 64)
    with torch.no_grad():
        assert torch.equal(dec_ft(z), dec_pre(z))


if __name__ == "__main__":
    test_postnet_identity_at_init()
    test_postnet_zero_conv_receives_gradient()
    test_decoder_postnet_flag_and_identity()
    test_decoder_strict_false_load_preserves_pretrained_function()
    print("ALL POSTNET TESTS PASSED")
