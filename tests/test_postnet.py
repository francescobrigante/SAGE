# ===============
# ResidualPostNet tests: exact identity at init, gradient flow into the
# zero conv, and SAGEDecoder integration (use_postnet flag + strict=False
# checkpoint loading leaves the pretrained function bit-identical), and the
# time-sliced pass of long inputs (same output as one pass).
# ===============
import sys

import pytest
import torch

from sage.model.swin.postnet import ResidualPostNet
from sage.model.decoder import SAGEDecoder


def _tiny_decoder(use_postnet: bool) -> SAGEDecoder:
    """2-stage real decoder on the production (1024, 128) STFT geometry."""
    return SAGEDecoder(
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


def _forward_to_norm(dec: SAGEDecoder, z: torch.Tensor) -> torch.Tensor:
    """Replicate SAGEDecoder.forward up to (but excluding) norm/unembed."""
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


def _trained_postnet(hidden: int = 8) -> ResidualPostNet:
    """A postnet whose zero conv has moved (random weights), so it is not the identity."""
    torch.manual_seed(0)
    net = ResidualPostNet(channels=4, hidden=hidden).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return net


def _budget(net: ResidualPostNet, F: int, frames: int) -> int:
    """max_elements that lets one pass hold ``frames`` padded frames of one batch item."""
    return net._frame_elements(F) * frames


@pytest.mark.parametrize("B, T, frames, cap", [(1, 100, 20, 16384), (2, 97, 11, 16384), (3, 50, 7, 16384),
                                               (1, 64, 65, 16384), (2, 33, 12, 16384), (1, 100, 400, 9), (2, 50, 400, 1)])
def test_sliced_postnet_matches_one_pass(B, T, frames, cap):
    """Slices of a few frames (down to 1 frame + halos + pads), T not a multiple of the slice, B > 1, one frame
    short of a single pass, and slices capped by max_slice_frames."""
    net = _trained_postnet()
    S = torch.randn(B, 4, 16, T)                        # (B, C, F, T)
    with torch.no_grad():
        ref = net(S)
        net.max_elements = _budget(net, 16, frames) if frames < T + 2 else _budget(net, 16, T + 1)
        net.max_slice_frames = cap
        out = net(S)
    assert (out - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()


def test_sliced_postnet_needs_the_full_halo():
    """With one frame of halo less, slice edges see the reflect padding: the halo is what makes it exact."""
    net = _trained_postnet()
    S = torch.randn(1, 4, 16, 60)
    with torch.no_grad():
        ref = net(S)
        net.max_elements = _budget(net, 16, 10)
        net._halo -= 1
        out = net(S)
    assert (out - ref).abs().max().item() > 1e-2


def test_postnet_below_the_limit_is_one_pass():
    """Inputs that fit run the original single pass: bit-identical."""
    net = _trained_postnet()
    S = torch.randn(2, 4, 16, 40)
    with torch.no_grad():
        ref = S + net.net(S)
        net.max_elements = 2 * _budget(net, 16, 40 + 2)  # exactly fits: B * (T + 2 pad) frames
        assert torch.equal(net(S), ref)


def test_sliced_postnet_rejects_a_budget_below_one_frame():
    net = _trained_postnet()
    net.max_elements = _budget(net, 16, 6)               # less than 1 frame + 2 halos + 2 pads
    with pytest.raises(ValueError, match="exceeds"), torch.no_grad():
        net(torch.randn(1, 4, 16, 20))


def _count_passes(net: ResidualPostNet) -> list:
    calls = []
    net.net[0].register_forward_hook(lambda *_: calls.append(1))
    return calls


def test_postnet_slices_above_max_slice_frames():
    """More than max_slice_frames frames over the batch run in slices per item (same output up to rounding), even
    when a single pass would fit; up to max_slice_frames run the single pass bit for bit."""
    net = _trained_postnet()
    S, small = torch.randn(2, 4, 16, 50), torch.randn(1, 4, 16, 7)
    calls = _count_passes(net)
    with torch.no_grad():
        ref, ref_small = net(S), net(small)
        assert len(calls) == 2                                       # one pass each
        net.max_slice_frames = 7
        out = net(S)
        assert len(calls) == 2 + 2 * 8                               # 2 items x ceil(50 / 7) slices
        assert torch.equal(net(small), ref_small)                    # B * T <= 7: single pass
        assert len(calls) == 2 + 2 * 8 + 1
    assert (out - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()


def test_sliced_postnet_default_limit_is_32_bit():
    """Production geometry: the paper decoder's postnet (hidden 64, F 1024) runs one pass up to 16384 frames over
    the batch (~190 s), in slices of 16384 above; the 32-bit budget alone would allow 32575, where its padded hidden
    tensor (64, 1030, T + 2) reaches 2**31 elements (measured on A100: the conv slows down 4-5x right above, the
    reflection pad fails above 32767 frames)."""
    net = ResidualPostNet(channels=4, hidden=64)
    assert ResidualPostNet.max_elements == 2**31 - 1 and ResidualPostNet.max_slice_frames == 16384
    assert net.max_elements // net._frame_elements(1024) - 2 == 32575
    assert ResidualPostNet.__doc__ and "32-bit index math" in ResidualPostNet.__doc__


if __name__ == "__main__":
    test_postnet_identity_at_init()
    test_postnet_zero_conv_receives_gradient()
    test_decoder_postnet_flag_and_identity()
    test_decoder_strict_false_load_preserves_pretrained_function()
    print("ALL POSTNET TESTS PASSED")
