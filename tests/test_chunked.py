# ===============
# sage.chunked: chunk planning, and chunked encode / decode against one pass on a fake codec with a finite
# receptive field (K latent steps, zero padding at its own edges) laid out like the SAGE latent (band-major grid,
# encoder output stored token-major). With a margin covering K the chunked result is the one-pass one exactly.
# Geometries with bands != frames_per_step keep the latent length and the frame count apart.
# ===============
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sage import chunked

HOP, R, WINDOW = 4, 2, 4                            # samples per frame, frames per latent step, window steps
K = 3                                               # receptive field of the fake codec, latent steps per side
GEOS = [chunked.LatentGeometry(hop=HOP, frames_per_step=R, bands=b, grid=WINDOW * R) for b in (1, 2, 3)]
GEO = GEOS[1]


def _context(x: torch.Tensor) -> torch.Tensor:
    """Mixes K latent steps per side along the last axis, zero padding at the ends: (..., T) -> (..., T)."""
    w = torch.arange(1.0, 2 * K + 2).view(1, 1, -1) / (2 * K + 1)
    return F.conv1d(x.reshape(-1, 1, x.shape[-1]), w, padding=K).view(x.shape)


class FakeCodec:
    """Posterior params [mu | scale] of 2 latent channels on a (bands, T_lat) grid, T_lat = frames / R."""

    def __init__(self, geo: chunked.LatentGeometry):
        self.geo = geo

    def encode_params(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, 1, N) -> (B, 4, bands * T_lat), stored token-major like SAGEEncoder's output (with one latent
        channel the layout of the noise would not matter)."""
        B, N = audio.shape[0], audio.shape[-1]
        L = N // HOP + 1
        frames = audio[..., torch.clamp(torch.arange(L) * HOP, max=N - 1)]          # (B, 1, L): centre sample
        steps = _context(frames.view(B, 1, L // R, R).mean(-1))                     # (B, 1, T_lat)
        grid = torch.cat([steps * (b + 1) for b in range(self.geo.bands)], 1)       # (B, bands, T_lat)
        params = torch.stack([grid, 0.5 * grid, 0.1 * grid.tanh(), grid.cos()], 1)  # (B, 4, bands, T_lat)
        return params.reshape(B, 4, -1).transpose(1, 2).contiguous().transpose(1, 2)   # token-major

    @staticmethod
    def sample(params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, scale = params.chunk(2, dim=1)                                        # (B, 2, bands * T_lat)
        return mean if deterministic else mean + torch.randn_like(mean) * F.softplus(scale)

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        """(B, 2, bands * T_lat) -> (B, 1, n): one value per latent step, spread over its R * HOP samples."""
        g = self.geo.to_grid(z).sum(-2).sum(1, keepdim=True)                       # (B, 1, T_lat)
        return _context(g).repeat_interleave(R * HOP, -1)[..., :n]                  # (B, 1, n)

    def mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample(self.encode_params(x), deterministic=True)

    def chunked_mu(self, x: torch.Tensor, core: int, margin: int) -> torch.Tensor:
        return chunked.encode(x, self.geo, core, margin, self.encode_params, lambda p: self.sample(p, True))


def _audio(N: int, B: int = 1) -> torch.Tensor:
    return torch.randn(B, 1, N, generator=torch.Generator().manual_seed(N))


LENGTHS = [N for N in range(200, 260) if (N // HOP + 1) % R == 0][:6]
LENGTHS += [4 * 8 * 7 - 4, 4 * 8 * 7 + 4]                                       # last chunk full or 2 frames


@pytest.mark.parametrize("frames, core, margin", [(64, 16, 8), (60, 16, 0), (16, 16, 8), (10, 32, 8), (100, 24, 40)])
def test_plan_covers_every_frame_once_with_clipped_margins(frames, core, margin):
    spans = list(chunked.plan(frames, core, margin))
    assert [s.start for s in spans] == list(range(0, frames, core))
    assert [s.stop for s in spans[:-1]] == [s.start for s in spans[1:]] and spans[-1].stop == frames
    for s in spans:
        assert s.lo == max(s.start - margin, 0) and s.hi == min(s.stop + margin, frames)


@pytest.mark.parametrize("core, margin", [(0, 0), (8, -1)])
def test_plan_rejects_bad_sizes(core, margin):
    with pytest.raises(ValueError):
        list(chunked.plan(64, core, margin))


def test_geometry_frames_round_to_the_window_grid():
    g = chunked.LatentGeometry(hop=512, frames_per_step=4, bands=4, grid=128)
    assert g.frames(300, 44100) == 202 * 128 and g.frames(6, 44100) == 4 * 128 and g.frames(0.5, 44100) == 0
    z = torch.arange(2 * 3 * 4 * 5.0).view(2, 3, 20)
    assert torch.equal(g.to_flat(g.to_grid(z)), z) and g.to_grid(z).shape == (2, 3, 4, 5)
    assert g.frames_of(20) == 20                                                # paper model: bands == frames/step
    assert chunked.LatentGeometry(hop=512, frames_per_step=4, bands=1, grid=128).frames_of(20) == 80


@pytest.mark.parametrize("geo", GEOS, ids=lambda g: f"bands{g.bands}")
@pytest.mark.parametrize("N", LENGTHS)
@pytest.mark.parametrize("B", [1, 2])
def test_chunked_encode_and_decode_equal_one_pass_with_a_margin_covering_the_context(geo, N, B):
    codec, x, margin = FakeCodec(geo), _audio(N, B), geo.grid                  # margin: 4 steps >= K
    mu = codec.mu(x)
    mu_c = codec.chunked_mu(x, geo.grid, margin)
    assert mu_c.shape == mu.shape == (B, 2, geo.bands * ((N // HOP + 1) // R))
    assert torch.allclose(mu_c, mu, atol=1e-6)
    y = codec.decode(mu, N)
    y_c = chunked.decode(mu, geo, geo.grid, margin, codec.decode, target_length=N)
    assert y_c.shape == y.shape == (B, 1, N) and torch.allclose(y_c, y, atol=1e-6)


@pytest.mark.parametrize("geo", GEOS, ids=lambda g: f"bands{g.bands}")
def test_chunked_differs_without_margin(geo):
    """The fake codec does need its context: the equality above is not trivial."""
    codec, x = FakeCodec(geo), _audio(LENGTHS[0])
    mu = codec.mu(x)
    assert (codec.chunked_mu(x, geo.grid, 0) - mu).abs().max() > 1e-2
    y_c = chunked.decode(mu, geo, geo.grid, 0, codec.decode, target_length=x.shape[-1])
    assert (y_c - codec.decode(mu, x.shape[-1])).abs().max() > 1e-2


def test_chunked_sampling_draws_the_one_pass_noise():
    """Same seed, same z: the stitched params are stored like the encoder's output, so randn_like lays the noise
    out identically (a contiguous copy would not)."""
    codec, x = FakeCodec(GEO), _audio(LENGTHS[0], B=2)
    params = codec.encode_params(x)
    torch.manual_seed(0)
    z = codec.sample(params)
    torch.manual_seed(0)
    z_c = chunked.encode(x, GEO, GEO.grid, GEO.grid, codec.encode_params, codec.sample)
    assert torch.allclose(z_c, z, atol=1e-6)
    torch.manual_seed(0)
    assert (codec.sample(params.contiguous()) - z).abs().max() > 1e-1          # layout matters


@pytest.mark.parametrize("geo", GEOS, ids=lambda g: f"bands{g.bands}")
def test_decode_lengths_and_short_right_context(geo):
    """target_length defaults to (frames - 1) * hop and a shorter one is honoured; with margin 0 the right
    context is still one latent step."""
    codec, x = FakeCodec(geo), _audio(LENGTHS[-1])
    mu = codec.mu(x)
    L = geo.frames_of(mu.shape[-1])
    assert L == x.shape[-1] // HOP + 1
    assert chunked.decode(mu, geo, geo.grid, geo.grid, codec.decode).shape[-1] == (L - 1) * HOP
    assert chunked.decode(mu, geo, geo.grid, geo.grid, codec.decode, target_length=10).shape[-1] == 10
    calls = []
    chunked.decode(mu, geo, geo.grid, 0, lambda z, n: calls.append(z.shape[-1]) or codec.decode(z, n))
    assert calls[:-1] == [geo.bands * (geo.grid + R) // R] * (len(calls) - 1)   # kept frames + one step


def test_encode_rejects_unpadded_audio():
    """A frame count off the latent step fails before any chunk runs, pointing to SAGE.pad."""
    calls = []
    codec = FakeCodec(GEO)
    with pytest.raises(ValueError, match="SAGE.pad"):
        chunked.encode(_audio(4 * 8 * 7), GEO, GEO.grid, GEO.grid,
                       lambda a: calls.append(1) or codec.encode_params(a), codec.sample)
    assert not calls
