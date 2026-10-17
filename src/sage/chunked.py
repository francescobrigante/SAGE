# ===============
# Chunked inference for long audio, as SAME does it (overlap-discard): encode and decode run on chunks that carry
# a margin of context per side, and the margin is thrown away. Model-agnostic: the codec passes the functions that
# encode a waveform to raw posterior parameters, sample them, and decode; this module plans the chunks and
# stitches the latent along its time axis.
# ===============
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple, Optional

import torch


class Span(NamedTuple):
    """Frames ``[start, stop)`` are kept; the chunk is computed on ``[lo, hi)``, the kept frames plus the margins."""
    start: int
    stop: int
    lo: int
    hi: int


@dataclass(frozen=True)
class LatentGeometry:
    """How STFT frames map to the latent of a SAGE model.

    The latent ``(B, C, L)`` is the encoder's ``(bands, L // bands)`` token grid flattened band-major, so its time
    axis is the last one of ``view(B, C, bands, L // bands)``; one latent step covers ``frames_per_step`` frames.
    """
    hop: int                    # STFT hop, in samples
    frames_per_step: int        # STFT frames per latent time step (the encoder's time downsampling)
    bands: int                  # latent frequency bands (1 when the encoder folds frequency into channels)
    grid: int                   # frames of the coarsest attention window: chunk starts stay on its multiples

    def frames(self, seconds: float, sample_rate: int) -> int:
        """``seconds`` in frames, rounded to a multiple of ``grid``."""
        return round(seconds * sample_rate / self.hop / self.grid) * self.grid

    def frames_of(self, latent_len: int) -> int:
        """STFT frames of a latent of length ``bands * T_lat``."""
        return latent_len // self.bands * self.frames_per_step

    def to_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Flat latent -> the token grid whose last axis is time."""
        return z.view(*z.shape[:-1], self.bands, z.shape[-1] // self.bands)       # (B, C, bands, T_lat)

    @staticmethod
    def to_flat(g: torch.Tensor) -> torch.Tensor:
        """Token grid -> flat latent, band-major (the inverse of ``to_grid``)."""
        return g.reshape(*g.shape[:-2], -1)                                        # (B, C, bands * T_lat)


def plan(frames: int, core: int, margin: int) -> Iterator[Span]:
    """Chunks of ``core`` frames from frame 0 (the last one shorter), each with ``margin`` frames of context per side
    where the signal has them."""
    if core < 1 or margin < 0:
        raise ValueError(f"chunk core must be >= 1 frame and margin >= 0, got core={core}, margin={margin}")
    for start in range(0, frames, core):
        stop = min(start + core, frames)
        yield Span(start, stop, max(start - margin, 0), min(stop + margin, frames))


def _like_layout(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``t`` stored in the same dimension order as ``ref``. Sampling uses ``randn_like``, which keeps the strides:
    the stitched parameters must be laid out like the encoder's output for a seed to give the one-pass noise."""
    order = sorted(range(ref.dim()), key=lambda d: -ref.stride(d))
    inverse = [order.index(d) for d in range(ref.dim())]
    return t.permute(order).contiguous().permute(inverse)


def _samples_end(span_hi: int, frames: int, length: int, hop: int) -> int:
    """Waveform end of frames ``[., span_hi)`` (centred STFT: N samples give N // hop + 1 frames)."""
    return (span_hi - 1) * hop if span_hi < frames else length


def encode(
    audio: torch.Tensor,
    geo: LatentGeometry,
    core: int,
    margin: int,
    encode_params: Callable[[torch.Tensor], torch.Tensor],
    sample: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``(B, C, N)`` waveform on the encoder's frame grid -> ``(B, latent_ch, bands * T_lat)`` latent, with
    ``T_lat = L / frames_per_step`` and ``L = N // hop + 1`` STFT frames.

    Each chunk yields its posterior parameters; they are cut to the kept frames along the latent time axis,
    stitched, and sampled once on the full latent, so a seed draws the same noise as a one-pass encode.
    """
    N = audio.shape[-1]
    L = N // geo.hop + 1
    r = geo.frames_per_step
    if L % r:
        raise ValueError(f"{L} STFT frames is not a multiple of {r}: pad the waveform first (SAGE.pad)")
    parts, ref = [], None
    for s in plan(L, core, margin):
        p = encode_params(audio[..., s.lo * geo.hop:_samples_end(s.hi, L, N, geo.hop)])   # (B, 2C, hi - lo)
        ref = p if ref is None else ref
        parts.append(geo.to_grid(p)[..., (s.start - s.lo) // r:(s.stop - s.lo) // r])      # (B, 2C, bands, kept)
    params = _like_layout(geo.to_flat(torch.cat(parts, -1)), ref)                         # (B, 2C, L)
    return sample(params)                                                                 # (B, C, L)


def decode(
    latents: torch.Tensor,
    geo: LatentGeometry,
    core: int,
    margin: int,
    decode_fn: Callable[[torch.Tensor, int], torch.Tensor],
    target_length: Optional[int] = None,
) -> torch.Tensor:
    """``(B, C, bands * T_lat)`` latent -> ``(B, audio_ch, N)`` waveform of its ``L = T_lat * frames_per_step``
    STFT frames, ``N = target_length`` (default ``(L - 1) * hop``).

    The right context is at least one latent step even with no margin: the last samples of a chunk's kept frames
    need the next frame of the centred iSTFT.
    """
    L = geo.frames_of(latents.shape[-1])
    N = target_length if target_length is not None else (L - 1) * geo.hop
    r = geo.frames_per_step
    z = geo.to_grid(latents)                                                              # (B, C, bands, T_lat)
    out = []
    for s in plan(L, core, margin):
        hi = min(s.stop + max(margin, r), L)
        chunk = geo.to_flat(z[..., s.lo // r:hi // r].contiguous())                       # (B, C, hi - lo)
        y = decode_fn(chunk, _samples_end(hi, L, N, geo.hop) - s.lo * geo.hop)            # (B, audio_ch, n_chunk)
        out.append(y[..., (s.start - s.lo) * geo.hop:(s.stop - s.lo) * geo.hop if s.stop < L else None])
    return torch.cat(out, -1)[..., :N]                                                    # (B, audio_ch, N)
