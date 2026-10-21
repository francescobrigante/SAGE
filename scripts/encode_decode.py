#!/usr/bin/env python3
# =============================================================================
# Encode audio to SAGE latents, decode latents to audio, or do both.
#
#   python scripts/encode_decode.py reconstruct song.wav song_rec.wav --ckpt SAGE_FTe992.ckpt
#   python scripts/encode_decode.py encode      song.wav song.pt      --ckpt SAGE_FTe992.ckpt
#   python scripts/encode_decode.py decode      song.pt  song_rec.wav --ckpt SAGE_FTe992.ckpt
#
# Input audio is resampled to 44.1 kHz and made stereo (mono is duplicated). Latents are
# saved with torch.save as {"latent": (1, 16, T), "num_samples", "padded_samples", "sample_rate"},
# so `decode` restores the exact input length. Inputs longer than --chunk-seconds (300 s) run in
# chunks with context margins, so the memory stays bounded. Needs only the core install (pip install .).
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from sage import SAGE
from sage.constants import DEFAULT_CHUNK_MARGIN_SECONDS, DEFAULT_CHUNK_SECONDS
from sage.precision import DEFAULT_PRECISION, PRECISIONS


def load_audio(path: Path, sample_rate: int, channels: int) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)        # (N, C)
    wav = torch.from_numpy(data.T.copy())                                   # (C, N)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] < channels:
        wav = wav.repeat(channels // wav.shape[0], 1)                       # mono -> stereo
    return wav[:channels]


def save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    sf.write(str(path), wav.clamp(-1, 1).T.cpu().numpy(), sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["encode", "decode", "reconstruct"])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--ckpt", type=Path, required=True, help="SAGE checkpoint (e.g. SAGE_FTe992.ckpt)")
    ap.add_argument("--device", default=None, help="cpu, cuda, mps (default: cuda if available)")
    ap.add_argument("--varlen", default=None, help="variable-length attention preset; 'off' disables it (default: tri2)")
    ap.add_argument("--sdpa", action=argparse.BooleanOptionalAction, default=True,
                    help="attention through F.scaled_dot_product_attention (same formula, faster; default on); "
                         "--no-sdpa runs the original attention")
    ap.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                    help=f"inputs longer than this run in chunks of this length (default {DEFAULT_CHUNK_SECONDS:g} s)")
    ap.add_argument("--chunk-margin-seconds", type=float, default=DEFAULT_CHUNK_MARGIN_SECONDS,
                    help=f"context per side of each chunk, discarded (default {DEFAULT_CHUNK_MARGIN_SECONDS:g} s; "
                         "3 s or more gives the one-pass output)")
    ap.add_argument("--no-chunk", action="store_true", help="always one pass (memory grows with the length)")
    ap.add_argument("--precision", choices=PRECISIONS, default=DEFAULT_PRECISION,
                    help="tf32 (default, the paper's timings), fp32 (no TF32) or bf16 (autocast: faster on long inputs "
                         "only, slightly lower fidelity)")
    ap.add_argument("--deterministic", action="store_true",
                    help="use the posterior mean instead of a sampled latent (the paper metrics use sampled z)")
    args = ap.parse_args()

    codec = SAGE.from_checkpoint(args.ckpt, device=args.device, varlen=args.varlen, sdpa=args.sdpa,
                                 chunk_seconds=None if args.no_chunk else args.chunk_seconds,
                                 chunk_margin_seconds=args.chunk_margin_seconds, precision=args.precision)
    sr, channels = codec.sample_rate, codec.audio_channels

    if args.command == "reconstruct":
        wav = load_audio(args.input, sr, channels)
        save_audio(args.output, codec.reconstruct(wav.to(codec.device), deterministic=args.deterministic), sr)

    elif args.command == "encode":
        wav = load_audio(args.input, sr, channels)
        padded, n = codec.pad(wav.unsqueeze(0))
        latent = codec.encode(padded, deterministic=args.deterministic).cpu()
        torch.save({"latent": latent, "num_samples": n, "padded_samples": padded.shape[-1], "sample_rate": sr},
                   args.output)
        print(f"latent {tuple(latent.shape)} -> {args.output}")

    else:
        item = torch.load(args.input, map_location="cpu", weights_only=True)
        rec = codec.decode(item["latent"], target_length=item["padded_samples"])[0, ..., : item["num_samples"]]
        save_audio(args.output, rec, sr)

    if args.command != "encode":
        print(f"audio -> {args.output}")


if __name__ == "__main__":
    main()
