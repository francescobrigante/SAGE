#!/usr/bin/env python3
# =============================================================================
# Strip a training checkpoint down to what inference needs: the inference_config
# and the EMA autoencoder weights (no optimizer states, discriminator or live
# training weights). The released SAGE checkpoint was made with this script:
#
#   python scripts/export_checkpoint.py <training.ckpt> models/SAGE_FTe992.ckpt
#
# The result loads with SAGE.from_checkpoint and with `+init_from=` exactly like
# the source; --no-verify skips the check that both rebuild the same model.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

PREFIX = "ema_autoencoder."


def export(src: Path, dst: Path) -> dict:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "inference_config" not in ckpt:
        raise SystemExit(f"{src} has no inference_config: not a SAGE training checkpoint")
    weights = {k: v for k, v in ckpt["state_dict"].items() if k.startswith(PREFIX)}
    if not weights:
        raise SystemExit(f"{src} has no {PREFIX}* weights")
    out = {"inference_config": ckpt["inference_config"], "state_dict": weights,
           "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def verify(src: Path, dst: Path) -> None:
    """Both files rebuild the same autoencoder, weight for weight."""
    from sage import SAGE
    a = SAGE.from_checkpoint(src, device="cpu").autoencoder.state_dict()
    b = SAGE.from_checkpoint(dst, device="cpu").autoencoder.state_dict()
    assert a.keys() == b.keys(), "different parameter names"
    assert all(torch.equal(a[k], b[k]) for k in a), "different weights"
    print(f"verified: {len(a)} tensors identical")


def main() -> None:
    p = argparse.ArgumentParser(description="Strip a SAGE training checkpoint to an inference checkpoint (EMA weights).")
    p.add_argument("src", type=Path, help="training checkpoint (Lightning .ckpt with inference_config)")
    p.add_argument("dst", type=Path, help="output inference checkpoint")
    p.add_argument("--no-verify", action="store_true", help="skip rebuilding both models to compare them")
    args = p.parse_args()
    out = export(args.src, args.dst)
    print(f"{args.dst}: {len(out['state_dict'])} tensors, epoch {out['epoch']}, "
          f"{args.dst.stat().st_size / 2**20:.1f} MiB (source {args.src.stat().st_size / 2**20:.1f} MiB)")
    print(f"sha256 {sha256(args.dst)}")
    if not args.no_verify:
        verify(args.src, args.dst)


if __name__ == "__main__":
    main()
