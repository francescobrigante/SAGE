#!/usr/bin/env python3
"""
Download and extract FMA metadata + FMA large audio.

Sources (official):
- https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
- https://os.unil.cloud.switch.ch/fma/fma_large.zip
SHA1 (from official repo README):
- fma_metadata.zip: f0df49ffe5f2a6008d7dc83c6915b31835dfe733
- fma_large.zip:    497109f4dd721066b5ce5e5f250ec604dc78939e
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URLS = {
    "fma_metadata.zip": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
    "fma_large.zip": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
}

SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_large.zip": "497109f4dd721066b5ce5e5f250ec604dc78939e",
}


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_with_resume(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    existing = dst.stat().st_size if dst.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        # If server ignored Range, it will return 200 and we should restart.
        if existing > 0 and getattr(resp, "status", 200) == 200:
            existing = 0

        total = resp.headers.get("Content-Length")
        total = int(total) + existing if total is not None else None

        mode = "ab" if existing > 0 else "wb"
        downloaded = existing

        with open(dst, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    mb = downloaded / (1024 * 1024)
                    tmb = total / (1024 * 1024)
                    print(f"\r{dst.name}: {mb:,.1f} / {tmb:,.1f} MiB ({pct:5.1f}%)", end="")
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r{dst.name}: {mb:,.1f} MiB", end="")

    print()  # newline


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default="/home/ec2-user/cerovaz/data/fma_large", help="Directory where zips and extracted data will be stored.")
    p.add_argument("--skip_metadata", action="store_true", help="Do not download/extract fma_metadata.zip")
    p.add_argument("--skip_large", action="store_true", help="Do not download/extract fma_large.zip")
    p.add_argument("--no_extract", action="store_true", help="Only download; do not extract.")
    p.add_argument("--no_verify", action="store_true", help="Skip SHA1 verification.")
    args = p.parse_args()

    out_dir: Path = args.out_dir.expanduser().resolve()
    zips_dir = out_dir / "zips"
    data_dir = out_dir / "data"

    plan = []
    if not args.skip_metadata:
        plan.append("fma_metadata.zip")
    if not args.skip_large:
        plan.append("fma_large.zip")

    if not plan:
        print("Nothing to do (both --skip_metadata and --skip_large set).")
        return 0

    for name in plan:
        url = URLS[name]
        dst = zips_dir / name

        print(f"Downloading {name} ...")
        download_with_resume(url, dst)

        if not args.no_verify:
            print(f"Verifying SHA1 for {name} ...")
            got = sha1_file(dst)
            exp = SHA1[name]
            if got.lower() != exp.lower():
                raise RuntimeError(f"SHA1 mismatch for {name}:\n  expected {exp}\n  got      {got}")
            print("OK")

        if not args.no_extract:
            extract_zip(dst, data_dir)

    print(f"Done.\nDownloaded zips: {zips_dir}\nExtracted data: {data_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        returncode = 1
        raise SystemExit(returncode)
