# =================================================================================
# Scan Corrupt Audio
#
#   CLI script to scan a dataset directory and delete files that fail to decode
# =================================================================================

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

import torchaudio

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


QUIET_DECODE = True
FAIL_ON_STDERR = False
PROBE_FRAMES = 1
PROBE_POSITIONS = 1
FULL_SCAN = False


def _init_worker(
    quiet_decode: bool,
    fail_on_stderr: bool,
    probe_frames: int,
    probe_positions: int,
    full_scan: bool,
) -> None:
    global QUIET_DECODE
    global FAIL_ON_STDERR
    global PROBE_FRAMES
    global PROBE_POSITIONS
    global FULL_SCAN
    QUIET_DECODE = bool(quiet_decode)
    FAIL_ON_STDERR = bool(fail_on_stderr)
    PROBE_FRAMES = int(probe_frames)
    PROBE_POSITIONS = int(probe_positions)
    FULL_SCAN = bool(full_scan)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _normalize_exts(raw_exts: List[str]) -> List[str]:
    exts: List[str] = []
    for item in raw_exts:
        for part in item.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if not part.startswith("."):
                part = "." + part
            exts.append(part)
    return sorted(set(exts))


def _iter_audio_files(root: Path, exts: List[str], follow_symlinks: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            if not name.lower().endswith(tuple(exts)):
                continue
            yield Path(dirpath) / name


from src.ar_spectra.utils.audio_validation import (
    stderr_to, read_stderr, decode_chunk, compute_offsets, check_file
)

def _check_file(path_str: str):
    return check_file(path_str, PROBE_POSITIONS, PROBE_FRAMES, FULL_SCAN, QUIET_DECODE, FAIL_ON_STDERR)

def _progress(iterable, total: int):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, unit="file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a dataset directory and delete audio files that fail to decode."
    )
    parser.add_argument("root", type=Path, help="Root directory of the dataset to scan")
    parser.add_argument(
        "--ext",
        dest="exts",
        action="append",
        default=[".mp3"],
        help="Audio extension(s) to include (e.g. .mp3 or mp3, comma-separated allowed). Default: .mp3",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes to use (0 = no multiprocessing).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Chunk size for multiprocessing map.",
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=4096,
        help="Frames to decode per probe (ignored when --full is set).",
    )
    parser.add_argument(
        "--probe-positions",
        type=int,
        default=3,
        help="Number of positions to probe per file (1-3).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Decode the entire file to validate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete files; only report corrupt ones.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("corrupt_audio.txt"),
        help="Path to write a report of corrupt files.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks while scanning.",
    )
    parser.add_argument(
        "--no-quiet-decode",
        action="store_true",
        help="Do not silence decoder stderr messages.",
    )
    parser.add_argument(
        "--fail-on-stderr",
        action="store_true",
        help="Treat any decoder stderr output as corruption.",
    )

    args = parser.parse_args()

    if not args.root.exists():
        parser.error(f"Root directory does not exist: {args.root}")

    exts = _normalize_exts(args.exts)
    root = args.root.resolve()
    quiet_decode = not args.no_quiet_decode
    probe_frames = max(1, int(args.probe_frames))
    probe_positions = max(1, int(args.probe_positions))
    full_scan = bool(args.full)
    fail_on_stderr = bool(args.fail_on_stderr)

    print(f"Scanning: {root}")
    print(f"Extensions: {', '.join(exts)}")
    print(f"Workers: {args.workers}")
    probe_desc = "full" if full_scan else f"{probe_positions}x{probe_frames} frames"
    print(f"Probe: {probe_desc}")
    print(f"Fail on stderr: {fail_on_stderr}")
    print(f"Audio backends: {', '.join(torchaudio.list_audio_backends())}")

    files = list(_iter_audio_files(root, exts, args.follow_symlinks))
    total = len(files)
    if total == 0:
        print("No matching audio files found.")
        return

    bad: List[Tuple[str, str]] = []

    if args.workers <= 0:
        _init_worker(
            quiet_decode,
            fail_on_stderr,
            probe_frames,
            probe_positions,
            full_scan,
        )
        for path in _progress(files, total):
            path_str, err = _check_file(str(path))
            if err:
                bad.append((path_str, err))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                quiet_decode,
                fail_on_stderr,
                probe_frames,
                probe_positions,
                full_scan,
            ),
        ) as executor:
            results = executor.map(_check_file, (str(p) for p in files), chunksize=args.chunk_size)
            for path_str, err in _progress(results, total):
                if err:
                    bad.append((path_str, err))

    print(f"Corrupt files found: {len(bad)}")

    if bad:
        try:
            with args.log.open("w", encoding="utf-8") as handle:
                for path_str, err in bad:
                    handle.write(f"{path_str}\t{err}\n")
            print(f"Report written to: {args.log}")
        except Exception as exc:
            print(f"Failed to write log: {exc}")

    if args.dry_run:
        print("Dry run enabled; no files deleted.")
        return

    removed = 0
    failed: List[Tuple[str, str]] = []
    for path_str, _ in bad:
        try:
            os.remove(path_str)
            removed += 1
        except Exception as exc:
            failed.append((path_str, f"{type(exc).__name__}: {exc}"))

    print(f"Deleted: {removed}")
    if failed:
        print(f"Failed to delete: {len(failed)} (see stderr for details)")
        for path_str, err in failed[:20]:
            print(f"{path_str} -> {err}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
