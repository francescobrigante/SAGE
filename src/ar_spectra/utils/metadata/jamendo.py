# ===================================================================
# MTG-Jamendo Metadata Provider
#
#   parses an autotagging split TSV and returns the train audio paths
# ===================================================================

"""
MTG-Jamendo file provider.

Reads an official split TSV (e.g. split-0/autotagging-train.tsv) and maps the
`PATH` column (``XX/ID.mp3``) onto the audio root, returning absolute paths for
the leakage-free training split. Mirrors the FMA provider contract.

Usage in config:
    custom_metadata_module: "src.ar_spectra.utils.metadata.jamendo"
    custom_metadata_kwargs:
      split_tsv: ${config:JAMENDO_SPLIT_TSV}   # optional, defaults to JAMENDO_SPLIT_TSV
"""

import os
from pathlib import Path
from typing import List, Optional

from config import JAMENDO_SPLIT_TSV

# Column layout of the autotagging TSVs (tab-separated, one header row):
#   TRACK_ID  ARTIST_ID  ALBUM_ID  PATH  DURATION  TAGS...
_PATH_COL = 3


def get_audio_files(
    audio_dir: str,
    split_tsv: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Get the sorted list of MTG-Jamendo audio paths for a split TSV.

    Args:
        audio_dir: Root directory of the Jamendo audio (``XX/ID.mp3`` layout).
        split_tsv: Path to an autotagging split TSV; defaults to ``JAMENDO_SPLIT_TSV``.

    Returns:
        Sorted list of absolute ``.mp3`` file paths (one per track in the split).
    """
    tsv_path = split_tsv or JAMENDO_SPLIT_TSV
    if tsv_path is None or not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Jamendo split TSV not found at: {tsv_path}")

    audio_root = Path(audio_dir)
    files: List[str] = []
    with open(tsv_path, "r", encoding="utf-8") as fh:
        next(fh, None)  # skip header row
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            rel_path = fields[_PATH_COL]               # e.g. "14/214.mp3"
            files.append(str(audio_root / rel_path))

    return sorted(files)
