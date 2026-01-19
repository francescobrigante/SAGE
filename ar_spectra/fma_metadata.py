"""
FMA Dataset file provider.

Provides file lists based on FMA tracks.csv metadata,
filtering by split (training/validation/test) and subset (small/medium/large).

Usage in config:
    custom_metadata_module: "ar_spectra.fma_metadata"
    custom_metadata_kwargs:
      split: training
      subset: small
      metadata_csv: /path/to/tracks.csv  # optional, uses default if not specified
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, List

# Default path - can be overridden via kwargs
DEFAULT_METADATA_PATH = "/home/ec2-user/cerovaz/data/fma_metadata/tracks.csv"


@lru_cache(maxsize=1)
def _load_tracks_df(csv_path: str):
    """Load and cache the FMA tracks DataFrame."""
    import pandas as pd
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FMA metadata not found at: {csv_path}")
    
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    return df


def get_audio_files(
    audio_dir: str,
    split: str = "training",
    subset: Optional[str] = "small",
    metadata_csv: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Get list of audio files for a specific FMA split/subset.
    
    Args:
        audio_dir: Root directory of FMA audio files
        split: One of "training", "validation", "test"
        subset: One of "small", "medium", "large", or None for all
        metadata_csv: Path to tracks.csv (uses default if not specified)
    
    Returns:
        List of absolute file paths
    """
    csv_path = metadata_csv or DEFAULT_METADATA_PATH
    df = _load_tracks_df(csv_path)
    
    # Filter by split
    mask = df[("set", "split")] == split
    
    # Optionally filter by subset
    if subset is not None:
        mask &= df[("set", "subset")] == subset
    
    track_ids = df[mask].index.tolist()
    
    # Build file paths
    files = []
    audio_dir = Path(audio_dir)
    
    for tid in track_ids:
        # FMA naming: track 2 -> 000/000002.mp3
        folder = f"{tid:06d}"[:3]
        filename = f"{tid:06d}.mp3"
        filepath = audio_dir / folder / filename
        
        if filepath.exists():
            files.append(str(filepath))
    
    return files
