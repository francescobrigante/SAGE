"""
FMA Dataset metadata provider.

Loads track metadata from the FMA tracks.csv file and provides
genre and other info for each audio file.

Usage in config:
    custom_metadata_module: "ar_spectra.fma_metadata"
    
Requires FMA_METADATA_CSV environment variable or default path.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Default path - can be overridden via environment variable
DEFAULT_METADATA_PATH = "/home/ec2-user/cerovaz/data/fma_metadata/tracks.csv"


@lru_cache(maxsize=1)
def _load_tracks_df():
    """Load and cache the FMA tracks DataFrame."""
    import pandas as pd
    
    csv_path = os.environ.get("FMA_METADATA_CSV", DEFAULT_METADATA_PATH)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FMA metadata not found at: {csv_path}")
    
    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    return df


def get_track_id_from_path(path: str) -> Optional[int]:
    """
    Extract track_id from FMA file path.
    
    FMA files are named like: 000/000002.mp3 -> track_id = 2
    """
    filename = Path(path).stem  # e.g., "000002"
    try:
        return int(filename)
    except ValueError:
        return None


def get_custom_metadata(info: dict, audio) -> dict:
    """
    Get metadata for an FMA track.
    
    Args:
        info: Dict with "path" and "relpath" keys
        audio: Audio tensor (unused but required by interface)
    
    Returns:
        Dict with genre, title, split, and other FMA metadata
    """
    track_id = get_track_id_from_path(info["path"])
    
    if track_id is None:
        return {"genre": "unknown", "title": info["relpath"]}
    
    try:
        df = _load_tracks_df()
        
        if track_id not in df.index:
            return {"genre": "unknown", "title": info["relpath"], "track_id": track_id}
        
        row = df.loc[track_id]
        
        genre = row[("track", "genre_top")]
        if isinstance(genre, float):  # NaN check
            genre = "unknown"
        
        title = row[("track", "title")]
        if isinstance(title, float):
            title = str(track_id)
        
        return {
            "track_id": track_id,
            "genre": str(genre),
            "title": str(title),
            "split": str(row[("set", "split")]),
            "subset": str(row[("set", "subset")]),
        }
    
    except Exception:
        return {"genre": "unknown", "title": info["relpath"], "track_id": track_id}


def get_fma_split_files(
    audio_dir: str,
    metadata_csv: str,
    split: str = "training",
    subset: Optional[str] = "small",
) -> list[str]:
    """
    Get list of audio files for a specific FMA split.
    
    Args:
        audio_dir: Root directory of FMA audio files
        metadata_csv: Path to tracks.csv
        split: One of "training", "validation", "test"
        subset: One of "small", "medium", "large", or None for all
    
    Returns:
        List of absolute file paths
    """
    import pandas as pd
    
    df = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    
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
