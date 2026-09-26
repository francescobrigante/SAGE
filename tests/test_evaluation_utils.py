import pytest
import sys
from pathlib import Path


from evaluation.utils import collect_moisesdb_files

def test_collect_moisesdb_files(tmp_path: Path):
    # Create mock files
    files = [
        "track1__mixture.wav",
        "track1_vocals.wav",
        "track1_drums.wav",
        "track2__mixture.wav",
        "track2_bass.wav",
        "not_audio.txt"
    ]
    for f in files:
        (tmp_path / f).touch()
        
    # Test mixtures
    mixtures = collect_moisesdb_files(tmp_path, "mixtures", max_files=0)
    assert len(mixtures) == 2
    assert all(p.name.endswith("_mixture.wav") for p in mixtures)
    
    # Test stems
    stems = collect_moisesdb_files(tmp_path, "stems", max_files=0)
    assert len(stems) == 3
    assert not any(p.name.endswith("_mixture.wav") for p in stems)
    assert all(p.suffix == ".wav" for p in stems)
    
    # Test max_files
    stems_capped = collect_moisesdb_files(tmp_path, "stems", max_files=2)
    assert len(stems_capped) == 2

    # Test unknown split
    with pytest.raises(ValueError, match="Unknown moisesdb split"):
        collect_moisesdb_files(tmp_path, "unknown", max_files=0)
