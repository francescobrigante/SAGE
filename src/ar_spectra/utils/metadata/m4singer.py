# ===================================================================
# M4Singer Metadata Provider
#
#   recursively scans the M4Singer wav tree into a canonical sorted list
# ===================================================================

"""
M4Singer file provider.

M4Singer ships no split metadata, so the canonical track order is simply the
sorted recursive listing of its ``.wav`` files (``m4singer/Singer#Song/NNNN.wav``).
A frozen ``filelist.txt`` at the audio root is honoured if present (via
``get_audio_filenames``). Sorted order is what the rotating sampler chunks on,
so it must be stable — hence the explicit ``sorted``.

Usage in config:
    custom_metadata_module: "src.ar_spectra.utils.metadata.m4singer"
    custom_metadata_kwargs: {}        # audio_dir alone is sufficient
"""

from typing import List

from ar_spectra.utils.file_scanning import get_audio_filenames


def get_audio_files(
    audio_dir: str,
    extensions: List[str] = (".wav",),
    **kwargs,
) -> List[str]:
    """Get the sorted list of M4Singer ``.wav`` paths under ``audio_dir``.

    Args:
        audio_dir: Root of the M4Singer audio tree (scanned recursively).
        extensions: Audio extensions to collect (M4Singer is all ``.wav``).

    Returns:
        Sorted list of absolute ``.wav`` file paths.
    """
    files = get_audio_filenames(audio_dir, exts=list(extensions))
    return sorted(files)
