# ===================================================================
# File Scanning Utilities
#
#   contains fast directory traversal for finding audio files
# ===================================================================

import os
from typing import Union, List
from config import DEFAULT_AUDIO_EXTENSIONS

def fast_scandir(dir: str, ext: list) -> tuple[list[str], list[str]]:
    """
    Very fast glob alternative for recursively scanning directories.
    From https://stackoverflow.com/a/59803793/4259243
    """
    subfolders, files = [], []
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    
    try:
        for f in os.scandir(dir):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")
                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except OSError:
                pass
    except OSError:
        pass

    for subdir in list(subfolders):
        sf, f = fast_scandir(subdir, ext)
        subfolders.extend(sf)
        files.extend(f)
    
    return subfolders, files

def get_audio_filenames(
    paths: Union[str, List[str]],
    exts: list = None,
) -> list[str]:
    """Recursively get a list of audio filenames from directories."""
    if exts is None:
        exts = list(DEFAULT_AUDIO_EXTENSIONS)
    
    filenames = []
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        # Check for filelist.txt at the root of the directory
        filelist_path = os.path.join(path, "filelist.txt")
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = [os.path.join(path, line.strip()) for line in f if line.strip()]
                filenames.extend(files)
            continue
        
        _, files = fast_scandir(path, exts)
        filenames.extend(files)
    
    return filenames
