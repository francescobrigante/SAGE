"""
Metadata providers for custom dataset configurations.

Each provider module should implement:
    get_custom_metadata(info: dict, audio: torch.Tensor) -> dict

Where `info` contains at minimum:
    - "path": absolute path to the audio file
    - "relpath": relative path from audio_dir

Example provider:
    def get_custom_metadata(info, audio):
        return {"prompt": info["relpath"]}
"""

import importlib
from typing import Callable, Optional


def load_metadata_fn(module_path: Optional[str]) -> Optional[Callable]:
    """
    Dynamically load a get_custom_metadata function from a module path.
    
    Args:
        module_path: Dotted module path, e.g. "my_project.metadata.jamendo"
                    The module must have a `get_custom_metadata(info, audio)` function.
    
    Returns:
        The get_custom_metadata function, or None if module_path is None.
    """
    if module_path is None:
        return None
    
    module = importlib.import_module(module_path)
    if not hasattr(module, "get_custom_metadata"):
        raise AttributeError(
            f"Module '{module_path}' must define a 'get_custom_metadata(info, audio)' function."
        )
    return module.get_custom_metadata


# --- Built-in metadata providers ---

def get_relpath_metadata(info: dict, audio) -> dict:
    """Default: use relative path as prompt."""
    return {"prompt": info.get("relpath", info.get("path", ""))}


def get_filename_metadata(info: dict, audio) -> dict:
    """Use filename (without extension) as prompt."""
    import os
    path = info.get("path", "")
    return {"prompt": os.path.splitext(os.path.basename(path))[0]}
