"""
File providers for custom dataset configurations.

Each provider module should implement:
    get_audio_files(audio_dir: str, **kwargs) -> list[str]

Where kwargs can include dataset-specific parameters like:
    - split: "training", "validation", "test"
    - subset: "small", "medium", "large"
    - metadata_csv: path to metadata file

Example provider:
    def get_audio_files(audio_dir, split="training", subset="small", **kwargs):
        # Load metadata, filter by split/subset, return file paths
        return ["/path/to/file1.mp3", "/path/to/file2.mp3"]
"""

import importlib
from typing import Callable, Optional


def load_file_provider_fn(module_path: Optional[str]) -> Optional[Callable]:
    """
    Dynamically load a get_audio_files function from a module path.
    
    Args:
        module_path: Dotted module path, e.g. "ar_spectra.fma_metadata"
                    The module must have a `get_audio_files(audio_dir, **kwargs)` function.
    
    Returns:
        The get_audio_files function, or None if module_path is None.
    """
    if module_path is None:
        return None
    
    module = importlib.import_module(module_path)
    if not hasattr(module, "get_audio_files"):
        raise AttributeError(
            f"Module '{module_path}' must define a 'get_audio_files(audio_dir, **kwargs)' function."
        )
    return module.get_audio_files


# Legacy alias for backwards compatibility
load_metadata_fn = load_file_provider_fn
