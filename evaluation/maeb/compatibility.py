#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/compatibility.py
# Environment shims that MUST run before the heavy imports (mteb / datasets /
# stable_audio_tools). Importing this module performs the import-time side
# effects (HF cache routing + datasets 'List' alias); the rest are functions
# the encoder calls before importing stable_audio_tools.
# =============================================================================
from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path
from typing import Any


def route_hf_cache() -> str | None:
    """Point HF_DATASETS_CACHE at $FAST/hf_cache/datasets (fallback $WORK).

    Must run before `datasets` is imported so the cache path is honored.
    Returns the resolved cache path, or None if neither $FAST nor $WORK is set.
    """
    cache = os.path.expandvars("$FAST/hf_cache/datasets")
    if cache == "$FAST/hf_cache/datasets":               # $FAST unset (local dev)
        cache = os.path.expandvars("$WORK/hf_cache/datasets")
    if cache == "$WORK/hf_cache/datasets":               # neither var resolved
        return None
    os.environ.setdefault("HF_DATASETS_CACHE", cache)
    return cache


def register_datasets_list_alias() -> None:
    """Alias the datasets 4.x 'List' feature type to 3.x 'Sequence'.

    A few MAEB datasets (e.g. JamALT) are authored with the 4.x schema; the
    pinned datasets 3.x cannot deserialize 'List' without this alias.
    """
    from datasets.features import features as _hf_features
    _hf_features._FEATURE_TYPES.setdefault("List", _hf_features.Sequence)


def patch_retrieval_config_names() -> None:
    """Fix get_dataset_config_names for offline reranking/retrieval tasks.

    Root cause: MTEB uses CamelCase HF paths (e.g. 'mteb/GTZANAudioReranking')
    but some datasets were cached under lowercase paths ('mteb/gtzan_audio_reranking').
    The datasets library's cache lookup does path.replace('/', '___') with NO
    case normalization, so the CamelCase path misses the lowercase cache.

    In offline mode, get_dataset_config_names falls back to returning ['default']
    (no network, no card) instead of the actual config subdirs. This makes
    RetrievalLoader._load_qrels look for config 'default', which doesn't exist.

    Fix: after the call returns ['default'], check whether the cache directory
    (reached via the path.replace('/', '___') symlink we created) actually
    contains named config subdirs, and return those instead.
    """
    import datasets as _datasets
    import datasets.inspect as _inspect
    import datasets.config as _dsconfig

    _orig_gcn = _inspect.get_dataset_config_names

    def _patched_gcn(path: str, revision: Any = None, *args: Any, **kwargs: Any) -> list[str]:
        try:
            result = _orig_gcn(path, revision=revision, *args, **kwargs)
        except (ConnectionError, OSError, Exception) as exc:
            if "offline" not in str(exc).lower() and "hub" not in str(exc).lower():
                raise
            result = ["default"]   # offline fallback; may be wrong — fix below

        if result == ["default"]:
            # Check the on-disk cache for actual config subdirectories.
            cache_root = os.environ.get("HF_DATASETS_CACHE",
                                        str(_dsconfig.HF_DATASETS_CACHE))
            ds_dir = os.path.join(cache_root, path.replace("/", "___"))
            if os.path.isdir(ds_dir):
                actual = [
                    d for d in os.listdir(ds_dir)
                    if os.path.isdir(os.path.join(ds_dir, d)) and not d.startswith(".")
                ]
                if actual and actual != ["default"]:
                    return actual
        return result

    # Patch both the submodule attribute AND the top-level package export.
    # MTEB's retrieval_dataset_loaders.py does `from datasets import get_dataset_config_names`
    # which binds to datasets.get_dataset_config_names at import time — so we must patch
    # the top-level reference BEFORE mteb is imported (compatibility.py is imported first).
    _inspect.get_dataset_config_names = _patched_gcn
    _datasets.get_dataset_config_names = _patched_gcn


def patch_num_proc(task: Any) -> None:
    """Wrap a task's dataset_transform() to swallow num_proc when it lacks it.

    Workaround for an mteb 2.12.30 bug: some task classes (VoxCelebSA,
    VoxPopuliLanguageID, CREMADClustering, VoxPopuliGenderClustering) override
    dataset_transform() WITHOUT a num_proc parameter, but the core loader calls
    dataset_transform(num_proc=...).
    """
    orig = task.dataset_transform
    sig = inspect.signature(orig)
    accepts = "num_proc" in sig.parameters or any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts:
        task.dataset_transform = lambda *a, num_proc=None, **k: orig()


# Heavy / unavailable optional deps of stable_audio_tools that the SAO-ACE
# autoencoder does NOT need for encoding — stubbed so `import` does not fail.
_STABLE_AUDIO_STUBS = (
    "k_diffusion", "laion_clap", "prefigure", "wandb", "gradio",
    "v_diffusion_pytorch", "local_attention", "vector_quantize_pytorch",
    "webdataset", "pytorch_lightning",
)


def stub_stable_audio_deps(names: tuple[str, ...] = _STABLE_AUDIO_STUBS) -> None:
    """Insert dummy modules so heavy optional stable_audio_tools deps import cleanly."""
    for name in names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = None  # type: ignore[attr-defined]
            sys.modules[name] = mod


def add_stable_audio_to_path(repo_root: Path) -> None:
    """Put the vendored stable_audio_baseline on sys.path so stable_audio_tools imports."""
    vendored = repo_root / "stable_audio_baseline"
    if vendored.is_dir() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))


def add_ar_spectra_to_path(repo_root: Path) -> None:
    """Put src/ on sys.path so ar_spectra/c_vae are importable in maeb_dl.

    Called only by swin_encoder.py at module load time (NOT at compatibility
    import time, so SAO jobs are unaffected).
    """
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# --- import-time side effects: cache routing MUST precede the datasets import ---
route_hf_cache()
register_datasets_list_alias()
patch_retrieval_config_names()
