#!/usr/bin/env python3
# =============================================================================
# Shims for the MAEB stack (mteb / datasets) that MUST run before those
# packages are imported: importing this module registers the datasets 'List'
# alias and the offline config-name fix; patch_num_proc is applied per task.
# =============================================================================
from __future__ import annotations

import inspect
import os
from typing import Any


def register_datasets_list_alias() -> None:
    """Alias the datasets 4.x 'List' feature type to 3.x 'Sequence'.

    A few MAEB datasets (e.g. JamALT) are authored with the 4.x schema; the
    pinned datasets 3.x cannot deserialize 'List' without this alias.
    """
    from datasets.features import features as _hf_features
    if hasattr(_hf_features, "_FEATURE_TYPES"):
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


# --- import-time side effects (must precede the mteb / datasets imports) ---
register_datasets_list_alias()
patch_retrieval_config_names()
