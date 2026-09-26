from __future__ import annotations

# =============================================================================
# Hydra instantiation of model components and export of their specs into checkpoints.
# =============================================================================

import importlib
import json
from copy import deepcopy
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate



def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def _locate_class(class_path: Union[str, type]) -> type:
    """Supports either a string path 'pkg.mod.Class' or a class already passed."""
    if not isinstance(class_path, str):
        return class_path
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _instantiate_component(spec: Dict[str, Any]) -> Any:
    """Instantiate a component from a Hydra-style config dict.
    
    Supports both new Hydra format (_target_) and converts legacy format (class/kwargs)
    for backwards compatibility with existing checkpoints.
    """
    if spec is None:
        return None
    
    # If already has _target_, use hydra directly
    if "_target_" in spec:
        return hydra_instantiate(spec, _convert_="all")
    
    # Legacy format conversion: class/kwargs -> _target_
    if "class" in spec:
        converted = {"_target_": spec["class"]}
        kwargs = spec.get("kwargs", {}) or {}
        converted.update(kwargs)
        return hydra_instantiate(converted, _convert_="all")
    
    raise ValueError("spec must contain either '_target_' (Hydra) or 'class' (legacy) key.")

def _class_path(obj: Union[str, type, nn.Module]) -> str:
    if isinstance(obj, str):
        return obj
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"

def _canonicalize_module_spec(module_like: Union[nn.Module, Dict[str, Any], str, type, None]) -> Optional[Dict[str, Any]]:
    """Convert encoder/decoder references into exportable configuration dicts.

    Training code may build the autoencoder by passing instantiated modules,
    fully-qualified class names, or already-normalised spec dictionaries.  When
    we later embed the architecture inside checkpoints we need a consistent
    representation so inference can rebuild the same modules.  This helper
    performs that normalisation while also preserving lightweight metadata such
    as ``target_channels`` when it is exposed by the module instance.
    """
    if module_like is None:
        return None
    if isinstance(module_like, dict):
        return deepcopy(module_like)

    class_path = _class_path(module_like)
    spec: Dict[str, Any] = {"class": class_path}
    if isinstance(module_like, nn.Module):
        extra_kwargs: Dict[str, Any] = {}
        if hasattr(module_like, "target_channels"):
            extra_kwargs["target_channels"] = getattr(module_like, "target_channels")
        if extra_kwargs:
            spec["kwargs"] = extra_kwargs
    return spec

def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON export produced by ``export_model_config`` or ``build_from_json``.

    Keeping this helper alongside the model makes it straightforward for
    tooling such as ``fast_inference`` to hydrate an autoencoder from a saved
    configuration file without duplicating JSON parsing code elsewhere.
    """
    with open(path, "r") as f:
        return json.load(f)
