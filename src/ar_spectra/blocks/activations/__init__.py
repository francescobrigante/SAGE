# =============================================================================
# Entrypoint and registry for activations, handles instantiation and eulero.nn alias.
# =============================================================================
from __future__ import annotations
import importlib
import sys
import types
import torch.nn as nn

from .relu  import (
    CELU, ModReLUScalar, ModReLU2d, ModReLU1d, ModReLU2dPerFreq,
    CReLU, zReLU, CardioidActivation, CPReLU, SplitReLU, magReLU,
)
from .gelu  import ComplexGELU1d, ComplexGELU2d, CGeLU
from .silu  import CSiLU, Abs_SiLU, CSwish
from .snake import Snake1d, snake, Snake1dComplex, snake_complex
from .misc  import CTanh, CGLU, CMish, CSoftplus, CCeLU


def get_activation(activation: str = None, is_complex: bool = False, channels=None, **kwargs):
    """Return an activation module by name.

    Args:
        activation: Class name string (e.g. ``"CReLU"``, ``"Snake"``). ``None`` → Identity.
        is_complex: If True, resolves from the ``eulero.nn`` namespace (complex activations).
        channels:   Required for channel-wise activations like ``Snake1d``.
        **kwargs:   Forwarded to the activation constructor.
    """
    if activation is None:
        return nn.Identity()

    name = activation

    if not is_complex:
        if name.lower() == "snake":
            assert channels is not None, "Snake activation requires `channels`."
            return Snake1d(channels=channels)
        try:
            return getattr(nn, name)(**kwargs)
        except AttributeError:
            return nn.Identity()
    else:
        try:
            complex_lib = importlib.import_module("eulero.nn")
        except Exception:
            complex_lib = sys.modules[__name__]  # fallback: self

        try:
            act_cls = getattr(complex_lib, name)
        except AttributeError:
            # Case-insensitive fallback
            matches = [attr for attr in dir(complex_lib) if attr.lower() == name.lower()]
            if not matches:
                return nn.Identity()
            act_cls = getattr(complex_lib, matches[0])

        if channels is not None and "channels" in act_cls.__init__.__code__.co_varnames:
            return act_cls(channels=channels, **kwargs)
        return act_cls(**kwargs)


# ── Register this module as ``eulero.nn`` for the complex activation namespace ─
_this = sys.modules[__name__]
_pkg  = sys.modules.setdefault("eulero", types.ModuleType("eulero"))
setattr(_pkg, "nn", _this)
sys.modules["eulero.nn"] = _this

__all__ = [
    # relu
    "CELU", "ModReLUScalar", "ModReLU2d", "ModReLU1d", "ModReLU2dPerFreq",
    "CReLU", "zReLU", "CardioidActivation", "CPReLU", "SplitReLU", "magReLU",
    # gelu
    "ComplexGELU1d", "ComplexGELU2d", "CGeLU",
    # silu
    "CSiLU", "Abs_SiLU", "CSwish",
    # snake
    "Snake1d", "snake", "Snake1dComplex", "snake_complex",
    # misc
    "CTanh", "CGLU", "CMish", "CSoftplus", "CCeLU",
    # factory
    "get_activation",
]
