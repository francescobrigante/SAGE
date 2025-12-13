import importlib
import torch
from torch import nn
import torch.nn.functional as F

# Ensure complex activation alias is registered on import
from . import complex_activations  

# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)



def get_activation(activation: str = None, is_complex: bool = False, channels=None, **kwargs):
    """Factory for activations.

    - If `is_complex=False`, returns a standard torch.nn activation or custom real one (e.g., Snake1d).
    - If `is_complex=True`, resolves the activation from the `eulero.nn` namespace (complex_activations).
    """
    if activation is None:
        return nn.Identity()
    name = activation
    if not is_complex:
        if name.lower() == "snake":
            assert channels is not None, "Snake activation requires `channels`."
            return Snake1d(channels=channels)
        try:
            act_cls = getattr(nn, name)
            return act_cls(**kwargs)
        except AttributeError:
            return nn.Identity()
    else:
        try:
            complex_lib = importlib.import_module("eulero.nn")
        except Exception:
            from . import complex_activations as complex_lib  # fallback

        try:
            act_cls = getattr(complex_lib, name)
        except AttributeError:
            # Gracefully handle case/casing mismatches (e.g., "CRelu" vs "CReLU").
            matches = [attr for attr in dir(complex_lib) if attr.lower() == name.lower()]
            if not matches:
                return nn.Identity()
            act_cls = getattr(complex_lib, matches[0])

        # Some complex activations require `channels`. Pass if available.
        if (channels is not None) and ("channels" in act_cls.__init__.__code__.co_varnames):
            return act_cls(channels=channels, **kwargs)
        return act_cls(**kwargs)

def _build_activation(name: str, ch: int, params: dict):
    if name is None:
        return nn.Identity()
    try:
        # get_activation handles 'snake' and delegates complex to eulero.nn
        return get_activation(name, channels=ch, **params)
    except (AttributeError, TypeError):
        # Safe fallback
        return nn.Identity()