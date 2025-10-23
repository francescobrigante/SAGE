import torch
import torch.nn as nn
import math

class ComplexWeightNorm(nn.Module):
    def __init__(self, module: nn.Module, name: str = 'weight', dim: int = 0, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name, self.dim, self.eps = name, dim, eps

        # get existing complex weight
        w = getattr(self.module, self.name)  # complex Parameter
        if not torch.is_complex(w):
            raise TypeError("Weight must be complex (complex32/complex64/complex128).")

        # build v and g like in torch
        v = nn.Parameter(w.data)
        # norm along dim with keepdim for broadcasting
        w_norm = torch.linalg.vector_norm(w.data, dim=dim, keepdim=True)
        g = nn.Parameter(w_norm.real)  # real, broadcastable shape

        # replace the parameter
        delattr(self.module, self.name)
        self.module.register_parameter(f'{self.name}_v', v)
        self.module.register_parameter(f'{self.name}_g', g)

        # register reconstruction pre-hook
        self.module.register_forward_pre_hook(self._recompute_weight, with_kwargs=True)

    def _recompute_weight(self, mod, *args, **kwargs):
        v = getattr(mod, f'{self.name}_v')          # complex
        g = getattr(mod, f'{self.name}_g')          # real
        denom = torch.linalg.vector_norm(v, dim=self.dim, keepdim=True).clamp_min(self.eps)
        w = g.to(v.dtype) * (v / denom)
        setattr(mod, self.name, w)
        return

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
