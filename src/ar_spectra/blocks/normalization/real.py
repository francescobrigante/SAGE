
# =============================================================================
# Classic normalization layers (LayerNorm, ConvLayerNorm) to stabilize real layers.
# =============================================================================

import torch
from torch import nn
import typing as tp
import einops
from complextorch.nn.modules.layernorm import LayerNorm as ComplexLayerNorm

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x

class ComplexWeightNorm(nn.Module):
    """Complex-valued weight normalization.

    This is a lightweight reparameterization similar to PyTorch's ``weight_norm``
    but adapted for complex tensors. Given a complex parameter ``weight`` we
    replace it by ``g * v / ||v||`` where ``g`` is a real-valued scaling factor
    (broadcastable over the chosen normalization dimension) and ``v`` is a
    complex tensor. The reconstruction happens in a forward pre-hook so the
    wrapped module always sees a correctly normalized complex weight.

    Differences vs torch.nn.utils.weight_norm:
      * Ensures the original weight is complex valued, raising an error otherwise.
      * Stores separate parameters ``weight_v`` (complex) and ``weight_g`` (real).
      * Provides attribute delegation so external code can access properties of
        the wrapped convolution (e.g. ``kernel_size``) transparently.

    Parameters
    ----------
    module : nn.Module
        Module containing a complex parameter named by ``name``. Typically a
        convolution layer with ``dtype=torch.complex64`` or higher precision.
    name : str, default 'weight'
        Name of the parameter inside ``module`` to reparameterize.
    dim : int, default 0
        Dimension along which to compute the vector norm.
    eps : float, default 1e-12
        Numerical stability epsilon added to the denominator.
    """
    def __init__(self, module: nn.Module, name: str = 'weight', dim: int = 0, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name, self.dim, self.eps = name, dim, eps

        # Retrieve existing weight and validate it is complex.
        w = getattr(self.module, self.name)
        if not torch.is_complex(w):
            raise TypeError("Weight must be complex (complex32/complex64/complex128).")

        # Create v (complex) and g (real) parameters.
        v = nn.Parameter(w.data)
        w_norm = torch.sqrt(torch.sum(w.data.real**2 + w.data.imag**2, dim=dim, keepdim=True))
        g = nn.Parameter(w_norm.real)  # real scaling, broadcastable shape

        # Replace original parameter with v/g pair.
        delattr(self.module, self.name)
        self.module.register_parameter(f'{self.name}_v', v)
        self.module.register_parameter(f'{self.name}_g', g)

        # Register pre-hook to reconstruct weight before every forward.
        self.module.register_forward_pre_hook(self._recompute_weight, with_kwargs=True)

    def _recompute_weight(self, mod, *args, **kwargs):  # noqa: D401 - internal hook
        v = getattr(mod, f'{self.name}_v')
        g = getattr(mod, f'{self.name}_g')
        denom = torch.sqrt(torch.sum(v.real**2 + v.imag**2, dim=self.dim, keepdim=True)).clamp_min(self.eps)
        w = g.to(v.dtype) * (v / denom)
        setattr(mod, self.name, w)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # Explicit property delegation to avoid fragile __getattr__ recursion.
    @property
    def kernel_size(self):
        return self.module.kernel_size

    @property
    def stride(self):
        return self.module.stride

    @property
    def padding(self):
        return self.module.padding

    @property
    def dilation(self):
        return self.module.dilation

    @property
    def in_channels(self):
        return self.module.in_channels

    @property
    def out_channels(self):
        return self.module.out_channels

    @property
    def groups(self):
        return self.module.groups

    def extra_repr(self) -> str:
        return f"ComplexWeightNorm(name={self.name}, dim={self.dim}, eps={self.eps})"

class ComplexConvLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 4D complex tensors.

    Expects inputs with shape ``(B, C, H, W)`` and ``dtype=torch.complex*``.
    Normalization is applied only across the channel dimension ``C`` and not
    over spatial dimensions, mimicking a per-channel normalization akin to a
    single-group GroupNorm but in the complex domain.
    """

    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = False) -> None:
        super().__init__()

        # ComplexLayerNorm di complextorch normalizza sulle *ultime* dims,
        # quindi facciamo un wrapper che sposta C in coda -> (B,H,W,C)
        self.ln = ComplexLayerNorm(
            normalized_shape=num_channels,
            eps=eps,
            elementwise_affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)  ->  permuta -> (B,H,W,C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()

        #LayerNorm sui canali complessi
        x_norm = self.ln(x_perm)

        # ritorna al layout conv-friendly (B,C,H,W) = (B, C, F, T)
        return x_norm.permute(0, 3, 1, 2).contiguous()

class ComplexConvLayerNorm1d(nn.Module):
    """Channel-wise LayerNorm for 3D complex tensors.

    Expects inputs with shape ``(B, C, T)`` and complex dtype. Applies
    normalization only across channels ``C`` (time axis preserved), equivalent
    to a single-group GroupNorm in the complex setting.
    """

    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = False) -> None:
        super().__init__()

        # ComplexLayerNorm normalizza sulle ultime dimensioni,
        # quindi spostiamo C in coda -> (B, T, C)
        self.ln = ComplexLayerNorm(
            normalized_shape=num_channels,
            eps=eps,
            elementwise_affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> permuta -> (B, T, C)
        x_perm = x.permute(0, 2, 1).contiguous()

        # LayerNorm sui canali complessi
        x_norm = self.ln(x_perm)

        # ritorna al layout conv-friendly (B, C, T)
        return x_norm.permute(0, 2, 1).contiguous()
