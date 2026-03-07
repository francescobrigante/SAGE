"""Normalization modules.

This file provides a set of normalization layers for both real and complex valued
signals, including convolution-friendly LayerNorm variants, complex reparameterized
weight normalization, complex LayerNorm wrappers for 1D/2D convolution layouts,
Group Normalization with per-group complex whitening, and BatchNorm variants.

Key goals:
    * Preserve convolutional layouts while applying per-channel normalization.
    * Offer complex-valued analogs of common real-valued normalization strategies.
    * Provide numerically stable whitening (inverse square root) of 2x2 covariance
        matrices (real/imag parts) used in complex normalization.

All docstrings are written in English for clarity.
"""

import typing as tp

import einops
import torch
from torch import nn
import complextorch as cplx
from complextorch.nn.modules.layernorm import LayerNorm as ComplexLayerNorm
import torch.nn.init as init


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
    
    
class ComplexGroupNorm(nn.Module):
    """Complex Group Normalization with per-group whitening.

    Supports inputs of shape ``(B, C, F, T)`` or ``(B, C, T)`` (treated as
    ``F=1``). For each group of channels we compute the mean and 2x2 real/imag
    covariance, derive an inverse square root (whitening) transform, and apply
    it to the centered real and imaginary parts.

    Parameters
    ----------
    num_channels : int
        Total number of channels ``C``.
    num_groups : int
        Number of groups ``G`` (``C`` must be divisible by ``G``).
    eps : float, default 1e-4
        Numerical stability term added to covariance diagonals.
    affine : bool, default True
        Whether to learn an affine transform post-whitening.
    reduce_spatial : bool, default True
        If True, statistics are aggregated over channel(s) in group and all
        spatial dimensions. If False, only over channels (classic GroupNorm).
    """
    def __init__(self,
                 num_channels: int,
                 num_groups: int,
                 eps: float = 1e-4,
                 affine: bool = True,
                 reduce_spatial: bool = True):
        super().__init__()
        assert num_channels > 0
        assert 1 <= num_groups <= num_channels and num_channels % num_groups == 0, \
            "num_groups deve dividere num_channels"
        self.C = num_channels
        self.G = num_groups
        self.eps = eps
        self.reduce_spatial = reduce_spatial

        if affine:
            self.weight = nn.Parameter(torch.empty(num_channels, 3))
            self.bias   = nn.Parameter(torch.empty(num_channels, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)

    @torch.no_grad()
    def _safe_inv_sqrt_params(self, Crr, Cii, Cri):
        det = Crr * Cii - Cri * Cri
        det = torch.clamp(det, min=0.0)
        s = torch.sqrt(det + 1e-12)
        t = torch.sqrt(torch.clamp(Cii + Crr + 2.0 * s, min=1e-12))
        denom = torch.clamp(s * t, min=self.eps)
        inv_st = 1.0 / denom
        Rrr = (Cii + s) * inv_st
        Rii = (Crr + s) * inv_st
        Rri = -Cri * inv_st
        return Rrr, Rii, Rri

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x), "Atteso dtype complesso"
        original_3d = False
        if x.dim() == 3:
            # (B,C,T) -> (B,C,1,T)
            x = x.unsqueeze(2)
            original_3d = True
        elif x.dim() != 4:
            raise ValueError(f"Shape non supportata: {x.shape}")

        B, C, F, T = x.shape
        G = self.G
        Cg = C // G

        xg = x.view(B, G, Cg, F, T)

        if self.reduce_spatial:
            reduce_dims = (2, 3, 4)  # canali gruppo + F + T (F=1 se era 3D)
            N = Cg * F * T
        else:
            reduce_dims = (2,)
            N = Cg

        mean_r = xg.real.mean(dim=reduce_dims, keepdim=True)
        mean_i = xg.imag.mean(dim=reduce_dims, keepdim=True)
        mean = torch.complex(mean_r, mean_i)
        xg = xg - mean

        r = xg.real
        i = xg.imag

        Crr = (r.pow(2).sum(dim=reduce_dims, keepdim=True) / float(N)) + self.eps
        Cii = (i.pow(2).sum(dim=reduce_dims, keepdim=True) / float(N)) + self.eps
        Cri = (r.mul(i).sum(dim=reduce_dims, keepdim=True) / float(N))

        with torch.no_grad():
            Rrr, Rii, Rri = self._safe_inv_sqrt_params(Crr, Cii, Cri)

        r_wh = Rrr * r + Rri * i
        i_wh = Rri * r + Rii * i
        x_wh = torch.complex(r_wh, i_wh)

        y = x_wh.view(B, C, F, T)

        if self.weight is not None:
            w_rr = self.weight[:, 0].view(1, C, 1, 1)
            w_ii = self.weight[:, 1].view(1, C, 1, 1)
            w_ri = self.weight[:, 2].view(1, C, 1, 1)
            b_r  = self.bias[:, 0].view(1, C, 1, 1)
            b_i  = self.bias[:, 1].view(1, C, 1, 1)
            yr = y.real
            yi = y.imag
            y = torch.complex(
                w_rr * yr + w_ri * yi + b_r,
                w_ri * yr + w_ii * yi + b_i
            )

        if original_3d:
            y = y.squeeze(2)  # ritorna a (B,C,T)
        return y

from complexPyTorch.complexLayers import _ComplexBatchNorm
from complexPyTorch.complexLayers import ComplexBatchNorm2d
class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, inp):
        """Complex BatchNorm aggregating over all non-channel dimensions.

        Accepts inputs:
          * ``(B, C)``   : batch of vectors.
          * ``(B, C, T...)`` : batch of sequences with one or more additional
            temporal/spatial dimensions. All non-channel dimensions are folded
            into the batch for statistics computation. Channel dimension is
            normalized with complex whitening of its 2x2 covariance.
        """
        original_shape = inp.shape
        if inp.dim() < 2:
            raise ValueError(f"Expected tensor with >=2 dims, got shape={original_shape}")

        # If more than 2D, move channels last then flatten remaining spatial dims.
        if inp.dim() > 2:
            permute_order = (0, *range(2, inp.dim()), 1)  # (B, C, T1, T2, ...) -> (B, T1, T2, ..., C)
            x = inp.permute(permute_order)
            flat_batch = int(torch.prod(torch.tensor(x.shape[:-1])))
            x = x.reshape(flat_batch, self.num_features)
        else:
            x = inp

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        # Mean
        if self.training or (not self.track_running_stats):
            mean_r = x.real.mean(dim=0).type(torch.complex64)
            mean_i = x.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        x = x - mean[None, :]

        # Covariance (real-real, imag-imag, real-imag)
        if self.training or (not self.track_running_stats):
            n = x.size(0)
            Crr = x.real.var(dim=0, unbiased=False) + self.eps
            Cii = x.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (x.real * x.imag).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]
            n = x.size(0)

        if self.training and self.track_running_stats:
            with torch.no_grad():
                corr = n / (n - 1) if n > 1 else 1.0
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * corr
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )
                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * corr
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )
                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * corr
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # Whitening transform
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        x_whiten = (Rrr[None, :] * x.real + Rri[None, :] * x.imag).type(torch.complex64) \
                   + 1j * (Rii[None, :] * x.imag + Rri[None, :] * x.real).type(torch.complex64)

        if self.affine:
            x_aff = (
                self.weight[None, :, 0] * x_whiten.real
                + self.weight[None, :, 2] * x_whiten.imag
                + self.bias[None, :, 0]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2] * x_whiten.real
                + self.weight[None, :, 1] * x_whiten.imag
                + self.bias[None, :, 1]
            ).type(torch.complex64)
        else:
            x_aff = x_whiten

        # Restore original shape if there were extra spatial dims.
        if inp.dim() > 2:
            new_shape = (*original_shape[:1], *original_shape[2:], self.num_features)
            x_aff = x_aff.view(new_shape)  # (B, T1, T2, ..., C)
            # Move channel back to second position: (B, C, T1, T2, ...)
            x_aff = x_aff.permute(0, -1, *range(1, x_aff.dim()-1))

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return x_aff