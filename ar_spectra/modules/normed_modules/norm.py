"""Normalization modules."""

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


class ComplexConvLayerNorm2d(nn.Module):
    """
    LayerNorm “conv-friendly” per tensori complessi 4-D.

    • Input atteso:(B, C, H, W), dtype=torch.complex64
    • Normalizza **solo** l’asse dei canali C, lasciando invariati H e W
      (equivalente a GroupNorm con g = 1 ma in algebra complessa).
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
    """
    LayerNorm “conv-friendly” per tensori complessi 3-D.

    • Input atteso: (B, C, T), dtype=torch.complex64
    • Normalizza solo l’asse dei canali C, lasciando invariato T
      (equivalente a GroupNorm con g = 1 ma in algebra complessa).
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
    
    
class ComplexGroupNorm2d(nn.Module):

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
            # stessa forma del BN complesso: 3 pesi reali per canale (w_rr, w_ii, w_ri),
            # e 2 bias (b_r, b_i)
            self.weight = nn.Parameter(torch.empty(num_channels, 3))
            self.bias   = nn.Parameter(torch.empty(num_channels, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            # diag ~ sqrt(2), offdiag (ri) = 0; bias = 0 (come nel tuo BN)
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)

    @torch.no_grad()
    def _safe_inv_sqrt_params(self, Crr, Cii, Cri):
        # calcolo robusto dei parametri dell'inverse sqrt 2x2
        det = Crr * Cii - Cri * Cri
        det = torch.clamp(det, min=0.0)  # numerica
        s = torch.sqrt(det + 1e-12)
        t = torch.sqrt(torch.clamp(Cii + Crr + 2.0 * s, min=1e-12))
        denom = s * t
        denom = torch.clamp(denom, min=self.eps)

        inv_st = 1.0 / denom
        Rrr = (Cii + s) * inv_st
        Rii = (Crr + s) * inv_st
        Rri = -Cri * inv_st
        return Rrr, Rii, Rri

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T), dtype=torch.complex64
        """
        assert torch.is_complex(x), "Atteso dtype complesso"
        B, C, F, T = x.shape
        G = self.G
        Cg = C // G

        # (B, G, Cg, F, T)
        xg = x.view(B, G, Cg, F, T)

        # media per-sample per-gruppo (su canali del gruppo e, opzionalmente, su F,T)
        if self.reduce_spatial:
            reduce_dims = (2, 3, 4)  # Cg, F, T
        else:
            reduce_dims = (2,)       # solo canali del gruppo

        mean_r = xg.real.mean(dim=reduce_dims, keepdim=True).to(torch.complex64)
        mean_i = xg.imag.mean(dim=reduce_dims, keepdim=True).to(torch.complex64)
        mean = mean_r + 1j * mean_i
        xg = xg - mean  # broadcast su (B,G,1,1,1)

        # covarianza Re/Im per gruppo
        r = xg.real
        i = xg.imag

        if self.reduce_spatial:
            N = Cg * F * T
        else:
            N = Cg

        # medie (biased) come in GN/IN
        Crr = (r.pow(2).sum(dim=reduce_dims, keepdim=True) / float(N)) + self.eps
        Cii = (i.pow(2).sum(dim=reduce_dims, keepdim=True) / float(N)) + self.eps
        Cri = (r.mul(i).sum(dim=reduce_dims, keepdim=True) / float(N))

        # inverse sqrt della covarianza (per gruppo)
        # shape: (B,G,1,1,1)
        with torch.no_grad():
            Rrr, Rii, Rri = self._safe_inv_sqrt_params(Crr, Cii, Cri)

        # applica whiten (broadcast su (Cg,F,T))
        r_wh = Rrr * r + Rri * i
        i_wh = Rri * r + Rii * i
        x_wh = torch.complex(r_wh, i_wh)

        # ricomponi (B,C,F,T)
        y = x_wh.view(B, C, F, T)

        # affine per-canale con mixing Re/Im, come nel tuo BN
        if self.weight is not None:
            w_rr = self.weight[:, 0].view(1, C, 1, 1)
            w_ii = self.weight[:, 1].view(1, C, 1, 1)
            w_ri = self.weight[:, 2].view(1, C, 1, 1)  # off-diagonale condivisa
            b_r  = self.bias[:, 0].view(1, C, 1, 1)
            b_i  = self.bias[:, 1].view(1, C, 1, 1)

            yr = y.real
            yi = y.imag
            y = torch.complex(
                w_rr * yr + w_ri * yi + b_r,
                w_ri * yr + w_ii * yi + b_i
            )

        return y

