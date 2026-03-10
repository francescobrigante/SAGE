
# =============================================================================
# Normalization layers adapted to operate on complex tensors while respecting phase.
# =============================================================================

import torch
from torch import nn
import torch.nn.init as init
from complexPyTorch.complexLayers import _ComplexBatchNorm, ComplexBatchNorm2d

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
