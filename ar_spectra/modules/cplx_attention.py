import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ar_spectra.modules.normed_modules.conv import SConv1d, SConv2d
from ar_spectra.modules.normed_modules.conv import SConvTranspose1d, SConvTranspose2d, NormLinear
from ar_spectra.modules.cplx_dropout import ComplexDropout
import numpy as np

def _merge_heads(x: torch.Tensor, H: int):
    # x: (B*, H, L, D) -> (B*, L, H*D)
    return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], H * x.shape[3])

def _split_heads(x: torch.Tensor, H: int):
    # x: (B*, L, C) -> (B*, H, L, D)
    B_, L, C = x.shape
    D = C // H
    x = x.view(B_, L, H, D)
    return x.transpose(1, 2).contiguous()


def make_score_mod_from_mask(
    mask: Optional[torch.Tensor],
    B: int,
    L_q: int,
    L_k: int,
    device: torch.device,
):
    """Builds a score_mod callback for flex attention from a padding mask.

    Args:
        mask: Boolean-like tensor with shape ``(B, 1, L_k)`` or ``(B, L_q, L_k)``
            where ``1`` means keep and ``0`` means mask.
        B: Batch size used for validation.
        L_q: Query length.
        L_k: Key/Value length.
        device: Target device for the generated mask function.

    Returns:
        Callable that mirrors flex attention's ``score_mod`` signature and
        applies the provided padding mask uniformly across heads.
    """
    if mask is None:
        def score_mod(score, batch, head, q_idx, k_idx):
            return score
        return score_mod

    assert mask.dim() == 3, f"Expected mask dim 3, got {mask.dim()}"
    Bm, M1, Mk = mask.shape
    assert Bm == B and Mk == L_k, "Inconsistent mask shape"

    if M1 == 1:
        mask = mask.expand(B, L_q, L_k)
    elif M1 == L_q:
        pass
    else:
        raise ValueError(f"mask second dim must be 1 or L_q={L_q}, got {M1}")

    keep = (mask == 1).to(device=device)  # (B, L_q, L_k), bool

    def score_mod(score, b, h, q_idx, k_idx):
        cond = keep[b, q_idx, k_idx]
        return torch.where(cond, score, score.new_tensor(-float("inf")))
    return score_mod



class CMultiHeadedAttention(nn.Module):
    """Complex-valued multi-head attention accelerated with flex attention.

    The module keeps queries, keys, and values in the complex domain while
    leveraging high-performance real-valued ``flex_attention`` kernels. It does
    so by concatenating the real and imaginary parts of Q/K for scoring while
    propagating real and imaginary value channels independently, then
    reassembling the complex tensor before the output projection.
    """

    def __init__(self, n_head, n_feat, dropout_rate, is_complex: bool = True):
        super().__init__()
        self.h = n_head
        self.dropout_rate = dropout_rate
        self.is_complex = is_complex
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"
        self.d_k = n_feat // n_head
        
        self.linear_q = NormLinear(n_feat, n_feat, is_complex=is_complex)
        self.linear_k = NormLinear(n_feat, n_feat, is_complex=is_complex)
        self.linear_v = NormLinear(n_feat, n_feat, is_complex=is_complex)
        self.linear_out = NormLinear(n_feat, n_feat, is_complex=is_complex)
        self.dropout = ComplexDropout(p=dropout_rate)
        
    def forward_qkv(self, query, key, value):
        # query/key/value: (B, L, d_model) complex tensors
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        q = q.transpose(1, 2)  # (B, H, L_q, D_k)
        k = k.transpose(1, 2)  # (B, H, L_k, D_k)
        v = v.transpose(1, 2)  # (B, H, L_k, D_k)
        return q, k, v
        

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Runs complex-valued attention with optional padding mask.

        Args:
            query: Complex tensor of shape ``(B, L_q, d_model)``.
            key: Complex tensor of shape ``(B, L_k, d_model)``.
            value: Complex tensor of shape ``(B, L_k, d_model)``.
            mask: Optional keep/drop mask shaped ``(B, 1, L_k)`` or
                ``(B, L_q, L_k)``.
        """
        q, k, v = self.forward_qkv(query, key, value)  # (B, H, L_q/L_k, D)
        B_, H, L_q, D = q.shape
        _, _, L_k, _ = k.shape

        # convert complex embeddings into real tensors for scoring
        Qr = torch.cat([q.real, q.imag], dim=-1)  # (B, H, L_q, 2D)
        Kr = torch.cat([k.real, k.imag], dim=-1)  # (B, H, L_k, 2D)
        Vr = v.real                                # (B, H, L_k, D)
        Vi = v.imag                                # (B, H, L_k, D)

        # build score modifier shared across heads for padding mask
        score_mod = make_score_mod_from_mask(
            mask=mask,
            B=B_,
            L_q=L_q,
            L_k=L_k,
            device=Qr.device,
        )

        # scaled dot-product factor for doubled feature dimension
        scale = 1.0 / math.sqrt(2 * D)

        # run real-valued flex attention independently for real/imag
        Yr = flex_attention(
            Qr, Kr, Vr,
            score_mod=score_mod,
            block_mask=None,
            scale=scale,
        )
        Yi = flex_attention(
            Qr, Kr, Vi,
            score_mod=score_mod,
            block_mask=None,
            scale=scale,
        )

        # reconstruct complex tensor, merge heads, apply final projection
        Y = torch.complex(Yr, Yi)          # (B, H, L_q, D)
        Y = self.dropout(Y)                # ComplexDropout on the reconstructed tensor
        Y = _merge_heads(Y, self.h)        # (B, L_q, H*D = d_model)
        Y = self.linear_out(Y)             # (B, L_q, d_model) complex output

        return Y
