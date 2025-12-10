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
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    
    
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

    def __init__(self, n_head, n_feat, dropout_rate, is_complex: bool = True, 
                 attention_dtype: Optional[torch.dtype] = torch.float32,
                 use_attn_dtype: Optional[bool] = False, bias: bool = True):
        super().__init__()
        self.h = n_head
        self.dropout_rate = dropout_rate
        self.is_complex = is_complex
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"
        self.d_k = n_feat // n_head
        self.use_attn_dtype = use_attn_dtype
        self.atten_dtype = attention_dtype

        
        self.linear_q = nn.Linear(n_feat, n_feat, bias=bias, dtype=torch.complex64)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=bias, dtype=torch.complex64)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=bias, dtype=torch.complex64)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=bias, dtype=torch.complex64)
        self.dropout = ComplexDropout(p=dropout_rate)
        self._init_weights()
        
    
    def _init_one_complex_linear(self, lin: nn.Linear, gain: float = 1.0):
        fan_in = lin.in_features
        fan_out = lin.out_features
        # complex Xavier: bound_real = sqrt(6/(fan_in+fan_out))
        # qui usiamo bound = bound_real / sqrt(2) = sqrt(3/(fan_in+fan_out))
        base_bound = math.sqrt(3.0 / (fan_in + fan_out))
        bound = gain * base_bound

        with torch.no_grad():
            lin.weight.real.uniform_(-bound, bound)
            lin.weight.imag.uniform_(-bound, bound)
            if lin.bias is not None:
                lin.bias.real.zero_()
                lin.bias.imag.zero_()

    def _init_weights(self):
        for lin in [self.linear_q, self.linear_k, self.linear_v, self.linear_out]:
            self._init_one_complex_linear(lin)

        
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
        
        base_float_dtype = query.real.dtype
        if not self.use_attn_dtype:
            self.atten_dtype = base_float_dtype

        # convert complex embeddings into real tensors for scoring
        Qr = torch.cat([q.real, q.imag], dim=-1).to(self.atten_dtype)  # (B, H, L_q, 2D)
        Kr = torch.cat([k.real, k.imag], dim=-1).to(self.atten_dtype)  # (B, H, L_k, 2D)
        Vr = v.real.to(self.atten_dtype)                                # (B, H, L_k, D)
        Vi = v.imag.to(self.atten_dtype)                                # (B, H, L_k, D)

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

        Yr = Yr.to(base_float_dtype)
        Yi = Yi.to(base_float_dtype)
        # reconstruct complex tensor, merge heads, apply final projection
        Y = torch.complex(Yr, Yi)         # (B, H, L_q, D)
        Y = self.dropout(Y)                # ComplexDropout on the reconstructed tensor
        Y = _merge_heads(Y, self.h)        # (B, L_q, H*D = d_model)
        Y = self.linear_out(Y)             # (B, L_q, d_model) complex output
        
        return Y


def print_cuda_mem(tag: str = ""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA non aavailable")
        return
    device = torch.device("cuda:0")
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free = free_bytes / 1024**2
    total = total_bytes / 1024**2
    print(
        f"[{tag}] allocated={alloc:.1f} MB, "
        f"reserved={reserved:.1f} MB, "
        f"free={free:.1f} MB / total={total:.1f} MB"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L_q, L_k, d_model, n_head = 64, 36, 36, 128, 4
    dropout = 0.1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attn = CMultiHeadedAttention(
        n_head=n_head,
        n_feat=d_model,
        dropout_rate=dropout,
        is_complex=True,
        attention_dtype=torch.bfloat16,  # se vuoi testare bf16
    ).to(device)

    flex_backend = "flex_attention (optimized kernels)" if flex_attention.__module__.startswith(
        "torch.nn.attention.flex_attention"
    ) else f"custom callable from {flex_attention.__module__}"
    ok(f"Attention backend detected: {flex_backend}")

    query = torch.randn(B, L_q, d_model, dtype=torch.complex64, device=device)
    key = torch.randn(B, L_k, d_model, dtype=torch.complex64, device=device)
    value = torch.randn(B, L_k, d_model, dtype=torch.complex64, device=device)

    mask = torch.ones(B, 1, L_k, dtype=torch.int32, device=device)
    mask[:, :, -2:] = 0  # mask last two key positions to test padding logic

    # opzionale: pulisci picco precedente
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    print_cuda_mem("before attn")

    output = attn(query, key, value, mask=mask)

    print_cuda_mem("after attn")
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        print(f"[peak] allocated={peak_alloc:.1f} MB, reserved={peak_reserved:.1f} MB")

    ok(f"Output dtype: {output.dtype}")
    ok(f"Output shape: {output.shape}")
    
    wq = attn.linear_q.weight
    print("linear_q weight real std:", wq.real.std().item())
    print("linear_q weight imag std:", wq.imag.std().item())
    print("|w| mean:", (wq.abs()).mean().item())