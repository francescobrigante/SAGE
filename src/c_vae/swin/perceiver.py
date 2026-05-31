import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ar_spectra.blocks.conv.normed import NormLinear

class HermitianCrossAttention(nn.Module):
    """
    Global Cross-Attention using Hermitian Cosine scoring for complex representations,
    extracted from WindowAttention logic but without spatial window constraints.
    """
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., is_complex=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.is_complex = is_complex

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        self.q_proj = NormLinear(dim, dim, bias=False, is_complex=is_complex)
        self.k_proj = NormLinear(dim, dim, bias=False, is_complex=is_complex)
        self.v_proj = NormLinear(dim, dim, bias=False, is_complex=is_complex)

        _bias_dtype = torch.complex64 if is_complex else torch.float32
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
            self.k_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
            self.v_bias = nn.Parameter(torch.zeros(dim, dtype=_bias_dtype))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = NormLinear(dim, dim, bias=True, is_complex=is_complex)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in, k_in, v_in):
        B, N_q, C = q_in.shape
        _, N_k, _ = k_in.shape

        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)

        if self.q_bias is not None:
            q = q + self.q_bias
            k = k + self.k_bias
            v = v + self.v_bias

        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.is_complex:
            q_norm = q / q.abs().clamp(min=1e-6)
            k_norm = k / k.abs().clamp(min=1e-6)
            attn = q_norm.real @ k_norm.real.mT + q_norm.imag @ k_norm.imag.mT
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        if self.is_complex:
            out_re = (attn @ v.real).to(torch.float32)
            out_im = (attn @ v.imag).to(torch.float32)
            x = torch.complex(out_re, out_im)
        else:
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceiverCompression(nn.Module):
    def __init__(self, dim, num_queries, grid_size, num_heads=16, is_complex=False):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.grid_size = grid_size
        self.is_complex = is_complex
        
        H, W = grid_size
        _dtype = torch.complex64 if is_complex else torch.float32
        
        # Absolute 2D PE for Swin tokens (keys/values)
        self.abs_pe = nn.Parameter(torch.randn(H * W, dim, dtype=_dtype) * 0.02)
        
        # Abstract queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim, dtype=_dtype) * 0.02)
        
        self.cross_attn = HermitianCrossAttention(dim, num_heads=num_heads, is_complex=is_complex)
        self.self_attn = HermitianCrossAttention(dim, num_heads=num_heads, is_complex=is_complex)
        
    def forward(self, x):
        # x: (B, H*W, C)
        B = x.shape[0]
        
        # Add Absolute PE to Swin tokens
        kv = x + self.abs_pe.unsqueeze(0)
        
        # Expand queries to batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross Attention: Queries read from Swin Tokens
        out = self.cross_attn(q, kv, kv)
        
        # Self Attention: Refine the queries
        out = out + self.self_attn(out, out, out)
        
        return out


class PerceiverDecompression(nn.Module):
    def __init__(self, dim, num_queries, grid_size, num_heads=16, is_complex=False):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.grid_size = grid_size
        self.is_complex = is_complex
        
        H, W = grid_size
        _dtype = torch.complex64 if is_complex else torch.float32
        
        # Output queries living on the Swin grid space
        self.out_queries = nn.Parameter(torch.randn(H * W, dim, dtype=_dtype) * 0.02)
        self.abs_pe = nn.Parameter(torch.randn(H * W, dim, dtype=_dtype) * 0.02)
        
        self.self_attn = HermitianCrossAttention(dim, num_heads=num_heads, is_complex=is_complex)
        self.cross_attn = HermitianCrossAttention(dim, num_heads=num_heads, is_complex=is_complex)
        
    def forward(self, x):
        # x: (B, num_latents, C)
        B = x.shape[0]
        
        # Self Attention: Refine decoded tokens before expanding
        kv = x + self.self_attn(x, x, x)
        
        # Expand output queries to batch and add their PE
        q = self.out_queries.unsqueeze(0).expand(B, -1, -1) + self.abs_pe.unsqueeze(0)
        
        # Cross Attention: Grid queries read from refined decoded latents
        out = self.cross_attn(q, kv, kv)
        
        return out
