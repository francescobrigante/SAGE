# ===============================================================
# Swin Transformer V2: window-based multi-head self-attention.
# Implements cosine attention with learnable per-head temperature
# (logit_scale) and continuous relative position bias via a small
# Log-CPB MLP on log-transformed coordinates.
# No internal package dependencies.
# ===============================================================

#TODO explain w-MSA vs SW-MSA and how mask is used for cyclic shift in SW-MSA

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA / SW-MSA) with Swin V2 cosine bias.

    Args:
        dim: Token channel dimension.
        window_size: (Wh, Ww) height and width of the attention window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add learnable bias to Q and V projections (K bias omitted, V2 convention).
        attn_drop: Dropout rate on attention weights.
        proj_drop: Dropout rate on the output projection.
        pretrained_window_size: Window size used during pre-training for Log-CPB coordinate
            normalisation. None or [0, 0] falls back to window_size.
    """

    def __init__(
        self,
        dim: int,
        window_size,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size=None,
    ) -> None:
        
        super().__init__()
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]          # no pre-training transfer by default

        self.dim = dim
        self.window_size = window_size                                  # (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Per-head learnable temperature, initialised to log(10) so scale ≈ 10 at start
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # Log-CPB MLP: 2-D coordinate → 512 hidden → num_heads bias values
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # -----------------------------------------------------------------
        # Relative coordinate table (1, 2Wh-1, 2Ww-1, 2) — normalised + log-scaled
        # -----------------------------------------------------------------
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0).contiguous().unsqueeze(0)                # (1, 2Wh-1, 2Ww-1, 2)
        )
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            
        relative_coords_table *= 8                                     # normalise to [-8, 8]
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # -----------------------------------------------------------------
        # Pair-wise relative position index: (Wh*Ww, Wh*Ww)
        # Used to index into relative_coords_table via cpb_mlp.
        # -----------------------------------------------------------------
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))     # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                      # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1           # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)              # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection, bias on Q and V only (Swin V2 convention, K has no bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)                                # output projection
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Compute windowed cosine self-attention with Log-CPB relative position bias.

        Args:
            x: (num_windows * B, N, C) flattened window tokens.
            mask: (num_windows, N, N) cyclic-shift mask for SW-MSA, or None for W-MSA.

        Returns:
            (num_windows * B, N, C)
        """
        B_, N, C = x.shape                                             # (num_windows * B, N,  C)

        # Build QKV bias (Q and V get bias, K does not — Swin V2)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),   # K bias = 0
                self.v_bias,
            ))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)   # (num_windows * B, N,  3C)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                             # each (num_windows * B, num_heads, N, C/num_heads)

        # Cosine attention: Python float keeps clamp device-agnostic (MPS safe)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)  # (num_windows * B, num_heads, N, N)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale                                      # (num_windows * B, num_heads, N, N)

        # Log-CPB relative position bias -> (num_heads, N, N) -> broadcast over batch
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)    # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)             # (num_windows * B, num_heads, N, N)

        # SW-MSA: add cyclic-shift mask to prevent cross-region attention
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)                                      # (num_windows * B, num_heads, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)              # (num_windows * B, N,  C)
        x = self.proj(x)                                               # (num_windows * B, N,  C)
        x = self.proj_drop(x)                                          # (num_windows * B, N,  C)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )
