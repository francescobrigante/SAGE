import torch
import torch.nn as nn
from typing import Tuple
from ar_spectra.blocks.normalization import ComplexLayerNorm
from ar_spectra.blocks.conv import SConv2d

class PatchMergingLinearComplex(nn.Module):
    """
    Swin-style patch merging:

    - Input:  x ∈ C^{B × L × D}, with L = H * W.
    - Rebuild the grid: (B, H, W, D).
    - Group 2×2 blocks: concatenate 4 tokens → (B, H/2, W/2, 4D).
    - Apply ComplexLayerNorm + Linear(4D → out_dim).
    - Return the sequence: (B, L', out_dim) with L' = H/2 * W/2.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        use_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else 2 * dim  # default: raddoppia D

        self.norm = ComplexLayerNorm(4 * dim) if use_norm else None
        self.reduction = nn.Linear(4 * dim, self.out_dim, dtype=torch.complex64)

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x:       (B, L, D) complex
        grid_hw: (H, W) with H * W = L

        Returns:
            x_merged: (B, L', out_dim) with L' = (H'/W') = (H//2 * W//2)
            new_grid: (H', W') = (H//2, W//2) (including any padding)
        """
        B, L, D = x.shape
        H, W = grid_hw

        if D != self.dim:
            raise ValueError(f"PatchMergingLinearComplex dim mismatch: D={D} vs self.dim={self.dim}")
        if L != H * W:
            raise ValueError(f"L={L} != H*W={H*W} (inconsistent grid_hw)")

        # (B, L, D) → (B, H, W, D)
        x = x.view(B, H, W, D)

        # Pad if H or W is odd (as in Swin)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            # pad format: (dim_last, dim_last, W, W, H, H)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # Extract the 4 corners of each 2×2 block
        x0 = x[:, 0::2, 0::2, :]  # top-left    (B, H/2, W/2, D)
        x1 = x[:, 1::2, 0::2, :]  # bottom-left (B, H/2, W/2, D)
        x2 = x[:, 0::2, 1::2, :]  # top-right   (B, H/2, W/2, D)
        x3 = x[:, 1::2, 1::2, :]  # bottom-right(B, H/2, W/2, D)

        # Concatenate (4 × D) for each new grid cell
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4D)

        # Norm + linear 4D → out_dim
        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, out_dim)

        new_H, new_W = x.shape[1], x.shape[2]

        # Return to sequence: (B, L', out_dim) with L' = new_H * new_W
        x = x.view(B, new_H * new_W, self.out_dim)
        return x, (new_H, new_W)


class ConvPatchDownsampleComplex(nn.Module):
    """
    Downsample tokens with a 2D convolution:

    - Input:  x ∈ C^{B × L × D}, with L = H * W.
    - Rebuild the grid: (B, H, W, D).
    - (optional) ComplexLayerNorm on D.
    - (B, H, W, D) → (B, D, H, W) → Conv2d/SConv2d → (B, out_dim, H', W').
    - Return a new sequence: (B, L', out_dim) with L' = H' * W'.

    You can set anisotropy with stride=(sH, sW), kernel_size=(kH, kW).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        kernel_size: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (0, 0),
        use_norm: bool = True,
        use_sconv: bool = True,
        conv_kwargs: dict | None = None,
    ):
        """
        dim:         input token dimension (D).
        out_dim:     output token dimension (default = dim).
        kernel_size: 2D kernel (kH, kW).
        stride:      2D stride (sH, sW) → controls downsampling (including anisotropic).
        padding:     2D padding (pH, pW).
        use_sconv:   if True, use SConv2d (complex), otherwise nn.Conv2d (real).
        conv_kwargs: passed to SConv2d/Conv2d (norm, bias, etc.).
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim

        self.norm = ComplexLayerNorm(dim) if use_norm else None

        if conv_kwargs is None:
            conv_kwargs = {}

        if use_sconv:
            # Complex convolution; adjust to the arguments of your SConv2d
            self.conv = SConv2d(
                in_channels=dim,
                out_channels=self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                is_complex=True,
                **conv_kwargs,
            )
        else:
            # Variante reale (se i token sono reali)
            self.conv = nn.Conv2d(
                in_channels=dim,
                out_channels=self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **conv_kwargs,
            )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x:       (B, L, D) complex
        grid_hw: (H, W) with H * W = L

        Returns:
            x_ds:   (B, L', out_dim) with L' = H' * W'
            new_hw: (H', W') grid dimensions after conv/stride
        """
        B, L, D = x.shape
        H, W = grid_hw

        if D != self.dim:
            raise ValueError(f"ConvPatchDownsampleComplex dim mismatch: D={D} vs self.dim={self.dim}")
        if L != H * W:
            raise ValueError(f"L={L} != H*W={H*W} (inconsistent grid_hw)")

        # (B, L, D) → (B, H, W, D)
        x = x.view(B, H, W, D)

        # Normalize on D (last axis)
        if self.norm is not None:
            x = self.norm(x)

        # (B, H, W, D) → (B, D, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 2D convolution (possibly complex via SConv2d)
        x = self.conv(x)  # (B, out_dim, H', W')

        B, C, H_new, W_new = x.shape

        # (B, C, H', W') → (B, H', W', C) → (B, L', C)
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, H', W', out_dim)
        x = x.view(B, H_new * W_new, self.out_dim)    # (B, L', out_dim)

        return x, (H_new, W_new)



class PatchUnmergingLinearComplex(nn.Module):
    """
    Swin-style patch unmerging (upsampling — exact inverse of PatchMergingLinearComplex):

    - Input:  x ∈ C^{B × L × D}, with L = H * W.
    - Rebuild the grid: (B, H, W, D).
    - Apply ComplexLayerNorm + Linear(D → 4 * out_dim).
    - Split into 4 sub-tensors of shape (B, H, W, out_dim).
    - Interleave via 2×2 scatter → (B, 2H, 2W, out_dim).
    - Return the sequence: (B, L', out_dim) with L' = 2H * 2W.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        use_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim // 2  # inverse of merging: halve D

        self.norm = ComplexLayerNorm(dim) if use_norm else None
        self.expansion = nn.Linear(dim, 4 * self.out_dim, dtype=torch.complex64)

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x:       (B, L, D) complex
        grid_hw: (H, W) with H * W = L

        Returns:
            x_unmerged: (B, L', out_dim) with L' = 2H * 2W
            new_grid:   (2H, 2W)
        """
        B, L, D = x.shape
        H, W = grid_hw

        if D != self.dim:
            raise ValueError(f"PatchUnmergingLinearComplex dim mismatch: D={D} vs self.dim={self.dim}")
        if L != H * W:
            raise ValueError(f"L={L} != H*W={H*W} (inconsistent grid_hw)")

        # (B, L, D) → (B, H, W, D)
        x = x.view(B, H, W, D)                                           # (B, H,   W,   D)

        # Norm + linear D → 4 * out_dim  (expand channels before scattering)
        if self.norm is not None:
            x = self.norm(x)                                              # (B, H,   W,   D)
        x = self.expansion(x)                                             # (B, H,   W,   4*out_dim)

        # Split into 4 sub-tensors, one per 2×2 position
        x0, x1, x2, x3 = x.chunk(4, dim=-1)                             # 4 × (B, H, W, out_dim)

        # Scatter into doubled grid via 2×2 interleaving (inverse of merging extraction)
        new_H, new_W = H * 2, W * 2
        out = torch.zeros(B, new_H, new_W, self.out_dim, dtype=x0.dtype, device=x0.device)
        out[:, 0::2, 0::2, :] = x0  # top-left
        out[:, 1::2, 0::2, :] = x1  # bottom-left
        out[:, 0::2, 1::2, :] = x2  # top-right
        out[:, 1::2, 1::2, :] = x3  # bottom-right                      # (B, 2H,  2W,  out_dim)

        # (B, 2H, 2W, out_dim) → (B, L', out_dim) with L' = 2H * 2W
        x = out.view(B, new_H * new_W, self.out_dim)                     # (B, L',  out_dim)
        return x, (new_H, new_W)


class ConvPatchUpsampleComplex(nn.Module):
    """
    Upsample tokens with a 2D transposed convolution:

    - Input:  x ∈ C^{B × L × D}, with L = H * W.
    - Rebuild the grid: (B, H, W, D).
    - (optional) ComplexLayerNorm on D.
    - (B, H, W, D) → (B, D, H, W) → ConvTranspose2d/SConvTranspose2d → (B, out_dim, H', W').
    - Return a new sequence: (B, L', out_dim) with L' = H' * W' (upsampled).

    You can set anisotropy with stride=(sH, sW), kernel_size=(kH, kW).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        kernel_size: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (0, 0),
        use_norm: bool = True,
        use_sconv: bool = True,
        conv_kwargs: dict | None = None,
    ):
        """
        dim:         input token dimension (D).
        out_dim:     output token dimension (default = dim).
        kernel_size: 2D kernel (kH, kW).
        stride:      2D stride (sH, sW) → controls upsampling (including anisotropic).
        padding:     2D padding (pH, pW).
        use_sconv:   if True, use SConvTranspose2d (complex), otherwise nn.ConvTranspose2d (real).
        conv_kwargs: passed to SConvTranspose2d/ConvTranspose2d (norm, bias, etc.).
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim

        self.norm = ComplexLayerNorm(dim) if use_norm else None

        if conv_kwargs is None:
            conv_kwargs = {}

        if use_sconv:
            # Complex transposed convolution; adjust to the arguments of your SConvTranspose2d
            self.conv = SConvTranspose2d(
                in_channels=dim,
                out_channels=self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                is_complex=True,
                **conv_kwargs,
            )
        else:
            # Variante reale (se i token sono reali)
            self.conv = nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=self.out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **conv_kwargs,
            )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x:       (B, L, D) complex
        grid_hw: (H, W) with H * W = L

        Returns:
            x_us:   (B, L', out_dim) with L' = H' * W' (upsampled)
            new_hw: (H', W') grid dimensions after transposed conv/stride
        """
        B, L, D = x.shape
        H, W = grid_hw

        if D != self.dim:
            raise ValueError(f"ConvPatchUpsampleComplex dim mismatch: D={D} vs self.dim={self.dim}")
        if L != H * W:
            raise ValueError(f"L={L} != H*W={H*W} (inconsistent grid_hw)")

        # (B, L, D) → (B, H, W, D)
        x = x.view(B, H, W, D)

        # Normalize on D (last axis)
        if self.norm is not None:
            x = self.norm(x)

        # (B, H, W, D) → (B, D, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 2D transposed convolution (possibly complex via SConvTranspose2d)
        x = self.conv(x)  # (B, out_dim, H', W')

        B, C, H_new, W_new = x.shape

        # (B, C, H', W') → (B, H', W', C) → (B, L', C)
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, H', W', out_dim)
        x = x.view(B, H_new * W_new, self.out_dim)    # (B, L', out_dim)

        return x, (H_new, W_new)
