# ===============================================================
# Swin Transformer V2: all patch operations
#
#   PatchEmbed    spectrogram  → tokens      (encoder input)
#   PatchMerging  tokens ↓2x                 (end of encoder stages 0–2)
#   PatchExpand   tokens ↑2x                 (between decoder stages)
#   PatchUnembed  tokens       → spectrogram (decoder output)
# ===============================================================

from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from ar_spectra.blocks.conv.normed import NormLinear
from ar_spectra.blocks.conv.causal import SConv2d, SConvTranspose2d
from c_vae.swin.utils import make_norm


# --------------------------------------------------------------------------- #
# PatchEmbed                                                                   #
# --------------------------------------------------------------------------- #

class PatchEmbed(nn.Module):
    """Map a 2-D spectrogram to a flat sequence of patch tokens via strided Conv2d.

    With is_complex=False uses nn.Conv2d (float32); with is_complex=True uses
    SConv2d(is_complex=True) which operates on complex64 tensors.
    kernel_size == stride == patch_size → zero padding, non-overlapping patches.

    Args:
        img_size: (H, W) of the input spectrogram (or a single int for square).
        patch_size: kernel size = stride of the Conv2d (non-overlapping patches).
        in_chans: Input channels (2 = stereo STFT, 4 = CAC format).
        embed_dim: Output token dimension.
        norm_layer: Norm constructor used when is_complex=False. Ignored when
            is_complex=True (make_norm is used instead).
        is_complex: If True, uses SConv2d + ComplexLayerNorm.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
        is_complex: bool = False,
        abs_pos_embed: bool = False,
    ) -> None:

        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size                              # (H, W) of expected input
        self.patch_size = patch_size                          # (ph, pw) kernel and stride
        self.patches_resolution = patches_resolution          # (H/ph, W/pw) token grid
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # total tokens
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.is_complex = is_complex

        if is_complex:
            self.proj = SConv2d(in_chans, embed_dim, kernel_size=patch_size,
                                stride=patch_size, is_complex=True)
            self.norm = make_norm(embed_dim, is_complex=True) if norm_layer is not None else None
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        # Inter-window absolute position embedding (off by default).
        # Shape is tied to (num_patches, embed_dim) — incompatible across patch_size/embed_dim changes.
        if abs_pos_embed:
            _real = torch.zeros(1, self.num_patches, embed_dim)
            nn.init.trunc_normal_(_real, std=0.02)
            if is_complex:
                # PE modulates real part only; imaginary starts at 0 and is learned
                self.pos_embed = nn.Parameter(torch.complex(_real, torch.zeros_like(_real)))
            else:
                self.pos_embed = nn.Parameter(_real)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_chans, H, W)

        Returns:
            (B, num_patches, embed_dim)
        """
        
        _, _, H, W = x.shape                                          # (B, C, H, W)
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input size ({H}x{W}) does not match model ({self.img_size[0]}x{self.img_size[1]})."
        )
        x = self.proj(x)                                              # (B, embed_dim, H/ph, W/pw)
        x = x.flatten(2).transpose(1, 2)                             # (B, num_patches, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        if hasattr(self, 'pos_embed'):
            x = x + self.pos_embed
        return x


# --------------------------------------------------------------------------- #
# PatchUnembed                                                                 #
# --------------------------------------------------------------------------- #

class PatchUnembed(nn.Module):
    """Reconstruct a spectrogram from a flat token sequence via ConvTranspose2d.

    Exact inverse of PatchEmbed: accepts the same token format (B, num_patches, embed_dim)
    and internally reshapes to 2-D spatial before applying ConvTranspose2d.

    With is_complex=False uses nn.ConvTranspose2d; with is_complex=True uses
    SConvTranspose2d(is_complex=True). kernel_size == stride → zero padding trim.

    Args:
        input_resolution: (H, W) token grid (= PatchEmbed.patches_resolution).
        embed_dim: Input token channel dimension (= encoder embed_dim).
        out_channels: Output spectrogram channels (2 = stereo STFT, 4 = CAC).
        patch_size: Kernel = stride of ConvTranspose2d (must match PatchEmbed).
        is_complex: If True, uses SConvTranspose2d(is_complex=True).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        embed_dim: int,
        out_channels: int,
        patch_size: int,
        is_complex: bool = False,
    ) -> None:

        super().__init__()
        self.input_resolution = input_resolution   # (H, W) token grid
        self.embed_dim = embed_dim
        self.is_complex = is_complex

        if is_complex:
            self.conv = SConvTranspose2d(
                embed_dim, out_channels, kernel_size=patch_size, stride=patch_size,
                is_complex=True,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                embed_dim, out_channels, kernel_size=patch_size, stride=patch_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, embed_dim)

        Returns:
            (B, out_channels, H * patch_size, W * patch_size)
        """
        H, W = self.input_resolution
        B, L, C = x.shape                                             # (B, H*W, embed_dim)
        assert L == H * W, f"token count {L} != H*W = {H*W}"
        x = x.transpose(1, 2).view(B, C, H, W)                       # (B, embed_dim, H, W)
        return self.conv(x)                                           # (B, out_channels, H*ps, W*ps)


# --------------------------------------------------------------------------- #
# PatchMerging                                                                 #
# --------------------------------------------------------------------------- #

class PatchMerging(nn.Module):
    """Downsample spatial resolution by 2x via 2x2 patch concat + linear reduction.

    Concatenates 4 neighbours of every 2x2 patch along the channel dim -> 4C,
    then projects 4C -> 2C via a bias-free linear layer.
    Applied at the end of encoder stages 0-2.

    The geometric concat/scatter operations are dtype-agnostic (work on both
    float32 and complex64); only the learnable layers switch via is_complex.

    Args:
        input_resolution: (H, W) grid before downsampling.
        dim: Input channel dimension C; output will be 2C.
        norm_layer: Fallback norm constructor used when is_complex=False.
            Ignored when is_complex=True (make_norm is used instead).
        is_complex: If True, uses NormLinear(is_complex=True) and
            ComplexLayerNorm; if False, uses plain nn.Linear + norm_layer.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer=nn.LayerNorm,
        is_complex: bool = False,
    ) -> None:

        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.is_complex = is_complex

        if is_complex:
            self.reduction = NormLinear(4 * dim, 2 * dim, bias=False, is_complex=True)  # 4C -> 2C
            self.norm = make_norm(2 * dim, is_complex=True)
        else:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)                   # 4C -> 2C
            self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)

        Returns:
            (B, H/2 * W/2, 2C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape                                             # (B, H*W, C)
        assert L == H * W, f"token count {L} != H*W = {H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"grid ({H}x{W}) must be even for PatchMerging"

        x = x.view(B, H, W, C)                                        # (B, H, W, C)
        x = torch.cat([
            x[:, 0::2, 0::2, :],                                      # top-left
            x[:, 1::2, 0::2, :],                                      # bottom-left
            x[:, 0::2, 1::2, :],                                      # top-right
            x[:, 1::2, 1::2, :],                                      # bottom-right
        ], dim=-1)                                                    # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                                      # (B, H/2*W/2, 4C)
        x = self.reduction(x)                                         # (B, H/2*W/2, 2C)
        x = self.norm(x)                                              # (B, H/2*W/2, 2C)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# --------------------------------------------------------------------------- #
# PatchExpand                                                                  #
# --------------------------------------------------------------------------- #

class PatchExpand(nn.Module):
    """Upsample spatial resolution by 2x via linear expansion + 2x2 pixel shuffle.

    Exact arithmetic inverse of PatchMerging:
      - PatchMerging: (B, H*W, C) -> concat 2x2 -> Linear(4C->2C) -> (B, H/2*W/2, 2C)
      - PatchExpand : (B, H*W, 2C) -> Linear(2C->4C) -> 2x2 scatter -> (B, 4*H*W, C)

    Applied between decoder stages 0-2.

    Args:
        input_resolution: (H, W) grid before upsampling.
        dim: Input channel dimension 2C; output will be C = dim // 2.
        norm_layer: Fallback norm constructor used when is_complex=False.
            Ignored when is_complex=True (make_norm is used instead).
        is_complex: If True, uses NormLinear(is_complex=True) and
            ComplexLayerNorm; if False, uses plain nn.Linear + norm_layer.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer=nn.LayerNorm,
        is_complex: bool = False,
    ) -> None:

        super().__init__()
        self.input_resolution = input_resolution   # (H, W) before expansion
        self.dim = dim                             # input channels (2C)
        self.out_dim = dim // 2                   # output channels (C)
        self.is_complex = is_complex

        if is_complex:
            self.expand = NormLinear(dim, 2 * dim, bias=False, is_complex=True)  # 2C -> 4C
            self.norm = make_norm(self.out_dim, is_complex=True)
        else:
            self.expand = nn.Linear(dim, 2 * dim, bias=False)                   # 2C -> 4C
            self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, 2C)

        Returns:
            (B, 4*H*W, C) = (B, 2H * 2W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape                                             # (B, H*W, 2C)
        assert L == H * W,    f"token count {L} != H*W = {H*W}"
        assert C == self.dim, f"channel dim {C} != expected {self.dim}"

        x = self.expand(x)                                            # (B, H*W, 4C)
        x = x.view(B, H, W, 2 * C)                                    # (B, H, W, 4C)
        x = x.view(B, H, W, 2, 2, self.out_dim)                       # (B, H, W, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()                  # (B, H, 2, W, 2, C)
        x = x.view(B, 2 * H, 2 * W, self.out_dim)                     # (B, 2H, 2W, C)
        x = x.reshape(B, 4 * H * W, self.out_dim)                     # (B, 4H*W, C)
        x = self.norm(x)                                              # (B, 4H*W, C)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}->{self.out_dim}"


# --------------------------------------------------------------------------- #
# SpatialConvSmooth                                                            #
# --------------------------------------------------------------------------- #

class SpatialConvSmooth(nn.Module):
    """Token-space 2-D spatial smoother for Swin encoder/decoder.

    Reshapes the flat token sequence (B, H*W, C) into a 2-D spatial map,
    applies a stride-1 SConv2d with reflect padding, and reshapes back.
    Used after PatchExpand (decoder) and PatchMerging (encoder) to smooth
    the discontinuities introduced by pixel-shuffle operations at patch
    boundaries before the next Swin stage processes the tokens.

    Args:
        input_resolution: (H, W) spatial grid of the incoming token sequence.
        dim:              Token channel dimension C.
        kernel_size:      Conv kernel (int or 2-tuple). Default 3.
        is_complex:       If True, uses a complex64 Conv2d.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        is_complex: bool = False,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.conv = SConv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            is_complex=is_complex,
            causal=False,
            pad_mode='reflect',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape                                         # (B, H*W, C)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)              # (B, C, H, W)
        x = self.conv(x)                                          # (B, C, H, W)
        return x.permute(0, 2, 3, 1).contiguous().view(B, L, C)   # (B, H*W, C)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}"
