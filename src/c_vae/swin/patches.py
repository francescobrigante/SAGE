# ===============================================================
# Swin Transformer V2: all patch operations
#
#   PatchEmbed    spectrogram  → tokens      (encoder input)
#   PatchMerging  tokens ↓2x                 (end of encoder stages 0–2)
#   PatchExpand   tokens ↑2x                 (between decoder stages)
#   PatchUnembed  tokens       → spectrogram (decoder output)
# ===============================================================

from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import to_2tuple


# --------------------------------------------------------------------------- #
# PatchEmbed                                                                   #
# --------------------------------------------------------------------------- #

class PatchEmbed(nn.Module):
    """Map a 2-D spectrogram to a flat sequence of patch tokens via strided Conv2d.

    Args:
        img_size: (H, W) of the input spectrogram (or a single int for square).
        patch_size: kernel size = stride of the Conv2d (non-overlapping patches).
        in_chans: Input channels (2 = stereo STFT, 4 = CAC format).
        embed_dim: Output token dimension.
        norm_layer: Optional normalisation applied after projection.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
    ) -> None:
        
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size                             # (H, W) of expected input
        self.patch_size = patch_size                         # (ph, pw) kernel and stride
        self.patches_resolution = patches_resolution        # (H/ph, W/pw) token grid
        self.num_patches = patches_resolution[0] * patches_resolution[1] # total tokens per sample = (H/ph) * (W/pw)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

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
        return x


# --------------------------------------------------------------------------- #
# PatchUnembed                                                                 #
# --------------------------------------------------------------------------- #

class PatchUnembed(nn.Module):
    """Reconstruct a spectrogram from a flat token sequence via ConvTranspose2d.

    Exact inverse of PatchEmbed: accepts the same token format (B, num_patches, embed_dim)
    and internally reshapes to 2-D spatial before applying ConvTranspose2d.

    Args:
        input_resolution: (H, W) token grid (= PatchEmbed.patches_resolution).
        embed_dim: Input token channel dimension (= encoder embed_dim).
        out_channels: Output spectrogram channels (2 = stereo STFT, 4 = CAC).
        patch_size: Kernel = stride of ConvTranspose2d (must match PatchEmbed).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        embed_dim: int,
        out_channels: int,
        patch_size: int,
    ) -> None:
        
        super().__init__()
        self.input_resolution = input_resolution   # (H, W) token grid
        self.embed_dim = embed_dim
        self.conv = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
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

    Args:
        input_resolution: (H, W) grid before downsampling.
        dim: Input channel dimension C; output will be 2C.
        norm_layer: Normalisation applied after the linear reduction.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer=nn.LayerNorm,
    ) -> None:
        
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)     # 4C -> 2C
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
        norm_layer: Normalisation applied to the output tokens.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer=nn.LayerNorm,
    ) -> None:
        
        super().__init__()
        self.input_resolution = input_resolution   # (H, W) before expansion
        self.dim = dim                             # input channels (2C)
        self.out_dim = dim // 2                   # output channels (C)

        self.expand = nn.Linear(dim, 2 * dim, bias=False)   # 2C -> 4C
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
