# ===============================================================
# Swin Transformer V2  window partition / reverse utilities.
#
# Used by SwinTransformerBlock to split feature maps
# into non-overlapping local windows.
# ===============================================================

import torch


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition a feature map into non-overlapping local windows.

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        window_size: Side length of each square window.

    Returns:
        windows: ``(num_windows * B, window_size, window_size, C)``
    """
    B, H, W, C = x.shape                                                                # (B, H, W, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)      # (B, H/ws, ws, W/ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()                                  # (B, H/ws, W/ws, ws, ws, C)
    # num_windows = (H // window_size) * (W // window_size)
    windows = windows.view(-1, window_size, window_size, C)                             # (num_windows * B, ws, ws, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct a feature map from local windows (inverse of ``window_partition``).

    Args:
        windows: ``(num_windows * B, window_size, window_size, C)``
        window_size: Side length of each square window.
        H: Height of the original feature map.
        W: Width of the original feature map.

    Returns:
        x: ``(B, H, W, C)``
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)                    # (B, H/ws, W/ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()                      # (B, H/ws, ws, W/ws, ws, C)
    x = x.view(B, H, W, -1)                                           # (B, H, W, C)
    return x
