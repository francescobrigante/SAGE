from typing import Tuple, Union

import torch
from timm.models.layers import to_2tuple


def window_partition(x: torch.Tensor, window_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Partition a feature map into non-overlapping local windows.

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        window_size: Side length(s) of each window — int for square, (wh, ww) for rect.

    Returns:
        windows: ``(num_windows * B, wh, ww, C)``
    """
    wh, ww = to_2tuple(window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, wh, ww, C)


def window_reverse(windows: torch.Tensor, window_size: Union[int, Tuple[int, int]], H: int, W: int) -> torch.Tensor:
    """Reconstruct a feature map from local windows (inverse of ``window_partition``).

    Args:
        windows: ``(num_windows * B, wh, ww, C)``
        window_size: int or (wh, ww).
        H: Height of the original feature map.
        W: Width of the original feature map.

    Returns:
        x: ``(B, H, W, C)``
    """
    wh, ww = to_2tuple(window_size)
    B = int(windows.shape[0] / (H // wh * W // ww))
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)
