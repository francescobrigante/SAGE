# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import swin_window_process


class WindowProcess(torch.autograd.Function):
    """Fused roll + window_partition (K1 forward, K2 backward).

    Args (forward):
        input:    (B, H, W, C) float32 or float16 CUDA tensor (contiguous).
        B, H, W, C: tensor dimensions.
        shift_h:  cyclic shift along H (negative = shift up).
        shift_w:  cyclic shift along W (negative = shift left).
        window_h: window height.
        window_w: window width.

    Returns:
        (B*nH*nW, window_h, window_w, C) partitioned tensor.

    Complex64 usage: reinterpret as float32 with 2C channels before calling.
    """

    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_h, shift_w, window_h, window_w):
        output = swin_window_process.roll_and_window_partition_forward(
            input, B, H, W, C, shift_h, shift_w, window_h, window_w)
        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_h = shift_h
        ctx.shift_w = shift_w
        ctx.window_h = window_h
        ctx.window_w = window_w
        return output

    @staticmethod
    def backward(ctx, grad_in):
        grad_out = swin_window_process.roll_and_window_partition_backward(
            grad_in.contiguous(), ctx.B, ctx.H, ctx.W, ctx.C,
            ctx.shift_h, ctx.shift_w, ctx.window_h, ctx.window_w)
        return grad_out, None, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    """Fused window_merge + roll (K3 forward, K4 backward).

    Args (forward):
        input:    (B*nH*nW, window_h, window_w, C) float32 or float16 CUDA tensor (contiguous).
        B, H, W, C: spatial dimensions of the output.
        shift_h:  cyclic reverse-shift along H (positive = shift down).
        shift_w:  cyclic reverse-shift along W (positive = shift right).
        window_h: window height.
        window_w: window width.

    Returns:
        (B, H, W, C) merged and un-rolled tensor.

    Complex64 usage: reinterpret as float32 with 2C channels before calling.
    """

    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_h, shift_w, window_h, window_w):
        output = swin_window_process.window_merge_and_roll_forward(
            input, B, H, W, C, shift_h, shift_w, window_h, window_w)
        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_h = shift_h
        ctx.shift_w = shift_w
        ctx.window_h = window_h
        ctx.window_w = window_w
        return output

    @staticmethod
    def backward(ctx, grad_in):
        grad_out = swin_window_process.window_merge_and_roll_backward(
            grad_in.contiguous(), ctx.B, ctx.H, ctx.W, ctx.C,
            ctx.shift_h, ctx.shift_w, ctx.window_h, ctx.window_w)
        return grad_out, None, None, None, None, None, None, None, None
