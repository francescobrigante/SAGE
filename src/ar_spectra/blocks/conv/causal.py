import math
import typing as tp
import warnings
from torch import nn

from .normed import NormConv1d, NormConv2d, NormConvTranspose1d, NormConvTranspose2d, get_extra_padding_for_conv1d, pad1d, pad2d, unpad1d, unpad2d

class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', is_complex: bool = False, norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn('SConv1d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, is_complex=is_complex, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        x = self.conv(x)
        return x

class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 is_complex: bool = False, norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, is_complex=is_complex, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y

def tuple_it(x, num=2):
    if isinstance(x, list):
        return tuple(x[:2])
    elif isinstance(x, int):
        return tuple([x for _ in range(num)])
    else:
        return x

class SConv2d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tp.Union[int, tp.Tuple[int, int]],
                 stride: tp.Union[int, tp.Tuple[int, int]] = 1,
                 dilation: tp.Union[int, tp.Tuple[int, int]] = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', is_complex: bool = False, norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        kernel_size, stride, dilation = tuple_it(kernel_size, 2), tuple_it(stride, 2), tuple_it(dilation, 2)

        if max(stride) > 1 and max(dilation) > 1:
            warnings.warn('SConv2d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv2d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, is_complex=is_complex, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        B, C, F, T = x.shape
        padding_total_list: tp.List[int] = []
        extra_padding_list: tp.List[int] = []
        for i, (kernel_size, stride, dilation) in enumerate(zip(
                self.conv.conv.kernel_size,
                self.conv.conv.stride,
                self.conv.conv.dilation
        )):
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            if i == 0:
                # no extra padding for frequency dim
                extra_padding = 0
            else:
                extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
            padding_total_list.append(padding_total)
            extra_padding_list.append(extra_padding)

        if self.causal:
            # always non-causal padding for frequency axis
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            # causal padding for time axis
            time_after = extra_padding_list[1]
            time_before = padding_total_list[1]
            x = pad2d(x, ((time_before, time_after), (freq_before, freq_after)), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            time_after = padding_total_list[1] // 2
            time_before = padding_total_list[1] - time_after + extra_padding_list[1]
            x = pad2d(x, ((time_before, time_after), (freq_before, freq_after)), mode=self.pad_mode)
        x = self.conv(x)
        return x

class SConvTranspose2d(nn.Module):
    """ConvTranspose2d with some builtin handling of asymmetric or causal padding
    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tp.Union[int, tp.Tuple[int, int]],
                 stride: tp.Union[int, tp.Tuple[int, int]] = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1., is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 out_padding: tp.Union[int, tp.List[tp.Tuple[int, int]]] = 0,
                 groups: int = 1):
        super().__init__()
        self.convtr = NormConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, is_complex=is_complex, norm_kwargs=norm_kwargs,
                                          groups=groups)
        if isinstance(out_padding, int):
            self.out_padding = [(out_padding, out_padding), (out_padding, out_padding)]
        else:
            self.out_padding = out_padding
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_freq_total = kernel_size - stride
        kernel_size = self.convtr.convtr.kernel_size[1]
        stride = self.convtr.convtr.stride[1]
        padding_time_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        (freq_out_pad_left, freq_out_pad_right) = self.out_padding[0]
        (time_out_pad_left, time_out_pad_right) = self.out_padding[1]
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = math.ceil(padding_time_total * self.trim_right_ratio)
            padding_time_left = padding_time_total - padding_time_right
            y = unpad2d(y, (
                (max(padding_time_left - time_out_pad_left, 0), max(padding_time_right - time_out_pad_right, 0)),
                (max(padding_freq_left - freq_out_pad_left, 0), max(padding_freq_right - freq_out_pad_right, 0))
            ))
        else:
            # Asymmetric padding required for odd strides
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = padding_time_total // 2
            padding_time_left = padding_time_total - padding_time_right
            # y = unpad2d(y, ((padding_time_left, padding_time_right), (padding_freq_left, padding_freq_right)))
            y = unpad2d(y, (
                (max(padding_time_left - time_out_pad_left, 0), max(padding_time_right - time_out_pad_right, 0)),
                (max(padding_freq_left - freq_out_pad_left, 0), max(padding_freq_right - freq_out_pad_right, 0))
            ))
        return y