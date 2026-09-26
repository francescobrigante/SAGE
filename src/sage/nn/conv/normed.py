import torch
import math
import typing as tp
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm

from sage.nn.normalization import ConvLayerNorm


def _complex_norms():
    """Complex normalisations, imported only on the is_complex=True path (optional [complex] extra)."""
    from sage.nn.complex import normalization
    return normalization

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm', 
                                 "batch_norm"])

def _resolve_complex_dtype(preferred_real: tp.Optional[torch.dtype]) -> torch.dtype:
    """Map real precision to complex precision.
    - float16  -> complex32 (if available) else complex64
    - float32  -> complex64
    - float64  -> complex128
    - bfloat16 -> complex64 (complex-bf16 not supported)
    - None     -> complex64
    """
    if preferred_real in (torch.float16, torch.half):
        return getattr(torch, 'complex32', torch.complex64)
    if preferred_real in (torch.float32, torch.float):
        return torch.complex64
    if preferred_real in (torch.float64, torch.double):
        return torch.complex128
    if preferred_real == torch.bfloat16:
        return torch.complex64
    return torch.complex64

def apply_parametrization_norm(module: nn.Module, norm: str = 'none', is_complex: bool = False) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        # Use complex-aware reparam only when explicitly requested
        return _complex_norms().ComplexWeightNorm(module) if is_complex else weight_norm(module)
    elif norm == 'spectral_norm':
        if is_complex:
            raise NotImplementedError("Spectral norm for complex weights is not implemented yet.")
        else:
            return spectral_norm(module)
    else:
        # Any other choice doesn't need reparametrization
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', is_complex: bool = False, **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "batch_norm":
        if is_complex:
            if isinstance(module, nn.Conv1d):
                return _complex_norms().ComplexBatchNorm1d(module.out_channels, **norm_kwargs)
            else:
                return _complex_norms().ComplexBatchNorm2d(module.out_channels, **norm_kwargs)
        else:
            if isinstance(module, nn.Conv1d):
                return nn.BatchNorm1d(module.out_channels, **norm_kwargs)
            else:
                return nn.BatchNorm2d(module.out_channels, **norm_kwargs)
    if norm == 'layer_norm':
        if is_complex:
            if isinstance(module, nn.Conv1d):
                return _complex_norms().ComplexConvLayerNorm1d(module.out_channels, **norm_kwargs)
            else:
                return _complex_norms().ComplexConvLayerNorm2d(module.out_channels, **norm_kwargs)
        else:
            return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        num_groups = 1
        if "num_groups" in norm_kwargs:
            num_groups = norm_kwargs.pop("num_groups")
        if is_complex:
            return _complex_norms().ComplexGroupNorm(module.out_channels, num_groups, **norm_kwargs)
        else:
            return nn.GroupNorm(num_groups, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))

def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

def pad2d(x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    freq_len, time_len = x.shape[-2:]
    padding_time, padding_freq = paddings
    assert min(padding_freq) >= 0 and min(padding_time) >= 0, (padding_time, padding_freq)
    if mode == 'reflect':
        max_time_pad, max_freq_pad = max(padding_time), max(padding_freq)
        extra_time_pad = max_time_pad - time_len + 1 if time_len <= max_time_pad else 0
        extra_freq_pad = max_freq_pad - freq_len + 1 if freq_len <= max_freq_pad else 0
        extra_pad = [0, extra_time_pad, 0, extra_freq_pad]
        x = F.pad(x, extra_pad)
        padded = F.pad(x, (*padding_time, *padding_freq), mode, value)
        freq_end = padded.shape[-2]-extra_freq_pad
        time_end = padded.shape[-1]-extra_time_pad
        return padded[..., :freq_end, :time_end]
    else:
        return F.pad(x, (*paddings[0], *paddings[1]), mode, value)

def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]

def unpad2d(x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_time_left, padding_time_right = paddings[0]
    padding_freq_left, padding_freq_right = paddings[1]
    assert min(paddings[0]) >= 0 and min(paddings[1]) >= 0, paddings
    assert (padding_time_left + padding_time_right) <= x.shape[-1] and (padding_freq_left + padding_freq_right) <= x.shape[-2]

    freq_end = x.shape[-2] - padding_freq_right
    time_end = x.shape[-1] - padding_time_right
    return x[..., padding_freq_left: freq_end, padding_time_left: time_end]

class NormLinear(nn.Module):
    """Wrapper around Linear and normalization applied to this linear
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        if is_complex and ('dtype' not in kw or kw['dtype'] is None):
            real_pref = torch.get_default_dtype()
            kw['dtype'] = _resolve_complex_dtype(real_pref)
        self.linear = apply_parametrization_norm(nn.Linear(*args, **kw), norm, is_complex=is_complex)
        self.norm = get_norm_module(self.linear, False, norm, is_complex=is_complex, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none', is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        # ensure dtype kwarg for complex convs
        kw = dict(kwargs)
        if is_complex and ('dtype' not in kw or kw['dtype'] is None):
            # try to infer from bias/weight if provided in factory args; else default
            real_pref = torch.get_default_dtype()
            kw['dtype'] = _resolve_complex_dtype(real_pref)
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kw), norm, is_complex=is_complex)
        self.norm = get_norm_module(self.conv, causal, norm, is_complex=is_complex, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        # Removed `torch.isnan(torch.sum(x)).any()` NaN guard — it forced a
        # CUDA→CPU sync on every forward, breaking async kernel dispatch.
        # Use `Trainer(detect_anomaly=True)` for NaN diagnostics instead.
        x = self.conv(x)
        x = self.norm(x)
        return x

class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none', is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        if is_complex and ('dtype' not in kw or kw['dtype'] is None):
            real_pref = torch.get_default_dtype()
            kw['dtype'] = _resolve_complex_dtype(real_pref)
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kw), norm, is_complex=is_complex)
        self.norm = get_norm_module(self.conv, causal, norm, is_complex=is_complex, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none', is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        if is_complex and ('dtype' not in kw or kw['dtype'] is None):
            real_pref = torch.get_default_dtype()
            kw['dtype'] = _resolve_complex_dtype(real_pref)
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kw), norm, is_complex=is_complex)
        self.norm = get_norm_module(self.convtr, causal, norm, is_complex=is_complex, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x

class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none', is_complex: bool = False,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        if is_complex and ('dtype' not in kw or kw['dtype'] is None):
            real_pref = torch.get_default_dtype()
            kw['dtype'] = _resolve_complex_dtype(real_pref)
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kw), norm, is_complex=is_complex)
        self.norm = get_norm_module(self.convtr, causal, norm, is_complex=is_complex, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x