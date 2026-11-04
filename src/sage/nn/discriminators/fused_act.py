# =============================================================================
# Conv2d bias + LeakyReLU of the WavTokenizer critic in one Triton kernel (fused_bias_act). cuDNN's
# convolution adds the bias in a separate, non-vectorised kernel, then the LeakyReLU reads the result
# again; the fused kernel does both in one pass with the same float32 arithmetic (z + b, then the
# LeakyReLU), so the outputs are bit-identical. Its backward is the one of autograd (LeakyReLU backward
# on the output, bias gradient = sum over N, H, W), or, with fused_bias_grad, the same LeakyReLU
# backward with the bias-gradient sum fused in (NHWC only; another summation order for the bias).
# Anything that is not a float32 CUDA tensor outside autocast runs the plain conv + LeakyReLU.
# Kernels are launched through their compiled launcher once Triton has compiled them for a given
# specialisation: Triton's JIT dispatch costs tens of microseconds of CPU per call, which shows on the
# smaller (joint-layout) critics, whose forward is partly bound by the CPU. That launcher cache mirrors
# Triton internals (specialisation keys, CompiledKernel launch) of TRITON_VERSION only: with another
# Triton, or none, the plain path runs everywhere (one warning).
# =============================================================================

import typing as tp
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from sage.utils.console import warn

try:
    import triton
    import triton.language as tl
except ImportError:                                          # CPU-only installs: the plain path everywhere
    triton = None

FWD_BLOCK = 1024              # elements per program of the forward kernel
BWD_TILE = 4096               # elements per (rows x channels) tile of the backward kernel
MAX_NUMEL = 2**31 - BWD_TILE  # int32 offsets inside the kernels
TRITON_VERSION = "3.3.1"      # the Triton whose internals _spec and _launch mirror (tested on it)
_COMPILED: dict = {}          # (kernel, constexprs, specialisation of the args) -> triton CompiledKernel
_triton_ok: tp.Optional[bool] = None   # triton_supported(), resolved on first use


if triton is not None:
    @triton.jit
    def _bias_leaky_relu_fwd(z_ptr, b_ptr, y_ptr, numel, C, S, slope,
                             CHANNELS_LAST: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        if CHANNELS_LAST:
            c = offs % C                                     # NHWC: channel is the fastest index
        else:
            c = (offs // S) % C                              # NCHW: one plane of S = H*W per channel
        v = tl.load(z_ptr + offs, mask=mask) + tl.load(b_ptr + c, mask=mask)
        tl.store(y_ptr + offs, tl.where(v > 0, v, v * slope), mask=mask)

    @triton.jit
    def _leaky_relu_bwd_bias_sum(g_ptr, y_ptr, gz_ptr, part_ptr, M, C, slope,
                                 BM: tl.constexpr, BC: tl.constexpr):
        # NHWC viewed as (M = N*H*W rows, C): gz = LeakyReLU'(y) * g, and the sum of the BM rows of the
        # tile per channel into part[pid_m, c] (the bias gradient is part.sum(0)).
        pid_m, pid_c = tl.program_id(0), tl.program_id(1)
        m = pid_m * BM + tl.arange(0, BM)
        c = pid_c * BC + tl.arange(0, BC)
        mask = (m < M)[:, None] & (c < C)[None, :]
        offs = m[:, None] * C + c[None, :]
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        gz = tl.where(y > 0, g, g * slope)
        tl.store(gz_ptr + offs, gz, mask=mask)
        tl.store(part_ptr + pid_m * C + c, tl.sum(gz, axis=0), mask=c < C)


def triton_supported() -> bool:
    """True when Triton imports at TRITON_VERSION; otherwise warns once and the plain path runs."""
    global _triton_ok
    if _triton_ok is None:
        found = None if triton is None else triton.__version__
        _triton_ok = found == TRITON_VERSION
        if not _triton_ok:
            warn(f"Triton {found or 'not importable'}: the fused kernels' launcher cache is written against "
                 f"Triton {TRITON_VERSION}; the critic runs the plain conv + LeakyReLU", prefix="fused_bias_act")
    return _triton_ok


def _spec(a) -> tuple:
    """What Triton keys its compiled kernels on, per argument: pointer 16-byte alignment; ints equal
    to 1, divisible by 16, or 64-bit; floats are always fp32."""
    if isinstance(a, torch.Tensor):
        return (a.dtype, a.data_ptr() % 16 == 0)
    if isinstance(a, int):
        return (a == 1, a % 16 == 0, a >= 2**31)
    return (type(a),)


def _launch(kernel, grid: tuple, device: torch.device, *args) -> None:
    """kernel[grid](*args); args end with the constexprs. The first call of each specialisation goes
    through Triton's JIT (compiles, launches, returns the compiled kernel), later ones through the
    compiled launcher directly, on the current stream."""
    n_const = sum(p.is_constexpr for p in kernel.params)
    runtime, consts = args[:-n_const], args[-n_const:]
    key = (kernel, consts, *(_spec(a) for a in runtime))
    compiled = _COMPILED.get(key)
    with torch.cuda.device(device) if device.index != torch.cuda.current_device() else nullcontext():
        if compiled is None:
            _COMPILED[key] = kernel[grid](*runtime, **dict(zip(kernel.arg_names[-n_const:], consts)))
        else:
            compiled[grid](*args)


def _layout(t: torch.Tensor) -> tp.Optional[bool]:
    """True = NHWC-contiguous, False = NCHW-contiguous, None = neither."""
    if t.is_contiguous():
        return False
    return True if t.is_contiguous(memory_format=torch.channels_last) else None


def _kernel_ok(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype == torch.float32 and t.numel() < MAX_NUMEL and triton_supported()


def _bias_leaky_relu_forward(z: torch.Tensor, bias: torch.Tensor, slope: float) -> torch.Tensor:
    """leaky_relu(z + bias[c]) of a (N, C, H, W) conv output, in z's memory layout."""
    channels_last = _layout(z)
    if channels_last is None or not _kernel_ok(z):
        return F.leaky_relu(z + bias.view(1, -1, 1, 1), slope)       # same arithmetic, two kernels
    y = torch.empty_like(z)
    n = z.numel()
    _launch(_bias_leaky_relu_fwd, (triton.cdiv(n, FWD_BLOCK), 1, 1), z.device,
            z, bias, y, n, z.shape[1], z.shape[2] * z.shape[3], slope, channels_last, FWD_BLOCK)
    return y


def _leaky_relu_backward_bias_sum(g: torch.Tensor, y: torch.Tensor, slope: float
                                  ) -> tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]]:
    """(grad of the conv output, bias gradient) in one pass over NHWC tensors; None if not applicable."""
    if not (_layout(y) and _kernel_ok(y)):
        return None
    g = g.contiguous(memory_format=torch.channels_last)
    gz = torch.empty_like(y)
    C = y.shape[1]
    M = y.numel() // C
    bc = min(64, triton.next_power_of_2(C))
    bm = BWD_TILE // bc
    part = torch.empty(triton.cdiv(M, bm), C, device=y.device, dtype=torch.float32)
    _launch(_leaky_relu_bwd_bias_sum, (triton.cdiv(M, bm), triton.cdiv(C, bc), 1), y.device,
            g, y, gz, part, M, C, slope, bm, bc)
    return gz, part.sum(0)


class _BiasLeakyReLU(torch.autograd.Function):
    """y = leaky_relu(z + bias) with z the bias-free conv output. Backward as autograd's for the conv
    bias + LeakyReLU (aten convolution_backward sums grad_output over (0, 2, 3) for the bias), or with
    fused_grad the bias sum fused into the LeakyReLU backward."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, bias: torch.Tensor, slope: float, fused_grad: bool) -> torch.Tensor:
        y = _bias_leaky_relu_forward(z, bias, slope)
        ctx.save_for_backward(y)
        ctx.slope, ctx.fused_grad = slope, fused_grad
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (y,) = ctx.saved_tensors
        need_bias = ctx.needs_input_grad[1]
        fused = _leaky_relu_backward_bias_sum(g, y, ctx.slope) if (ctx.fused_grad and need_bias) else None
        if fused is not None:
            gz, gb = fused
        else:
            gz = torch.ops.aten.leaky_relu_backward(g, y, ctx.slope, True)   # LeakyReluBackward1 (on the result)
            gb = gz.sum((0, 2, 3)) if need_bias else None                   # ConvolutionBackward0's bias sum
        return gz, gb, None, None


class BiasLeakyReLUConv2d(nn.Conv2d):
    """Conv2d followed by its LeakyReLU. Float32 CUDA inputs outside autocast, with Triton at TRITON_VERSION:
    bias-free convolution, then bias + LeakyReLU in one kernel. Anything else: the plain conv (with bias)
    and LeakyReLU."""

    act_slope: float = 0.1            # negative slope of the LeakyReLU
    act_inplace: bool = False         # plain path: LeakyReLU in place (inplace_act)
    fused_grad: bool = False          # fused path: bias-gradient sum inside the LeakyReLU backward kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None or not self._fused(x):
            return F.leaky_relu(self._conv_forward(x, self.weight, self.bias), self.act_slope,
                                inplace=self.act_inplace)
        z = self._conv_forward(x, self.weight, None)                             # (B, C_out, H', W')
        if not (z.requires_grad or self.bias.requires_grad):                    # no graph: no Function
            return _bias_leaky_relu_forward(z, self.bias, self.act_slope)        # (B, C_out, H', W')
        return _BiasLeakyReLU.apply(z, self.bias, self.act_slope, self.fused_grad)  # (B, C_out, H', W')

    def _fused(self, x: torch.Tensor) -> bool:
        return (x.is_cuda and x.dtype == torch.float32 and self.weight.dtype == torch.float32
                and not torch.is_autocast_enabled("cuda") and triton_supported())


def fuse_conv_leaky_relu(conv: nn.Conv2d, slope: float, inplace: bool, fused_grad: bool) -> None:
    """Turn `conv` (weight_norm hooks and parameters kept) into a BiasLeakyReLUConv2d in place."""
    conv.__class__ = BiasLeakyReLUConv2d
    conv.act_slope, conv.act_inplace, conv.fused_grad = slope, inplace, fused_grad


def fuse_bias_act(module: nn.Module, fused_grad: bool = False) -> int:
    """Fuse every Conv2d + LeakyReLU of a critic; returns the number of fused pairs.

    Two patterns: the Vocos sub-discriminators (`convs`, each followed by a functional LeakyReLU of
    slope `lrelu_slope`; they skip it once `fused_act` is set) and DAC's WNConv2d blocks
    (nn.Sequential(Conv2d, LeakyReLU): the LeakyReLU becomes an Identity). conv_post is left alone.
    """
    n = 0
    for m in module.modules():
        if hasattr(m, "fused_act") and isinstance(getattr(m, "convs", None), nn.ModuleList):
            for conv in m.convs:
                fuse_conv_leaky_relu(conv, m.lrelu_slope, m.inplace_act, fused_grad)
                n += 1
            m.fused_act = True
        elif (isinstance(m, nn.Sequential) and len(m) == 2 and isinstance(m[0], nn.Conv2d)
              and isinstance(m[1], nn.LeakyReLU)):
            fuse_conv_leaky_relu(m[0], m[1].negative_slope, m[1].inplace, fused_grad)
            m[1] = nn.Identity()
            n += 1
    return n
