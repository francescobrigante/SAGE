# =============================================================================
# Numerical precision of inference (sage.inference.SAGE, `precision=`). The settings are the ones of the
# paper's timing protocol (Appendix C): "tf32" = TF32 matmuls and convolutions (Table 1), "fp32" = no TF32
# anywhere, "bf16" = encoder and decoder under torch.autocast(bfloat16) with FP32 weights, TF32 elsewhere,
# STFTs and the bottleneck in FP32. None leaves PyTorch's flags as the caller set them (the paper's
# reconstruction and probing tables ran with PyTorch's defaults: FP32 matmuls, TF32 convolutions).
# =============================================================================

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn

from sage.constants import DEFAULT_PRECISION  # noqa: F401  (re-exported: the default of SAGE and the CLI)

PRECISIONS = ("tf32", "fp32", "bf16")


def check_precision(precision: Optional[str]) -> Optional[str]:
    """Return ``precision`` if it is one of :data:`PRECISIONS` or None; raise ``ValueError`` otherwise."""
    if precision is not None and precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS} or None, got {precision!r}")
    return precision


@contextmanager
def tf32_flags(precision: Optional[str]) -> Iterator[None]:
    """Set the TF32 switches of ``precision`` for the duration of the block, then restore them.

    The switches are process-wide: another thread running CUDA work at the same time sees them too.
    """
    if precision is None:
        yield
        return
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    on = precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _to_fp32(out):
    """Low-precision float tensors of a block's output (tensor, tuple, list or dict) back to FP32."""
    if isinstance(out, torch.Tensor):
        return out.float() if out.dtype in (torch.bfloat16, torch.float16) else out
    if isinstance(out, (tuple, list)):
        return type(out)(_to_fp32(o) for o in out)
    if isinstance(out, dict):
        return {k: _to_fp32(v) for k, v in out.items()}
    return out


def set_autocast(block: nn.Module, dtype: Optional[torch.dtype]) -> None:
    """Run ``block.forward`` under ``torch.autocast(dtype)`` and hand FP32 tensors back to the caller.

    The weights and the state dict are untouched; ``dtype=None`` restores the plain forward. The device
    type of the autocast is read from the block's parameters at every call, so ``.to()`` keeps working.
    """
    block.__dict__.pop("forward", None)                          # back to the class forward
    if dtype is None:
        return
    plain = block.forward

    def forward(*args, **kwargs):
        device_type = next(block.parameters()).device.type
        with torch.autocast(device_type, dtype=dtype):
            out = plain(*args, **kwargs)
        return _to_fp32(out)

    block.forward = forward
