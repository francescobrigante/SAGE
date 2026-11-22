# =============================================================================
# torch.compile of inference (sage.inference.SAGE, `compile=`). Encoder and decoder are compiled in place
# (nn.Module.compile): the module tree, the state dict and the bf16 autocast wrapper of sage.precision stay as
# they are. Shapes are static (dynamic=False: dynamic shapes gave a wrong output at 30 s with torch 2.7.1), so
# the first call at each new input length compiles (~25-85 s on an A100) and later calls of that length reuse
# the code. "reduce-overhead" adds CUDA graphs: every call opens a new graph step and hands back copies of its
# outputs, so no tensor a caller keeps lives in the graph's memory pool, which the next replay overwrites.
# Inside the compiled graphs the window attention runs through FlexAttention on CUDA (sage/model/swin/sdpa.py),
# which takes FP32 or TF32 when it is compiled: the TF32 switches are among dynamo's guards, so a call under other
# precision flags (SAGE.set_precision) compiles again rather than reusing the code of the old ones.
# =============================================================================

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from sage.constants import DEFAULT_COMPILE  # noqa: F401  (re-exported: the default of SAGE and the CLI)

COMPILE_MODES = ("default", "reduce-overhead")
# Compiled graphs kept per function before dynamo falls back to eager, with a warning only (torch's default: 8).
# Encoder and decoder share one function (nn.Module._call_impl): each input length takes two entries.
RECOMPILE_LIMIT = 1024
ACCUMULATED_RECOMPILE_LIMIT = 8192


def check_compile(compile: Union[None, bool, str]) -> Optional[str]:
    """The mode of a ``compile=`` value: None for None / False (eager), ``"default"`` for True / ``"default"``,
    ``"reduce-overhead"`` for itself (CUDA graphs); ``ValueError`` for anything else."""
    if compile is None or compile is False:
        return None
    if compile is True:
        return "default"
    if isinstance(compile, str) and compile in COMPILE_MODES:
        return compile
    raise ValueError(f"compile must be None, False, True or one of {COMPILE_MODES}, got {compile!r}")


def uses_cudagraphs(mode: Optional[str]) -> bool:
    """Whether the compile mode records CUDA graphs (on CUDA tensors)."""
    return mode == "reduce-overhead"


def raise_recompile_limit() -> None:
    """Raise dynamo's recompile limits to at least :data:`RECOMPILE_LIMIT` / :data:`ACCUMULATED_RECOMPILE_LIMIT`
    (process-wide settings, never lowered), so that inputs of many lengths keep compiled code."""
    import torch._dynamo                                       # heavy: only when something is compiled

    config = torch._dynamo.config
    config.recompile_limit = max(config.recompile_limit, RECOMPILE_LIMIT)
    config.accumulated_recompile_limit = max(config.accumulated_recompile_limit, ACCUMULATED_RECOMPILE_LIMIT)


def _clone(out: Any) -> Any:
    """Copies of the tensors in a block's output (tensor, tuple, list or dict). ``clone`` keeps the strides
    (preserve_format): the encoder output is a transposed view, and sampling draws its noise in that layout."""
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, (tuple, list)):
        return type(out)(_clone(o) for o in out)
    if isinstance(out, dict):
        return {k: _clone(v) for k, v in out.items()}
    return out


def compile_module(module: nn.Module, mode: Optional[str]) -> None:
    """Compile ``module`` in place (``module.compile(mode=mode, dynamic=False)``), or back to eager with None.

    In place, as :meth:`nn.Module.compile`: the compiled function replaces the module's call, so its parameters,
    state-dict keys and attributes do not change, and a ``forward`` set on the instance (sage.precision's autocast
    wrapper) is traced too. With CUDA graphs each call first opens a new graph step
    (``torch.compiler.cudagraph_mark_step_begin``) and its output tensors are cloned out of the graph's memory pool.

    Args:
        module: The block to compile (SAGE: the encoder or the decoder).
        mode: One of :data:`COMPILE_MODES`, or None for the eager call.
    """
    mode = check_compile(mode)
    if mode is None:
        module._compiled_call_impl = None
        return
    raise_recompile_limit()
    module.compile(mode=mode, dynamic=False)
    if uses_cudagraphs(mode):
        compiled = module._compiled_call_impl

        def call(*args, **kwargs):
            torch.compiler.cudagraph_mark_step_begin()
            return _clone(compiled(*args, **kwargs))

        module._compiled_call_impl = call


def is_compiled(module: nn.Module) -> bool:
    """Whether ``module`` runs a compiled call (set by :func:`compile_module` or :meth:`nn.Module.compile`)."""
    return getattr(module, "_compiled_call_impl", None) is not None
