# ===============================================================
# SDPA path of WindowAttention (inference only). Same formula as
# WindowAttention.forward (Swin V2 cosine attention + Log-CPB bias + cyclic-shift
# mask + XSA), computed by F.scaled_dot_product_attention: one fused kernel on
# CUDA. Equal in exact arithmetic, not bit-identical; set per module by
# enable_sdpa, which sage.inference.SAGE calls by default (sdpa=False opts out).
# Options of the same path, on with it unless turned off, exact in exact arithmetic
# too: the Q/V bias added by the QKV GEMM itself (fused_bias), one QKV / output
# projection for all phases of a variable-length multi-phase block (shared_phases,
# SwinTransformerBlock._shared_phase_attention), and, inside a torch.compile graph
# on CUDA, FlexAttention with the bias and the shift mask computed from the token
# indices (flex): no (nW·h, N, N) bias tensor is built or read.
# The Log-CPB bias depends on the weights only: it is computed once in full FP32
# and cached, and recomputed whenever the weights, their dtype or device change.
# Calls larger than the CUDA launch grid allows are split along batch / heads.
# ===============================================================

import importlib.util
import math
import threading
from contextlib import contextmanager
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRECISION_LOCK = threading.Lock()           # the matmul precision is a process-wide setting
_GRID_LIMIT = 65535                          # CUDA cap on the y / z axes of a launch grid


def _has_flex() -> bool:
    """torch.nn.attention.flex_attention exists (torch >= 2.5). Looked up, not imported: the import pulls in
    torch._dynamo (~2.5 s), so it happens inside the compiled graph that uses it."""
    try:
        return importlib.util.find_spec("torch.nn.attention.flex_attention") is not None
    except ImportError:
        return False


FLEX_AVAILABLE = _has_flex()
# Where the FlexAttention path runs (inside a compiled graph only): CUDA, in the dtypes of its fused kernels. Its
# CPU lowering needs AVX2 and a recent C++ compiler and was never measured: compiled CPU graphs keep SDPA.
_FLEX_DEVICES = ("cuda",)
_FLEX_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


class ShiftGeometry(NamedTuple):
    """The SW-MSA call that ``shift_mask(Hp, Wp, (wh, ww), (sh, sw))`` (sage/model/swin/block.py) masks: padded
    grid, window and shift. The FlexAttention path computes the mask from it and the token indices."""
    Hp: int
    Wp: int
    wh: int
    ww: int
    sh: int
    sw: int

    @property
    def windows(self) -> int:
        """Windows per image, nW (window_partition order: row-major over the (Hp/wh, Wp/ww) window grid)."""
        return _windows(self)


def _windows(g: ShiftGeometry) -> int:
    """``g.windows`` for traced code: Dynamo does not evaluate properties of a NamedTuple (it would compare a
    GetAttrVariable with an int and raise a TypeError, which splits the compiled graph at every call)."""
    return (g.Hp // g.wh) * (g.Wp // g.ww)


@contextmanager
def _full_fp32():
    """Full-FP32 matmuls (no TF32) while the cached bias is computed: the cache must not depend on the
    precision flags of whichever call happens to fill it."""
    with _PRECISION_LOCK:
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(precision)


def cpb_bias(attn: nn.Module) -> torch.Tensor:
    """16·sigmoid(Log-CPB) of WindowAttention.forward, in the dtype of the module's weights.

    Same ops as the original (Linear → ReLU → Linear → gather → permute → 16·sigmoid), never
    under autocast: the bias stays in the weights' precision whatever the activations' dtype.

    Args:
        attn: A real-valued WindowAttention.
    Returns:
        (num_heads, N, N) with N = Wh·Ww.
    """
    with torch.autocast(attn.logit_scale.device.type, enabled=False):
        table = attn.cpb_mlp(attn.relative_coords_table).view(-1, attn.num_heads)            # ((2Wh-1)(2Ww-1), h)
        n = attn.window_size[0] * attn.window_size[1]
        bias = table[attn.relative_position_index.view(-1)].view(n, n, -1)                   # (N, N, h)
        bias = bias.permute(2, 0, 1).contiguous()                                            # (h, N, N)
        return 16 * torch.sigmoid(bias)                                                      # (h, N, N)


def _stamp(sources: tuple) -> tuple:
    """Storage address and version counter of each tensor the bias is computed from: changes when the data
    is replaced or edited in place (inference tensors have no version counter)."""
    return tuple((t.data_ptr(), -1 if t.is_inference() else t._version) for t in sources)


def refresh_cpb(attn: nn.Module) -> torch.Tensor:
    """Recompute the cached CPB bias of one WindowAttention from its current weights. Returns (h, N, N)."""
    sources = (*attn.cpb_mlp.parameters(), attn.relative_coords_table, attn.relative_position_index)
    with torch.no_grad(), _full_fp32():
        cpb = cpb_bias(attn)                                                                 # (h, N, N)
    attn._sdpa_cpb, attn._sdpa_sources, attn._sdpa_stamp = cpb, sources, _stamp(sources)
    return cpb


def _cache_fits_weights(attn: nn.Module) -> bool:
    """The cached bias exists on the weights' device and in their dtype (the check a compiled graph can make)."""
    cpb = getattr(attn, "_sdpa_cpb", None)
    weight = attn.cpb_mlp[-1].weight
    return cpb is not None and cpb.device == weight.device and cpb.dtype == weight.dtype


def _cached_cpb(attn: nn.Module) -> torch.Tensor:
    """The cached CPB bias, (h, N, N). Eager: checked against the weights on every call. Inside a
    torch.compile graph the check cannot run: the cache (refreshed by enable_sdpa, load_state_dict,
    .to() and dtype casts) is read as an input; a mismatch left anyway is recomputed in the graph."""
    if torch.compiler.is_compiling():
        return attn._sdpa_cpb if _cache_fits_weights(attn) else cpb_bias(attn)
    sources = getattr(attn, "_sdpa_sources", None)
    if sources is None or attn._sdpa_stamp != _stamp(sources):
        return refresh_cpb(attn)
    return attn._sdpa_cpb


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1), split when too large to launch.

    The memory-efficient CUDA kernel runs one block per (query tile, head, batch) on the grid axes
    (x, y, z) and does not check the 65535 cap on y and z (torch 2.7): SW-MSA puts the windows on
    the heads axis (nW·h > 65535 beyond ~760 s of audio in one pass at stage 1), W-MSA on the batch
    axis (nW·B > 65535 beyond ~6100 s summed over the batch). Above _GRID_LIMIT each axis is cut
    into slices of at most _GRID_LIMIT: every (batch, head) pair is computed on its own, so the
    values do not change; at or below it this is the single call, unchanged.

    Args:
        q, k, v: (B, H, N, d)
        bias: (1 or B, 1 or H, N, N), broadcast over its size-1 axes.
    Returns:
        (B, H, N, d)
    """
    for dim in (0, 1):                                                  # z = batch, y = heads
        n = q.shape[dim]
        if n > _GRID_LIMIT:
            parts = []
            for start in range(0, n, _GRID_LIMIT):
                size = min(_GRID_LIMIT, n - start)
                qs, ks, vs = (t.narrow(dim, start, size) for t in (q, k, v))
                bs = bias if bias.shape[dim] == 1 else bias.narrow(dim, start, size)
                parts.append(_sdpa(qs, ks, vs, bs))                     # the other axis is checked in there
            return torch.cat(parts, dim=dim)                            # (B, H, N, d)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)


def _region(pos: torch.Tensor, size: int, window: int, shift: int) -> torch.Tensor:
    """Region of ``shift_mask`` along one padded axis: [0, size - window) → 0, [size - window, size - shift) → 1,
    [size - shift, size) → 2 (int32, same shape as ``pos``)."""
    return (pos >= size - window).to(torch.int32) + (pos >= size - shift).to(torch.int32)


def flex_score_mod(cpb: torch.Tensor, shift: Optional[ShiftGeometry] = None, offset: int = 0):
    """score_mod of FlexAttention for :func:`sdpa_attend`: adds B_cpb[h, q, k], and -100 (as ``shift_mask``, not
    -inf) between tokens of different regions of the shifted grid.

    The region of a token follows from its index alone: the call's batch index b holds window
    ``w = (b + offset) mod nW`` (window_partition: batch-major, windows row-major), query / key index n sits at
    row ``(w // nWx)·wh + n // ww`` and column ``(w mod nWx)·ww + n mod ww`` of the padded grid. An axis without a
    shift has one region: ``shift_mask``'s last slice ``slice(-0, None)`` then covers the whole axis.

    Args:
        cpb: (h, N, N) cached CPB bias; a graph input (a tensor computed inside the graph does not lower).
        shift: Geometry of the shift mask; None for W-MSA (bias only).
        offset: Window index of batch row 0 of this call, when a call starts inside an image (grid split).
    Returns:
        ``score_mod(score, b, h, q_idx, kv_idx)``.
    """
    if shift is None:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + cpb[h, q_idx, kv_idx]
        return score_mod
    g = shift
    nW, nWx = _windows(g), g.Wp // g.ww

    def score_mod(score, b, h, q_idx, kv_idx):
        w = (b + offset) % nW
        bias = cpb[h, q_idx, kv_idx]
        same = None
        if g.sh > 0:
            row = (w // nWx) * g.wh
            same = _region(row + q_idx // g.ww, g.Hp, g.wh, g.sh) == _region(row + kv_idx // g.ww, g.Hp, g.wh, g.sh)
        if g.sw > 0:
            col = (w % nWx) * g.ww
            cols = _region(col + q_idx % g.ww, g.Wp, g.ww, g.sw) == _region(col + kv_idx % g.ww, g.Wp, g.ww, g.sw)
            same = cols if same is None else same & cols
        if same is not None:
            bias = torch.where(same, bias, bias - 100.0)                 # = cpb + shift_mask, as the SDPA bias
        return score + bias
    return score_mod


def _flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cpb: torch.Tensor,
          shift: Optional[ShiftGeometry]) -> torch.Tensor:
    """flex_attention(q, k, v, score_mod=flex_score_mod(...), scale=1), split when too large to launch.

    The fused kernel runs one block per (query tile, batch·head) on the grid axes (x, y): batch·heads above the
    65535 cap on y is cut along the batch axis. Slices hold whole images (multiples of nW rows) when an image
    fits in one launch, so every slice starts at window 0; beyond that (one image of more than 65535 / h windows)
    each slice passes the window index of its first row to the score_mod.

    Args:
        q, k, v: (nW·B, h, N, d) window tokens, window index fastest (window_partition order).
        cpb: (h, N, N) cached CPB bias.
        shift: Geometry of the shift mask, or None (W-MSA).
    Returns:
        (nW·B, h, N, d)
    """
    from torch.nn.attention.flex_attention import flex_attention  # imported here: pulls in torch._dynamo

    rows, heads = q.shape[0], q.shape[1]
    nW = _windows(shift) if shift is not None else 1
    per_call = max(_GRID_LIMIT // heads, 1)
    if per_call >= nW:
        per_call -= per_call % nW                                       # whole images per slice
    if rows <= per_call:
        return flex_attention(q, k, v, score_mod=flex_score_mod(cpb, shift), scale=1.0)
    parts = []
    for start in range(0, rows, per_call):
        size = min(per_call, rows - start)
        qs, ks, vs = (t.narrow(0, start, size) for t in (q, k, v))
        parts.append(flex_attention(qs, ks, vs, score_mod=flex_score_mod(cpb, shift, start % nW), scale=1.0))
    return torch.cat(parts, dim=0)                                      # (nW·B, h, N, d)


def flex_supported(attn: nn.Module) -> bool:
    """The FlexAttention path can serve this module: torch has it, the module is real-valued and its head size
    suits the fused kernels (a power of two, at least 16)."""
    if attn.is_complex or attn.dim % attn.num_heads:
        return False
    d = attn.dim // attn.num_heads
    return FLEX_AVAILABLE and d >= 16 and d & (d - 1) == 0


def _flex_ready(attn: nn.Module, q: torch.Tensor, dt: torch.dtype, mask: Optional[torch.Tensor],
                shift: Optional[ShiftGeometry]) -> bool:
    """FlexAttention instead of SDPA for this call: flex on, inside a compiled graph (eager flex_attention is an
    unfused fallback), on a device and dtype of its fused kernels, a shift mask described by its geometry, and the
    cached bias usable as a graph input."""
    return (attn.flex and torch.compiler.is_compiling() and q.device.type in _FLEX_DEVICES and dt in _FLEX_DTYPES
            and (mask is None or shift is not None) and _cache_fits_weights(attn))


def fusable_qkv_bias(attn: nn.Module) -> bool:
    """True when the Q/V bias can be added by the QKV GEMM: a real module with a Q/V bias whose QKV
    projection is a plain linear (no bias of its own, no norm after it)."""
    return (not attn.is_complex and attn.q_bias is not None
            and attn.qkv.norm_type == "none" and attn.qkv.linear.bias is None)


def _qkv_bias(attn: nn.Module) -> torch.Tensor:
    """The (3C,) bias cat(q_bias, 0, v_bias) of WindowAttention.forward (K bias = 0, Swin V2)."""
    return torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))


def sdpa_qkv(attn: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
    """QKV projection plus Q/V bias of WindowAttention.forward: ``qkv(x) + cat(q_bias, 0, v_bias)``.

    With ``attn.fused_bias`` the bias is added by the GEMM (F.linear with the same weight and bias): no
    separate pass over the (…, 3C) output. Equal in exact arithmetic. Under autocast the GEMM takes the
    bias in bf16/fp16 and rounds q/k/v once, where the original rounds x·W and then adds the FP32 bias;
    the result is cast to the original's dtype (FP32), so every later step sees the same dtypes.

    Args:
        attn: A real-valued WindowAttention.
        x: (…, C) tokens.
    Returns:
        (qkv, dt): qkv (…, 3C) in the dtype of the original sum (FP32 under bf16 autocast, the bias
        promotes it); dt, the dtype of the GEMM output (bf16 under autocast), in which the attention runs.
    """
    if attn.fused_bias:
        bias = _qkv_bias(attn)                                          # (3C,)
        qkv = F.linear(x, attn.qkv.linear.weight, bias)                 # (…, 3C), bias in the GEMM
        dt = qkv.dtype
        return qkv.to(torch.promote_types(dt, bias.dtype)), dt
    qkv = attn.qkv(x)                                                   # (…, 3C)
    dt = qkv.dtype                                                      # bf16/fp16 under autocast, else the weights'
    if attn.q_bias is not None:
        qkv = qkv + _qkv_bias(attn)
    return qkv, dt


def sdpa_attend(attn: nn.Module, qkv: torch.Tensor, mask: Optional[torch.Tensor],
                dt: torch.dtype, shift: Optional[ShiftGeometry] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention of WindowAttention.forward from the projected window tokens, before XSA and projection.

    Original:  softmax(s·(q̂·k̂ᵀ) + B_cpb + M_shift)·v   (s = per-head temperature, q̂/k̂ = L2-normalised q/k)
    Here:      SDPA(Q = s·q̂, K = k̂, V = v, attn_mask = B_cpb + M_shift, scale = 1)
    Equal in exact arithmetic because s is a per-head scalar: (s·q̂)·k̂ᵀ = s·(q̂·k̂ᵀ).
    Inside a compiled graph with ``attn.flex`` (see :func:`_flex_ready`), FlexAttention computes the same
    scores with B_cpb + M_shift added by :func:`flex_score_mod` from the token indices and ``shift``.

    Args:
        attn: The real-valued WindowAttention whose weights are used.
        qkv: (nW·B, N, 3C) window tokens from :func:`sdpa_qkv`, window index fastest (window_partition order).
        mask: (nW, N, N) cyclic-shift mask (0 / -100) or None.
        dt: Dtype the attention kernel runs in (second output of :func:`sdpa_qkv`).
        shift: Geometry of ``mask`` (the FlexAttention path needs it), or None.
    Returns:
        (y, v): (nW·B, h, N, d) attention output in ``dt`` and the value vectors in qkv's dtype (for XSA).
    """
    B_, N, _ = qkv.shape                                                # (nW·B, N, 3C)
    h = attn.num_heads
    qkv = qkv.reshape(B_, N, 3, h, -1).permute(2, 0, 3, 1, 4)           # (3, nW·B, h, N, d)
    q, k, v = qkv[0], qkv[1], qkv[2]                                    # (nW·B, h, N, d)
    s = torch.clamp(attn.logit_scale, max=math.log(1.0 / 0.01)).exp()  # (h, 1, 1)
    q = F.normalize(q, dim=-1) * s                                      # (nW·B, h, N, d)  s·q̂
    k = F.normalize(k, dim=-1)                                          # (nW·B, h, N, d)  k̂
    cpb = _cached_cpb(attn)                                             # (h, N, N)
    if _flex_ready(attn, q, dt, mask, shift):                           # mask unused: the graph drops it
        y = _flex(q.to(dt), k.to(dt), v.to(dt), cpb, shift if mask is not None else None)
        return y, v                                                     # (nW·B, h, N, d)
    if mask is None:
        bias = cpb.to(dt).unsqueeze(0)                                  # (1, h, N, N), broadcast over windows
        return _sdpa(q.to(dt), k.to(dt), v.to(dt), bias), v             # (nW·B, h, N, d)
    # The sum grows with the number of windows (≈ 1 GB per 30 s in FP32 over all blocks): rebuilt on
    # every call, never kept (one elementwise kernel, ~1-4% of an eager call).
    nW = mask.shape[0]
    B = B_ // nW
    bias = (cpb.unsqueeze(0) + mask.unsqueeze(1)).to(dt)                # (nW, h, N, N)
    bias = bias.reshape(1, nW * h, N, N)                                # (1, nW·h, N, N), broadcast over B
    qh, kh, vh = (t.to(dt).reshape(B, nW * h, N, -1) for t in (q, k, v))  # (B, nW·h, N, d)
    y = _sdpa(qh, kh, vh, bias)                                         # (B, nW·h, N, d)
    return y.reshape(B_, h, N, -1), v                                   # (nW·B, h, N, d)


def sdpa_output(attn: nn.Module, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """XSA (when on: each token's output minus its projection on the token's own value) and the output
    projection of WindowAttention.forward.

    Args:
        attn: The real-valued WindowAttention whose weights are used.
        y: (…, h, d) attention output of each token, heads in channel order.
        v: (…, h, d) value vectors of the same tokens (same dtype as in the original, for XSA).
    Returns:
        (…, C)
    """
    if attn.attention_variant == "xsa":
        y = attn._apply_xsa(y, v)                                       # same v (dtype included) as the original
    return attn.proj_drop(attn.proj(y.flatten(-2)))                     # (…, C)


def sdpa_forward(attn: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                 shift: Optional[ShiftGeometry] = None) -> torch.Tensor:
    """WindowAttention.forward of a real-valued module through F.scaled_dot_product_attention.

    :func:`sdpa_qkv` → :func:`sdpa_attend` → :func:`sdpa_output` (see there for the formula).

    Args:
        attn: The real-valued WindowAttention whose weights are used.
        x: (nW·B, N, C) window tokens, window index fastest (window_partition order).
        mask: (nW, N, N) cyclic-shift mask (0 / -100) or None.
        shift: Geometry of ``mask``, for the FlexAttention path; None without a mask or to keep SDPA.
    Returns:
        (nW·B, N, C)
    """
    qkv, dt = sdpa_qkv(attn, x)                                         # (nW·B, N, 3C)
    y, v = sdpa_attend(attn, qkv, mask, dt, shift)                      # (nW·B, h, N, d) each
    return sdpa_output(attn, y.transpose(1, 2), v.transpose(1, 2))      # (nW·B, N, C)


def _refresh_after_load(attn: nn.Module, incompatible_keys) -> None:
    """load_state_dict post-hook: new weights → new cached bias (also for an already compiled model)."""
    if attn.sdpa:
        refresh_cpb(attn)


def enable_sdpa(module: nn.Module, enabled: bool = True, *, fused_bias: bool = True,
                shared_phases: bool = True, flex: bool = True) -> int:
    """Route every real-valued WindowAttention of ``module`` through SDPA (or back).

    Inference-time only, like ``enable_varlen``: no weight is touched, and training mode or
    autograd always runs the original forward. The CPB bias is computed here, in full FP32, and
    cached outside the state_dict; it is recomputed when the weights are loaded, moved, cast or
    (by eager calls) edited in place. Not seen: edits through ``.data``, parameters replaced by
    hand, and in-place edits followed only by torch.compile calls (a compiled graph checks the
    cache's device and dtype, not the weights); call this function again after them.
    ``enabled=False`` restores the original forward exactly and frees the cache.

    Options of the SDPA path, equal in exact arithmetic and on with it by default:
    ``fused_bias`` adds the Q/V bias in the QKV GEMM (modules where :func:`fusable_qkv_bias` holds);
    ``shared_phases`` makes the multi-phase blocks of variable-length inference
    (sage/model/swin/varlen.py) project QKV and output once for all phases instead of once per phase;
    ``flex`` runs the attention through FlexAttention inside torch.compile graphs on CUDA (modules where
    :func:`flex_supported` holds; eager calls keep SDPA, see :func:`_flex_ready`).

    Args:
        module: Any module containing WindowAttention — typically the whole SAGEAutoencoder.
        enabled: True = SDPA path, False = original forward (options off too).
        fused_bias: Q/V bias added by the QKV GEMM.
        shared_phases: One QKV and one output projection per multi-phase block.
        flex: FlexAttention in compiled graphs.

    Returns:
        Number of attention modules affected (complex-valued ones are never touched).
    """
    from sage.model.swin.attention import WindowAttention
    from sage.model.swin.block import SwinTransformerBlock

    for block in module.modules():
        if isinstance(block, SwinTransformerBlock):
            block.shared_phases = enabled and shared_phases and not block.is_complex
    count = 0
    for attn in module.modules():
        if isinstance(attn, WindowAttention) and not attn.is_complex:
            attn.sdpa = enabled
            attn.fused_bias = enabled and fused_bias and fusable_qkv_bias(attn)
            attn.flex = enabled and flex and flex_supported(attn)
            for name in ("_sdpa_cpb", "_sdpa_sources", "_sdpa_stamp"):
                attn.__dict__.pop(name, None)
            hooks = attn._load_state_dict_post_hooks
            for hook_id in [i for i, hook in hooks.items() if hook is _refresh_after_load]:
                del hooks[hook_id]
            if enabled:
                refresh_cpb(attn)                                       # outside any torch.compile graph
                attn.register_load_state_dict_post_hook(_refresh_after_load)
            count += 1
    return count


def disable_sdpa(module: nn.Module) -> int:
    """Restore the original attention everywhere. See :func:`enable_sdpa`."""
    return enable_sdpa(module, False)
