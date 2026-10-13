# ===============================================================
# Opt-in SDPA path of WindowAttention (inference only). Same formula as
# WindowAttention.forward (Swin V2 cosine attention + Log-CPB bias + cyclic-shift
# mask + XSA), computed by F.scaled_dot_product_attention: one fused kernel on
# CUDA. Equal in exact arithmetic, not bit-identical; off by default.
# The Log-CPB bias depends on the weights only: it is computed once in full FP32
# and cached, and recomputed whenever the weights, their dtype or device change.
# ===============================================================

import math
import threading
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRECISION_LOCK = threading.Lock()           # the matmul precision is a process-wide setting


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


def _cached_cpb(attn: nn.Module) -> torch.Tensor:
    """The cached CPB bias, (h, N, N). Eager: checked against the weights on every call. Inside a
    torch.compile graph the check cannot run: the cache (refreshed by enable_sdpa, load_state_dict,
    .to() and dtype casts) is read as an input; a mismatch left anyway is recomputed in the graph."""
    if torch.compiler.is_compiling():
        cpb = getattr(attn, "_sdpa_cpb", None)
        weight = attn.cpb_mlp[-1].weight
        if cpb is not None and cpb.device == weight.device and cpb.dtype == weight.dtype:
            return cpb
        return cpb_bias(attn)
    sources = getattr(attn, "_sdpa_sources", None)
    if sources is None or attn._sdpa_stamp != _stamp(sources):
        return refresh_cpb(attn)
    return attn._sdpa_cpb


def sdpa_forward(attn: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """WindowAttention.forward of a real-valued module through F.scaled_dot_product_attention.

    Original:  softmax(s·(q̂·k̂ᵀ) + B_cpb + M_shift)·v   (s = per-head temperature, q̂/k̂ = L2-normalised q/k)
    Here:      SDPA(Q = s·q̂, K = k̂, V = v, attn_mask = B_cpb + M_shift, scale = 1)
    Equal in exact arithmetic because s is a per-head scalar: (s·q̂)·k̂ᵀ = s·(q̂·k̂ᵀ).

    Args:
        attn: The real-valued WindowAttention whose weights are used.
        x: (nW·B, N, C) window tokens, window index fastest (window_partition order).
        mask: (nW, N, N) cyclic-shift mask (0 / -100) or None.
    Returns:
        (nW·B, N, C)
    """
    B_, N, C = x.shape                                                  # (nW·B, N, C)
    h = attn.num_heads
    qkv = attn.qkv(x)                                                   # (nW·B, N, 3C)
    dt = qkv.dtype                                                      # bf16/fp16 under autocast, else the weights'
    if attn.q_bias is not None:
        qkv = qkv + torch.cat((attn.q_bias, torch.zeros_like(attn.v_bias), attn.v_bias))
    qkv = qkv.reshape(B_, N, 3, h, -1).permute(2, 0, 3, 1, 4)           # (3, nW·B, h, N, d)
    q, k, v = qkv[0], qkv[1], qkv[2]                                    # (nW·B, h, N, d)
    s = torch.clamp(attn.logit_scale, max=math.log(1.0 / 0.01)).exp()  # (h, 1, 1)
    q = F.normalize(q, dim=-1) * s                                      # (nW·B, h, N, d)  s·q̂
    k = F.normalize(k, dim=-1)                                          # (nW·B, h, N, d)  k̂
    cpb = _cached_cpb(attn)                                             # (h, N, N)
    if mask is None:
        bias = cpb.to(dt).unsqueeze(0)                                  # (1, h, N, N), broadcast over windows
        y = F.scaled_dot_product_attention(q.to(dt), k.to(dt), v.to(dt), attn_mask=bias, scale=1.0)
    else:
        # The sum grows with the number of windows (≈ 1 GB per 30 s in FP32 over all blocks): rebuilt on
        # every call, never kept (one elementwise kernel, ~1-4% of an eager call).
        nW = mask.shape[0]
        B = B_ // nW
        bias = (cpb.unsqueeze(0) + mask.unsqueeze(1)).to(dt)            # (nW, h, N, N)
        bias = bias.reshape(1, nW * h, N, N)                            # (1, nW·h, N, N), broadcast over B
        qh, kh, vh = (t.to(dt).reshape(B, nW * h, N, -1) for t in (q, k, v))  # (B, nW·h, N, d)
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias, scale=1.0)
        y = y.reshape(B_, h, N, -1)                                     # (nW·B, h, N, d)
    if attn.attention_variant == "xsa":
        y = attn._apply_xsa(y, v)                                       # same v (dtype included) as the original
    y = y.transpose(1, 2).reshape(B_, N, C)                             # (nW·B, N, C)
    return attn.proj_drop(attn.proj(y))                                 # (nW·B, N, C)


def _refresh_after_load(attn: nn.Module, incompatible_keys) -> None:
    """load_state_dict post-hook: new weights → new cached bias (also for an already compiled model)."""
    if attn.sdpa:
        refresh_cpb(attn)


def enable_sdpa(module: nn.Module, enabled: bool = True) -> int:
    """Route every real-valued WindowAttention of ``module`` through SDPA (or back).

    Inference-time only, like ``enable_varlen``: no weight is touched, and training mode or
    autograd always runs the original forward. The CPB bias is computed here, in full FP32, and
    cached outside the state_dict; it is recomputed when the weights are loaded, moved, cast or
    (by eager calls) edited in place. Not seen: edits through ``.data``, parameters replaced by
    hand, and in-place edits followed only by torch.compile calls (a compiled graph checks the
    cache's device and dtype, not the weights); call this function again after them.
    ``enabled=False`` restores the original forward exactly and frees the cache.

    Args:
        module: Any module containing WindowAttention — typically the whole SAGEAutoencoder.
        enabled: True = SDPA path, False = original forward.

    Returns:
        Number of attention modules affected (complex-valued ones are never touched).
    """
    from sage.model.swin.attention import WindowAttention

    count = 0
    for attn in module.modules():
        if isinstance(attn, WindowAttention) and not attn.is_complex:
            attn.sdpa = enabled
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
