# ===============================================================
# Variable-length inference for collapsed Swin stages.
# At the deepest stage the attention window spans the whole training
# segment, so the shift is frozen to 0 and longer audio gets tiled into
# disjoint 1.486 s blocks. Here: multi-phase attention + per-token
# combination weights that remove the resulting periodic seams.
# ===============================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

COMBINE_MODES: Tuple[str, ...] = ("mean", "tri", "hard")


@dataclass(frozen=True)
class VarlenConfig:
    """Multi-phase attention settings for one block.

    Args:
        phases: Grid offsets, in time tokens, at which the attention branch is
            re-run. Must be distinct and inside ``[0, window_time)``. Phase 0 is
            the block's native partition and is the only one that never wraps,
            so it should always be present.
        combine: How the per-phase outputs are mixed per token —
            ``"mean"`` uniform, ``"tri"`` proportional to ``d + 1``,
            ``"hard"`` winner-takes-all on ``d``, where ``d`` is the token's
            distance to the nearest edge of its own attention group.
    """

    phases: Tuple[int, ...]   # grid offsets in time tokens, e.g. (0, 8, 16, 24)
    combine: str              # one of COMBINE_MODES

    def __post_init__(self) -> None:
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"Unsupported combine={self.combine!r}; expected one of {COMBINE_MODES}")
        if len(self.phases) == 0:
            raise ValueError("phases must not be empty")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError(f"phases must be distinct, got {self.phases}")
        if any(p < 0 for p in self.phases):
            raise ValueError(f"phases must be non-negative, got {self.phases}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VarlenConfig":
        """Build from a Hydra/OmegaConf node such as ``{phases: [0, 16], combine: tri}``."""
        return cls(phases=tuple(int(p) for p in mapping["phases"]), combine=str(mapping["combine"]))


# --------------------------------------------------------------------------
# Per-token combination weights
# --------------------------------------------------------------------------

def group_bounds(time_tokens: int, window: int, phase: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention-group bounds of every real token, for one grid phase.

    Reproduces exactly what ``SwinTransformerBlock._attention_branch`` does for a
    given shift: zero-pad the time axis up to a multiple of the window, roll by
    ``-phase``, partition, and split the wrapped last window into two regions with
    the attention mask. The group of a token is the set of tokens it can actually
    attend to along time — which for the wrapped head and for a truncated tail is
    *smaller* than the window. Padding tokens are excluded: they carry no signal,
    so a group that runs into the padding is treated as ending at the last real
    token.

    Args:
        time_tokens: Number of real time tokens W at this stage.
        window: Window size along time (ww).
        phase: Grid offset in tokens.

    Returns:
        (start, end) — two ``(W,)`` int64 tensors with the half-open bounds
        ``[start, end)`` of each token's group, in real-token coordinates.
    """
    n_windows = (time_tokens + window - 1) // window
    padded = n_windows * window                                    # what the block partitions
    t = torch.arange(time_tokens, dtype=torch.int64)
    rolled = (t - phase) % padded                                  # position after torch.roll(-phase)
    win_idx = rolled // window                                     # window each token lands in

    if phase > 0:
        # Same three regions as the attention mask: everything before the last
        # window, the genuine tail, and the head tokens wrapped in behind it.
        region = (rolled >= padded - window).to(torch.int64) + (rolled >= padded - phase).to(torch.int64)
        group = win_idx * 3 + region
    else:
        group = win_idx                                            # no roll, no mask, one group per window

    # Every group is a contiguous run in real-token coordinates; take maximal runs
    # so a group truncated by the padding ends at the last real token.
    is_new = torch.ones(time_tokens, dtype=torch.bool)
    is_new[1:] = group[1:] != group[:-1]
    run = torch.cumsum(is_new.to(torch.int64), 0) - 1              # run index of each token
    counts = torch.bincount(run)
    starts = torch.cumsum(counts, 0) - counts
    return starts[run], (starts + counts)[run]


def phase_weights(time_tokens: int, window: int, cfg: VarlenConfig) -> torch.Tensor:
    """Per-token, per-phase combination weights.

    The quantity that decides the weight is ``d``, the token's distance to the
    nearest edge of its attention group. A token with ``d = 0`` sits at a window
    edge and is reconstructed as if the audio ended there — that is exactly the
    periodic "hole". The weights are a deterministic function of position only:
    no dependence on content, so they are computed once per sequence length.

    Args:
        time_tokens: Number of real time tokens W at this stage.
        window: Window size along time (ww).
        cfg: Phases and combination mode.

    Returns:
        ``(P, W)`` float32 tensor summing to 1 over the phase axis.
    """
    dists = []
    for phase in cfg.phases:
        start, end = group_bounds(time_tokens, window, phase)       # (W,), (W,)
        t = torch.arange(time_tokens, dtype=torch.int64)
        dists.append(torch.minimum(t - start, end - 1 - t))         # (W,) distance to group edge
    d = torch.stack(dists).to(torch.float32)                        # (P, W)

    if cfg.combine == "mean":
        w = torch.ones_like(d)                                      # (P, W)
    elif cfg.combine == "tri":
        w = d + 1.0                                                 # (P, W) linear in the margin
    else:  # "hard"
        w = (d == d.max(dim=0, keepdim=True).values).to(d.dtype)    # (P, W) ties shared equally
    return w / w.sum(dim=0, keepdim=True)                           # (P, W)


# --------------------------------------------------------------------------
# Enabling / disabling on a loaded model
# --------------------------------------------------------------------------

def is_collapsed(block: nn.Module) -> bool:
    """True when the block's time window covers its whole training-time grid.

    That is the condition under which ``SwinTransformerBlock``'s collapse guard
    zeroes the shift, which is harmless at the training length and is what breaks
    variable-length inference. Stages whose window is genuinely local still
    alternate W-MSA / SW-MSA and need no correction.
    """
    return block.window_size[1] >= block.input_resolution[1]


def enable_varlen(module: nn.Module, cfg: Optional[VarlenConfig]) -> int:
    """Attach (or detach) a multi-phase config to every collapsed block.

    Inference-time only: no weight is touched and no architecture is rebuilt, so
    this can be applied to any loaded checkpoint. Blocks whose window is local are
    left alone. Passing ``cfg=None`` restores the original behaviour exactly.

    Args:
        module: Any module containing SwinTransformerBlocks — typically the whole
            ``SAGEAutoencoder``, which covers both encoder and decoder.
        cfg: Config to attach, or None to disable.

    Returns:
        Number of blocks affected.
    """
    from sage.model.swin.block import SwinTransformerBlock

    count = 0
    for block in module.modules():
        if not isinstance(block, SwinTransformerBlock) or not is_collapsed(block):
            continue
        block.varlen = cfg
        block._varlen_weight_cache.clear()
        count += 1
    return count


def disable_varlen(module: nn.Module) -> int:
    """Restore single-phase attention everywhere. See :func:`enable_varlen`."""
    return enable_varlen(module, None)


DEFAULT_CONFIG = Path(__file__).with_name("varlen.yaml")   # shipped with the package


def load_config(config_path=None):
    """Read the varlen presets (default: ``varlen.yaml`` next to this file).

    Kept out of this file on purpose: phases, combination modes and the variant
    list are configuration, not code.
    """
    from omegaconf import OmegaConf

    return OmegaConf.load(config_path or DEFAULT_CONFIG).varlen


def load_presets(config_path=None) -> Mapping[str, Any]:
    """Preset table from varlen.yaml: name → {phases, combine}."""
    return load_config(config_path).presets


def resolve(mode: str, presets: Optional[Mapping[str, Any]] = None) -> Optional[VarlenConfig]:
    """Map a variant name (e.g. ``"tri4"``, or ``"off"``) to a config.

    Args:
        mode: Preset name, or ``"off"`` / ``"baseline"`` for single-phase.
        presets: Preset table; loaded from varlen.yaml when omitted.

    Returns:
        The config, or None when the mode disables the fix.
    """
    if mode in ("off", "baseline", "none"):
        return None
    table = load_presets() if presets is None else presets
    if mode not in table:
        raise KeyError(f"Unknown varlen mode {mode!r}; available: {['off'] + list(table)}")
    return VarlenConfig.from_mapping(table[mode])


def applied_phases(time_tokens: int, window: int, cfg: Optional[VarlenConfig]) -> Sequence[int]:
    """Phases that would actually run at this length — empty when the fix is inert."""
    if cfg is None or time_tokens <= window:
        return ()
    return cfg.phases
