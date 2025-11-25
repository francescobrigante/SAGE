"""Utilities for deterministic experiment setup."""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch


def configure_reproducibility(
    seed: int,
    *,
    deterministic: bool = True,
    strict_deterministic: bool = True,
    warn: Optional[Callable[[str], None]] = None,
) -> None:
    """Set seeds and backend flags to favour reproducible behaviour.

    Args:
        seed: Global seed used for Python, NumPy and PyTorch RNGs.
        deterministic: When ``True`` configure CuDNN/CuBLAS for stable kernels.
            When ``False`` only the RNG seeds are initialised.
        strict_deterministic: Retained for backwards compatibility; currently
            unused because deterministic kernels are not enforced globally.
        warn: Optional callable used to surface non-fatal warnings back to the
            caller. When omitted warnings are emitted via ``print``.
    """

    def _emit(message: str) -> None:
        if warn is not None:
            warn(message)
        else:
            print(message)

    # Propagate seed to the core RNGs used across the codebase.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if not deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return

    # Configure deterministic-friendly execution for known CUDA libraries.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Explicit deterministic algorithm enforcement is intentionally skipped to
    # avoid runtime errors with kernels lacking deterministic implementations.
    if strict_deterministic:
        _emit(
            "strict_deterministic requested but global deterministic algorithms"
            " enforcement is disabled; seeds/cudnn flags remain set."
        )
    try:
        torch.use_deterministic_algorithms(False)
    except AttributeError:
        # Older PyTorch releases did not provide this toggle.
        pass
