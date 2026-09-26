
# =============================================================================
# Distributed rank helpers and run-name utilities.
# =============================================================================

from pathlib import Path
from typing import Mapping, Optional
import os
import re

import torch
from omegaconf import OmegaConf


def get_rank() -> int:
    """Get rank of current process."""
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Get the total number of distributed processes.

    Mirrors ``get_rank``: prefers the SLURM env (set on CINECA before the process
    group is initialized, so it is correct at dataloader-build time), then the
    initialized torch.distributed group, else 1 (single process).
    """
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def _is_rank0() -> bool:
    return get_rank() == 0


def get_checkpoint_dir(base_dir: str | Path, run_name: str) -> Path:
    """
    Get (and create) the checkpoint directory for a given run name.
    """
    checkpoint_path = Path(base_dir) / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def _sanitize_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(token)).strip("_")
    return cleaned or "run"


def _extract_loss_weights(loss_config: Mapping) -> list[tuple[str, float]]:
    weights: list[tuple[str, float]] = []

    def visit(node: Mapping, prefix: str) -> None:
        for key, value in node.items():
            if key == "weights" and isinstance(value, Mapping):
                for w_key, w_val in value.items():
                    if isinstance(w_val, (int, float)):
                        name = f"{prefix}.{w_key}" if prefix else str(w_key)
                        weights.append((name, float(w_val)))
            elif isinstance(value, Mapping):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(value, next_prefix)

    visit(loss_config, "")
    return weights


def build_run_name(model_name: str, loss_config: Optional[Mapping] = None) -> str:
    """Build a deterministic run name from the model name and loss weights."""
    model_part = _sanitize_token(model_name)
    if not loss_config:
        return model_part

    weights = _extract_loss_weights(loss_config)
    if not weights:
        return model_part

    weights = sorted(weights, key=lambda item: item[0])
    weight_tokens = [f"{_sanitize_token(name)}{value:g}" for name, value in weights]
    return "-".join([model_part] + weight_tokens)


def resolve_run_name(cfg) -> str:
    """Resolve run name using Hydra choices + loss_config in cfg."""
    from hydra.core.hydra_config import HydraConfig

    model_name = None
    try:
        model_name = HydraConfig.get().runtime.choices.get("model")
    except Exception:
        model_name = None

    if not model_name:
        model_name = "model"

    loss_config = None
    try:
        trainer_cfg = cfg.get("trainer") if hasattr(cfg, "get") else None
        if trainer_cfg is not None and hasattr(trainer_cfg, "get"):
            loss_config = trainer_cfg.get("loss_config")
    except Exception:
        loss_config = None

    if loss_config is not None:
        loss_config = OmegaConf.to_container(loss_config, resolve=True)

    return build_run_name(model_name, loss_config)
