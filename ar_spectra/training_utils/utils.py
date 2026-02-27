from pathlib import Path
from typing import Mapping, Optional, Tuple
import os
import re
import warnings

import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger, CometLogger

from ..interface.aeiou import pca_point_cloud

def get_rank():
    """Get rank of current process."""
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()

def _is_rank0() -> bool:
    return get_rank() == 0

def get_checkpoint_dir(base_dir: str | Path, run_name: str) -> Path:
    """
    Get the checkpoint directory based on an explicit run name.

    Args:
        base_dir: Base directory for checkpoints.
        run_name: The resolved run name (same across all ranks).

    Returns:
        Path to the checkpoint directory named after the run.
    """
    checkpoint_path = Path(base_dir) / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path

def parse_run_name(run_name: str) -> tuple[float | None, float | None]:
    """Extract beta and lambda from run name.

    Supports two formats:
    - 'beta0.5-lambda0.25' -> (0.5, 0.25)
    - 'rate60-lambda0.5-b1' -> (None, 0.5)  # beta is None for rate-based runs

    Returns:
        Tuple of (beta, lambda). Beta may be None for rate-based runs.
        Returns (None, None) if the format is not recognized.
    """
    # Try beta-lambda format first
    match = re.match(r"beta([\d.]+)-lambda([\d.]+)", run_name)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Try rate-lambda-b format (GECO runs)
    match = re.match(r"rate[\d.]+-lambda([\d.]+)-b[\d.]+", run_name)
    if match:
        return None, float(match.group(1))

    return None, None

def get_run_names(basedir: Path, pattern: str) -> list[str]:
    """Get run names matching pattern, or default grid if None.

    Args:
        pattern: Regex pattern to filter run names. If None, uses default grid.

    Returns:
        List of run names.
    """
    assert pattern is not None
    # if pattern is None:
    #     # Default: all beta/lambda combinations
    #     betas = [0.1, 0.5, 1, 2, 5]
    #     lambdas = [0, 0.25, 0.5, 0.75, 1]
    #     return [f"beta{b}-lambda{l}" for b in betas for l in lambdas]

    # Find all run directories matching the pattern
    regex = re.compile(pattern)
    runs = []
    for run_dir in basedir.iterdir():
        if run_dir.is_dir() and regex.search(run_dir.name):
            runs.append(run_dir.name)

    if len(runs) == 0:
        print(f"No runs found matching pattern: {pattern}")
        return []

    return sorted(runs)

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
    """
    Build a deterministic run name from the model config name and loss weights.
    """
    model_part = _sanitize_token(model_name)
    if not loss_config:
        return model_part

    weights = _extract_loss_weights(loss_config)
    if not weights:
        return model_part

    weights = sorted(weights, key=lambda item: item[0])
    weight_tokens = []
    for name, value in weights:
        safe_name = _sanitize_token(name)
        value_str = f"{value:g}"
        weight_tokens.append(f"{safe_name}{value_str}")

    return "-".join([model_part] + weight_tokens)

def resolve_run_name(cfg) -> str:
    """
    Resolve run name using Hydra choices + loss_config in cfg.
    """
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

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

def logger_project_name(logger) -> str:
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name

def log_metric(logger, key, value, step=None):
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if not _is_rank0():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)

def log_audio(logger, key, audio_path, sample_rate, caption=None):
    if not _is_rank0():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)

def log_image(logger, key, img_data):
    if not _is_rank0():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Image(img_data)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)

def log_point_cloud(logger, key, tokens, caption=None):
    if not _is_rank0():
        return
    try:
        if isinstance(logger, WandbLogger):
            point_cloud = pca_point_cloud(tokens)  
            logger.experiment.log({key: point_cloud})
        elif isinstance(logger, CometLogger):
            point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
            # logger.experiment.log_points_3d(scene_name=key, points=point_cloud)
    except Exception as e:
        warnings.warn(f"Skipping point cloud logging: {type(e).__name__}: {e}")
        pass

from rich.console import Console
console = Console()  

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")
