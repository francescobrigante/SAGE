# ==============================================================
# Console utils
#
#   contains utils for printing messages to the console
# ==============================================================

from rich.console import Console

_console = Console()

def ok(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold green")

def warn(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold yellow")

def err(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold red")

def info(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="cyan")


# =============================================================================
# Logger helpers (WandB / CometML)
# =============================================================================

import warnings as _warnings


def _is_rank0_local() -> bool:
    from .run_config import _is_rank0
    return _is_rank0()


def logger_project_name(logger) -> str:
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name


def log_metric(logger, key, value, step=None):
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if not _is_rank0_local():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)


def log_audio(logger, key, audio_path, sample_rate, caption=None):
    import wandb
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if not _is_rank0_local():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)


def log_image(logger, key, img_data):
    import wandb
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    if not _is_rank0_local():
        return
    if isinstance(logger, WandbLogger):
        logger.experiment.log({key: wandb.Image(img_data)})
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)


def log_point_cloud(logger, key, tokens, caption=None):
    from pytorch_lightning.loggers import WandbLogger, CometLogger
    from .aeiou import pca_point_cloud
    if not _is_rank0_local():
        return
    try:
        if isinstance(logger, WandbLogger):
            point_cloud = pca_point_cloud(tokens)
            logger.experiment.log({key: point_cloud})
        elif isinstance(logger, CometLogger):
            point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
    except Exception as e:
        _warnings.warn(f"Skipping point cloud logging: {type(e).__name__}: {e}")
