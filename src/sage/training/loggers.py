# ==============================================================
# Experiment-logger helpers (Weights & Biases / Comet / TensorBoard)
# used by the training loop. Rank-0 only; no-ops for other loggers.
# ==============================================================

# Optional loggers and utilities
try:
    from pytorch_lightning.loggers import WandbLogger, CometLogger, TensorBoardLogger
except ImportError:
    WandbLogger = None
    CometLogger = None
    TensorBoardLogger = None

try:
    import wandb
except ImportError:
    wandb = None

from sage.utils.console import warn
from sage.utils.run_config import _is_rank0


import warnings as _warnings


def _is_rank0_local() -> bool:
    return _is_rank0()


def logger_project_name(logger) -> str:
    if WandbLogger is not None and isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif CometLogger is not None and isinstance(logger, CometLogger):
        return logger.name


def log_metric(logger, key, value, step=None):
    if not _is_rank0_local():
        return
    if WandbLogger is not None and isinstance(logger, WandbLogger):
        logger.experiment.log({key: value})
    elif CometLogger is not None and isinstance(logger, CometLogger):
        logger.experiment.log_metrics({key: value}, step=step)


def log_histogram(logger, key: str, values, step: int | None = None):
    """Log a 1-D histogram (e.g. KL per channel) to TensorBoard or W&B."""
    if not _is_rank0_local():
        return
    try:
        import torch as _torch
        vals_cpu = values.detach().cpu() if isinstance(values, _torch.Tensor) else _torch.tensor(values)
        if WandbLogger is not None and isinstance(logger, WandbLogger):
            if wandb is not None:
                logger.experiment.log({key: wandb.Histogram(vals_cpu.numpy())})
        elif TensorBoardLogger is not None and isinstance(logger, TensorBoardLogger):
            logger.experiment.add_histogram(key, vals_cpu, global_step=step)
    except Exception as e:
        _warnings.warn(f"Skipping histogram logging for '{key}': {type(e).__name__}: {e}")


def log_audio(logger, key, audio_path, sample_rate, caption=None):
    if not _is_rank0_local():
        return
    if WandbLogger is not None and isinstance(logger, WandbLogger):
        if wandb is None:
            warn("wandb not installed, cannot log audio to WandbLogger")
            return
        logger.experiment.log({key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)})
    elif CometLogger is not None and isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)


def log_image(logger, key, img_data):
    if not _is_rank0_local():
        return
    if WandbLogger is not None and isinstance(logger, WandbLogger):
        if wandb is None:
            warn("wandb not installed, cannot log image to WandbLogger")
            return
        logger.experiment.log({key: wandb.Image(img_data)})
    elif CometLogger is not None and isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)


def log_point_cloud(logger, key, tokens, caption=None):
    if not _is_rank0_local():
        return
    try:
        from sage.utils.aeiou import pca_point_cloud  # lazy: pulls torchaudio/umap/plotly only when needed
        if WandbLogger is not None and isinstance(logger, WandbLogger):
            point_cloud = pca_point_cloud(tokens)
            logger.experiment.log({key: point_cloud})
        elif CometLogger is not None and isinstance(logger, CometLogger):
            point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
    except Exception as e:
        _warnings.warn(f"Skipping point cloud logging: {type(e).__name__}: {e}")
