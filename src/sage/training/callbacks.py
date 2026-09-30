import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
import json
import torch
import copy
from pathlib import Path
from rich.console import Console

from sage.utils.console import ok, warn
from sage.utils.model_info import extract_model_config, log_compression_stats

console = Console()

class DatasetEpochSetter(Callback):
    """Sets the epoch number on the train dataset only, to vary random crops across epochs.
    Val dataset is intentionally kept at epoch=0 (fixed seed) so val metrics are
    comparable across epochs — the same segments are always evaluated.
    """
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.train_dataloader is not None:
            dl = trainer.train_dataloader
            if hasattr(dl, "dataset") and hasattr(dl.dataset, "set_epoch"):
                dl.dataset.set_epoch(trainer.current_epoch)

class MultiCorpusEpochSetter(Callback):
    """Multi-corpus epoch wiring (train only).

    Advances BOTH the rotating sampler — so it selects the next M4Singer chunk and
    reshuffles the corpora for this epoch — and the ``MultiCorpusDataset`` children —
    so the 1.5 s crop window varies per epoch (the sampler lives in the main process;
    the dataset epoch reaches workers via its shared-memory counter). Each DDP rank
    runs this on its own sampler instance. Validation (FMA-only) never calls set_epoch,
    so it stays at epoch 0 → fixed crops, exactly like the single-corpus path.
    """
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dl = trainer.train_dataloader
        if dl is None:
            return
        epoch = trainer.current_epoch
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = getattr(dl, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)


class ModelInfoLogger(pl.Callback):
    """Log model info and structure at the beginning of training.
    - prints summary to console
    - saves a JSON with parameter counts and module list
    - optionally logs model structure to W&B
    """
    def __init__(self, filename: str = "model_info.json", max_module_lines: int = 500, log_structure: bool = False):
        super().__init__()
        self.filename = filename
        self.max_module_lines = max_module_lines
        self.log_structure = bool(log_structure)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # Extract info from the core model (autoencoder inside wrapper)
        model = getattr(pl_module, "autoencoder", pl_module)
        info_dict = extract_model_config(model)

        # Limit how many module lines to display
        modules = info_dict.get("modules", [])[: self.max_module_lines]

        # Console output: parameter summary + structure
        console.rule("[bold cyan]Model info")
        console.print(f"params total/trainable: {info_dict.get('num_parameters_total')}/{info_dict.get('num_parameters_trainable')}")
        console.print(f"model size (bytes): {info_dict.get('model_bytes')}")
        console.rule("[bold cyan]Model structure")
        if self.log_structure:
            console.print("[MODEL SUMMARY]\n", info_dict["repr"])
        console.rule()

        # Save JSON to disk
        try:
            # the logger's folder if any, else the run folder
            base_dir = Path(getattr(trainer.logger, "save_dir", "") or trainer.default_root_dir or ".")
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / self.filename
            out_path.write_text(json.dumps(info_dict, indent=2))
            ok(f"ModelInfoLogger: saved model info JSON to {str(out_path)}", prefix="TRAINER")
        except Exception as e:
            warn(f"ModelInfoLogger: not able to save JSON ({type(e).__name__}: {e})", prefix="TRAINER")
            out_path = None

        # log to W&B (if used)
        if isinstance(trainer.logger, WandbLogger):
            try:
                run = trainer.logger.experiment
                # Update run config with full model info
                run.config.update({"model_info": info_dict}, allow_val_change=True)
                # Log structure as preformatted text
                run.log(
                    {"model/structure": info_dict["repr"]},
                    step=int(getattr(trainer, "global_step", 0)),
                    commit=False,  # do not create a new step yet
                )
                # Upload JSON artifact
                if out_path is not None:
                    run.save(str(out_path), base_path=str(base_dir))
            except Exception as e:
                warn(f"ModelInfoLogger: W&B log skipped ({type(e).__name__}: {e})", prefix="TRAINER")

class CompressionStatsLogger(Callback):
    """Runs a dummy encoder forward pass at fit start to log latent shape and compression rate.
    Executed only on rank 0, after Lightning has moved the model to the correct device.
    """
    def __init__(self, train_dl, console: Console):
        super().__init__()
        self._train_dl = train_dl
        self._console = console

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log_compression_stats(pl_module, self._train_dl, self._console)


class EMACallback(Callback):
    """Exponential Moving Average (EMA) of model weights.
    
    Maintains a shadow copy of the autoencoder weights.
    Swaps the EMA weights into the active model during validation so metrics
    and audio demos are generated using the smoothed weights.
    Injects the EMA state dict into the Lightning checkpoint automatically.
    """
    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_state_dict = {}
        self.original_state_dict = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "autoencoder", None)
        if model is None:
            warn("EMACallback: autoencoder not found in pl_module. Disabling EMA.", prefix="EMA")
            return

        if self.ema_state_dict:
            # EMA already restored from the checkpoint: PL restores module+callbacks
            # BEFORE on_fit_start, so re-initializing here would silently reset the
            # EMA to the live weights at every resume. Checkpoint tensors are loaded
            # on CPU → move them to the model device for the in-place mul_/add_
            # updates in on_train_batch_end.
            device = next(model.parameters()).device
            self.ema_state_dict = {k: v.to(device) for k, v in self.ema_state_dict.items()}
            ok(f"EMA state preserved from checkpoint ({len(self.ema_state_dict)} tensors → {device}).", prefix="EMA")
            return

        ok(f"Initializing EMA model with decay={self.decay}...", prefix="EMA")
        # Use a dict of detached tensors instead of deepcopy to survive torch.compile/DDP
        self.ema_state_dict = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        model = getattr(pl_module, "autoencoder", None)
        if model is None or not self.ema_state_dict:
            return

        # Only advance the EMA when the generator actually stepped this batch.
        # With a discriminator the G/D phases alternate, so updating every batch
        # would decay the EMA on disc batches (where the weights are unchanged),
        # effectively halving the EMA half-life in generator-update terms. SAO
        # updates the EMA exclusively inside its generator branch. Default True
        # keeps the no-disc path (every batch is a gen step) unchanged.
        if not getattr(pl_module, "_ema_update_this_batch", True):
            return

        decay = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.ema_state_dict:
                    ema_v = self.ema_state_dict[k]
                    if v.dtype.is_floating_point or v.dtype.is_complex:
                        ema_v.mul_(decay).add_(v.detach(), alpha=1.0 - decay)
                    else:
                        ema_v.copy_(v.detach())

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "autoencoder", None)
        if model is None or not self.ema_state_dict:
            return
            
        # Save original training weights in RAM
        self.original_state_dict = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        
        # Load EMA weights into the active model for validation
        model.load_state_dict(self.ema_state_dict)
        
        if getattr(self, "_logged_swap_in", False) is False:
            ok("Swapped weights to EMA for validation metrics.", prefix="EMA")
            self._logged_swap_in = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "autoencoder", None)
        if model is None or self.original_state_dict is None:
            return
            
        # Restore training weights
        model.load_state_dict(self.original_state_dict)
        self.original_state_dict = None
        
        if getattr(self, "_logged_swap_out", False) is False:
            ok("Restored training weights after validation.", prefix="EMA")
            self._logged_swap_out = True

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        # SAGELightningModule.on_load_checkpoint strips ema_autoencoder.* from state_dict
        # and stashes them in checkpoint["_ema_state"] so we can recover them here.
        ema_state = checkpoint.pop("_ema_state", {})
        if ema_state:
            self.ema_state_dict = ema_state
            ok(f"Restored EMA state from checkpoint ({len(self.ema_state_dict)} tensors).", prefix="EMA")

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        # Inject EMA weights cleanly into the checkpoint under 'ema_autoencoder.*'
        if self.ema_state_dict:
            for k, v in self.ema_state_dict.items():
                checkpoint["state_dict"][f"ema_autoencoder.{k}"] = v
