import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
import json
from pathlib import Path
from rich.console import Console

from ar_spectra.utils.console import ok, warn
from ar_spectra.utils.model_info import extract_model_config

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
            # prova a usare la cartella di logging; fallback alla root di lavoro
            base_dir = Path(getattr(trainer.logger, "save_dir", "") or trainer.default_root_dir or ".")
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / self.filename
            out_path.write_text(json.dumps(info_dict, indent=2))
            ok(f"ModelInfoLogger: saved model info JSON to {str(out_path)}", prefix="TRAINER")
        except Exception as e:
            warn(f"ModelInfoLogger: not able to save JSON ({type(e).__name__}: {e})", prefix="TRAINER")
            out_path = None

        # logga su W&B (se presente)
        if isinstance(trainer.logger, WandbLogger):
            try:
                run = trainer.logger.experiment
                # Update run config with full model info
                run.config.update({"model_info": info_dict}, allow_val_change=True)
                # Log structure as preformatted text
                run.log(
                    {"model/structure": model_cfg["repr"]},
                    step=int(getattr(trainer, "global_step", 0)),
                    commit=False,  # do not create a new step yet
                )
                # Upload JSON artifact
                if out_path is not None:
                    run.save(str(out_path), base_path=str(base_dir))
            except Exception as e:
                warn(f"ModelInfoLogger: W&B log skipped ({type(e).__name__}: {e})", prefix="TRAINER")
