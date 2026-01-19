"""Training script for audio autoencoders using Hydra configuration.

All instantiation is done via hydra.utils.instantiate with _target_ configs.
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.profiler as torch_profiler
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, ModelSummary, TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
import wandb

from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.training_utils.autoencoders import AutoencoderTrainingWrapper, AutoencoderValDemoCallback
from ar_spectra.training_utils.initialization import collate_stft
from ar_spectra.training_utils.reproducibility import configure_reproducibility
from ar_spectra.training_utils.get_model_config import extract_model_config

from rich.console import Console
console = Console()

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")


class WandbConfigLogger:
    """Utility per caricare l'intera cartella di configurazione Hydra su W&B.
    - Non crea copie locali dei file
    - Può loggare i contenuti testuali oppure caricare i file come artifact
    - Di default usa un artifact (più pulito nel pannello W&B)
    """
    def __init__(self, conf_root: Path, extensions: tuple = (".yaml", ".yml"), use_artifact: bool = True, log_text: bool = False):
        self.conf_root = conf_root
        self.extensions = extensions
        self.use_artifact = use_artifact
        self.log_text = log_text

    def list_files(self) -> List[Path]:
        if not self.conf_root.exists():
            return []
        return [p for p in self.conf_root.rglob("*") if p.is_file() and p.suffix in self.extensions]

    def load_contents(self) -> Dict[str, str]:
        files = self.list_files()
        out: Dict[str, str] = {}
        for f in files:
            try:
                rel = f.relative_to(self.conf_root)
                key = f"conf/{rel.as_posix()}"
                out[key] = f.read_text()
            except Exception as e:
                warn(f"Skip file {f} ({type(e).__name__}: {e})")
        return out

    def log_to_wandb(self, run):
        data = self.load_contents()
        if not data:
            warn("Nessun file di configurazione trovato da loggare su W&B.")
            return
        rel_paths = list(data.keys())
        # Aggiorna config con la lista dei file (non con il contenuto completo)
        try:
            run.config.update({"hydra_conf_files": rel_paths}, allow_val_change=True)
        except Exception:
            pass
        if self.use_artifact:
            try:
                artifact = wandb.Artifact("hydra-conf", type="config")
                # Aggiunge i file originali senza copiarli altrove
                for f in self.list_files():
                    artifact.add_file(str(f))
                run.log_artifact(artifact)
                ok(f"Caricata cartella conf come artifact W&B ({len(data)} files).")
            except Exception as e:
                warn(f"Artifact upload fallito ({type(e).__name__}: {e}); provo fallback testuale.")
                self._fallback_text(run, data)
        elif self.log_text:
            self._fallback_text(run, data)
        else:
            # Se nessuna modalità è attiva logga solo la lista
            run.log({"hydra/num_conf_files": len(data)}, commit=True)
            ok("Loggata lista file di configurazione in W&B.")

    def _fallback_text(self, run, data: Dict[str, str]):
        # Log dei contenuti come testo (potrebbe generare molte chiavi)
        # Per evitare step fantasma usiamo un singolo dict + commit=True
        text_payload = {f"conf_text/{k}": v for k, v in data.items()}
        # Riduci dimensione se molto grande (evita saturare UI)
        MAX_LEN = 4000
        for k, v in list(text_payload.items()):
            if len(v) > MAX_LEN:
                text_payload[k] = v[:MAX_LEN] + "\n... [TRUNCATED]"
        run.log(text_payload, commit=True)
        ok(f"Loggati contenuti YAML (fallback) su W&B ({len(data)} files).")

class DatasetEpochSetter(pl.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(trainer.current_epoch)

class ModelInfoLogger(pl.Callback):
    """Log model info and structure at the beginning of training.
    - prints summary to console
    - saves a JSON with parameter counts and module list
    - logs (optionally) to Weights & Biases
    """
    def __init__(self, filename: str = "model_info.json", max_module_lines: int = 512, log_structure: bool = True ):
        super().__init__()
        self.filename = filename
        self.max_module_lines = int(max_module_lines)
        self.log_structure = bool(log_structure)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # Extract info from the core model (autoencoder inside wrapper)
        model = getattr(pl_module, "autoencoder", pl_module)
        info = extract_model_config(model)

        # Limit how many module lines to display
        modules = info.get("modules", [])[: self.max_module_lines]

        # Console output: parameter summary + structure
        console.rule("[bold cyan]Model info")
        console.print(f"params total/trainable: {info.get('num_parameters_total')}/{info.get('num_parameters_trainable')}")
        console.print(f"model size (bytes): {info.get('model_bytes')}")
        console.rule("[bold cyan]Model structure")
        model_cfg = extract_model_config(model)
        if self.log_structure:
            console.print("[MODEL SUMMARY]\n", model_cfg["repr"])
        console.rule()

        # Save JSON to disk
        try:
            # prova a usare la cartella di logging; fallback alla root di lavoro
            base_dir = Path(getattr(trainer.logger, "save_dir", "") or trainer.default_root_dir or ".")
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / self.filename
            out_path.write_text(json.dumps(info, indent=2))
            ok(f"ModelInfoLogger: saved model info JSON to {str(out_path)}")
        except Exception as e:
            warn(f"ModelInfoLogger: not able to save JSON ({type(e).__name__}: {e})")
            out_path = None

        # logga su W&B (se presente)
        if isinstance(trainer.logger, WandbLogger):
            try:
                run = trainer.logger.experiment
                # Update run config with full model info
                run.config.update({"model_info": info}, allow_val_change=True)
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
                warn(f"ModelInfoLogger: W&B log skipped ({type(e).__name__}: {e})")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entrypoint using native instantiate API.
    
    All datasets, models, and components are instantiated via hydra.utils.instantiate
    with _target_ configuration format.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Reproducibility setup
    # ─────────────────────────────────────────────────────────────────────────
    seed = int(cfg.trainer.seed)
    deterministic_flag = bool(cfg.trainer.trainer.get("deterministic", False))
    strict_deterministic_flag = bool(cfg.trainer.trainer.get("strict_deterministic", False))
    
    configure_reproducibility(
        seed,
        deterministic=deterministic_flag,
        strict_deterministic=strict_deterministic_flag,
        warn=warn,
    )
    seed_everything(seed, workers=True)
    ok(f"Reproducibility configured with seed={seed}")

    # ─────────────────────────────────────────────────────────────────────────
    # Directory setup
    # ─────────────────────────────────────────────────────────────────────────
    runs_dir = Path(get_original_cwd()) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(get_original_cwd()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = runs_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset instantiation via Hydra
    # ─────────────────────────────────────────────────────────────────────────
    train_ds = instantiate(cfg.data.train_dataset, seed=seed)
    ok(f"Train dataset instantiated: {len(train_ds)} samples")

    eval_ds = None
    if cfg.data.get("eval_dataset") is not None:
        eval_ds = instantiate(cfg.data.eval_dataset, seed=seed)
        ok(f"Eval dataset instantiated: {len(eval_ds)} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # DataLoader construction
    # ─────────────────────────────────────────────────────────────────────────
    dl_cfg = OmegaConf.to_container(cfg.data.train_dataloader, resolve=True)
    num_workers = int(dl_cfg.get("num_workers", 8))
    
    train_dl = DataLoader(
        train_ds,
        batch_size=int(dl_cfg.get("batch_size", 8)),
        num_workers=num_workers,
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        drop_last=bool(dl_cfg.get("drop_last", True)),
        persistent_workers=(dl_cfg.get("persistent_workers", False) if num_workers > 0 else False),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
        collate_fn=collate_stft,
    )

    eval_dl = None
    if eval_ds is not None:
        dl_eval_cfg = OmegaConf.to_container(cfg.data.eval_dataloader, resolve=True)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=int(dl_eval_cfg.get("batch_size", dl_cfg.get("batch_size", 8))),
            num_workers=num_workers,
            pin_memory=bool(dl_eval_cfg.get("pin_memory", False)),
            shuffle=bool(dl_eval_cfg.get("shuffle", False)),
            drop_last=bool(dl_eval_cfg.get("drop_last", False)),
            persistent_workers=(dl_eval_cfg.get("persistent_workers", False) if num_workers > 0 else False),
            prefetch_factor=int(dl_eval_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
            collate_fn=collate_stft,
        )
    ok("DataLoaders created")

    # ─────────────────────────────────────────────────────────────────────────
    # Model instantiation via Hydra
    # ─────────────────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model.model, resolve=True)
    autoencoder = AutoEncoder.from_config(model_cfg)
    ok("AutoEncoder instantiated via Hydra")

    # ─────────────────────────────────────────────────────────────────────────
    # Channel configuration (from data config - must be set manually)
    # ─────────────────────────────────────────────────────────────────────────
    audio_channels = int(cfg.data.audio_channels)
    model_channels = int(cfg.data.model_channels)
    sample_rate = int(cfg.data.train_dataset.sample_rate)
    
    # Build STFT params dict for the engine
    stft_params = {
        "sample_rate": sample_rate,
        "n_fft": int(cfg.data.train_dataset.n_fft),
        "hop_length": int(cfg.data.train_dataset.hop_length),
        "win_length": int(cfg.data.train_dataset.win_length),
        "center": bool(cfg.data.train_dataset.center),
        "normalized": bool(cfg.data.train_dataset.normalized),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Training wrapper instantiation
    # ─────────────────────────────────────────────────────────────────────────
    trainer_cfg = cfg.trainer
    pre_transform_spec = model_cfg.get("autoencoder", {}).get("pre_transform")
    
    wrapper = AutoencoderTrainingWrapper(
        autoencoder=autoencoder,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        model_channels=model_channels,
        loss_config=OmegaConf.to_container(trainer_cfg.get("loss_config", {}), resolve=True) or None,
        eval_loss_config=OmegaConf.to_container(trainer_cfg.get("eval_loss_config", {}), resolve=True) or None,
        optimizer_configs=None,
        warmup_steps=int(trainer_cfg.trainer.get("warmup_steps", 0)),
        warmup_mode=str(trainer_cfg.trainer.get("warmup_mode", "adv")),
        encoder_freeze_on_warmup=bool(trainer_cfg.trainer.get("encoder_freeze_on_warmup", False)),
        force_input_mono=bool(model_cfg.get("autoencoder", {}).get("force_input_mono", False)),
        latent_mask_ratio=float(model_cfg.get("autoencoder", {}).get("latent_mask_ratio", 0.0)),
        teacher_model=None,
        stft_params=stft_params,
        optimizer_spec=OmegaConf.to_container(trainer_cfg.get("optimizer", {}), resolve=True) or None,
        scheduler_spec=OmegaConf.to_container(trainer_cfg.get("scheduler", {}), resolve=True) or None,
        pre_transform_spec=pre_transform_spec,
    )
    ok("Training wrapper created")

    # Provide eval STFT params to engine for validation phase
    if eval_ds is not None:
        eval_stft_params = {
            "sample_rate": int(cfg.data.eval_dataset.sample_rate),
            "n_fft": int(cfg.data.eval_dataset.n_fft),
            "hop_length": int(cfg.data.eval_dataset.hop_length),
            "win_length": int(cfg.data.eval_dataset.win_length),
            "center": bool(cfg.data.eval_dataset.center),
            "normalized": bool(cfg.data.eval_dataset.normalized),
        }
        wrapper.engine.val_stft_params = eval_stft_params

    # ─────────────────────────────────────────────────────────────────────────
    # Logger setup (W&B or TensorBoard)
    # ─────────────────────────────────────────────────────────────────────────
    wandb_cfg = trainer_cfg.get("wandb", {}) or {}
    use_wandb = bool(wandb_cfg.get("use_wandb", False))
    
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=wandb_cfg.get("project", "ICML_2026"),
            name=wandb_cfg.get("name", "default_name"),
            save_dir=str(runs_dir),
            log_model=wandb_cfg.get("log_model", "all"),
            settings=wandb.Settings(_service_wait=7),
        )
        try:
            run = logger.experiment
            conf_root = Path(get_original_cwd()) / "conf"
            WandbConfigLogger(conf_root, use_artifact=True, log_text=False).log_to_wandb(run)
        except Exception as e:
            warn(f"Upload dir conf on W&B failed ({type(e).__name__}: {e})")
    else:
        logger = TensorBoardLogger(save_dir=str(runs_dir), name="lightning_logs", version=None)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks setup
    # ─────────────────────────────────────────────────────────────────────────
    pl_trainer_cfg = trainer_cfg.trainer
    
    callbacks = [
        ModelInfoLogger(
            filename="model_info.json",
            max_module_lines=768,
            log_structure=bool(pl_trainer_cfg.get("log_model_structure", False))
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch_{epoch:03d}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=int(pl_trainer_cfg.get("save_every_n_epochs", 3)),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
        TQDMProgressBar(refresh_rate=1),
        DatasetEpochSetter(train_ds),
    ]

    # Validation demo callback
    demo_cfg = OmegaConf.to_container(cfg.data.get("demo", {}), resolve=True) or {}
    if eval_dl is not None:
        callbacks.append(
            AutoencoderValDemoCallback(
                every_n_epochs=int(demo_cfg.get("every_n_epochs", 1)),
                max_demos=int(demo_cfg.get("max_demos", 8)),
                sample_rate=sample_rate,
                istft_params=demo_cfg.get("istft_params", {}),
                target_seconds=float(demo_cfg.get("target_seconds", 1.0)),
                save_basename=str(demo_cfg.get("save_basename", "recon_val")),
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Profiler (optional)
    # ─────────────────────────────────────────────────────────────────────────
    use_profiler = bool(pl_trainer_cfg.get("profile", False))
    profiler = None
    if use_profiler:
        profiler = PyTorchProfiler(
            dirpath=str(profiler_dir),
            filename="pl_profile",
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            schedule=torch_profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=torch_profiler.tensorboard_trace_handler(str(profiler_dir)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            profile_dataloader=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Precision warning for complex models
    # ─────────────────────────────────────────────────────────────────────────
    requested_precision = str(pl_trainer_cfg.get("precision", "32-true")).lower()
    is_bf16 = ("bf16" in requested_precision)
    try:
        has_complex_params = any(p.is_complex() for p in wrapper.autoencoder.parameters())
    except Exception:
        has_complex_params = False
    if is_bf16 and has_complex_params:
        warn("bf16 + complex detected: convolutions will use torch.complex64 (complex-bfloat16 not supported).")

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch Lightning Trainer
    # ─────────────────────────────────────────────────────────────────────────
    trainer = Trainer(
        default_root_dir=str(runs_dir),
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=int(pl_trainer_cfg.get("num_gpus", 1)),
        strategy=pl_trainer_cfg.get("strategy", "auto"),
        max_epochs=int(pl_trainer_cfg.get("epochs", 50)),
        precision="32-true",
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=int(pl_trainer_cfg.get("log_interval", 1)),
        num_sanity_val_steps=int(pl_trainer_cfg.get("num_sanity_val_steps", 0)),
        gradient_clip_val=float(pl_trainer_cfg.get("gradient_clip_val", 0.0)),
        detect_anomaly=bool(pl_trainer_cfg.get("detect_anomaly", False)),
        profiler=profiler,
        check_val_every_n_epoch=int(pl_trainer_cfg.get("check_val_every_n_epoch", 1)),
        val_check_interval=pl_trainer_cfg.get("val_check_interval", None),
        deterministic=deterministic_flag,
    )

    ok("Starting training...")
    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl)


if __name__ == "__main__":
    main()