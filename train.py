# ==============================================
# Train Entry Point
# ==============================================

from pathlib import Path
import sys
import os
from typing import Dict, List
import torch.profiler as torch_profiler
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader
from datetime import datetime
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
from rich.console import Console

console = Console()

from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.training.autoencoders import AutoencoderTrainingWrapper, AutoencoderValDemoCallback
from ar_spectra.training.initialization import collate_stft
from ar_spectra.utils.reproducibility import configure_reproducibility
from ar_spectra.utils.run_config import _is_rank0, get_checkpoint_dir, resolve_run_name
from ar_spectra.utils.console import ok, warn, err

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import config
OmegaConf.register_new_resolver("config", lambda key: getattr(config, key))
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))  # e.g. ${mul:${model.parameters_to_predict},${model.latent_channels}}

from ar_spectra.training.callbacks import DatasetEpochSetter, ModelInfoLogger
from ar_spectra.utils.model_info import log_compression_stats

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
                warn(f"Skip file {f} ({type(e).__name__}: {e})", prefix="TRAINER")
        return out

    def log_to_wandb(self, run):
        if not _is_rank0():
            return
        data = self.load_contents()
        if not data:
            warn("Nessun file di configurazione trovato da loggare su W&B.", prefix="TRAINER")
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
                ok(f"Caricata cartella conf come artifact W&B ({len(data)} files).", prefix="TRAINER")
            except Exception as e:
                warn(f"Artifact upload fallito ({type(e).__name__}: {e}); provo fallback testuale.", prefix="TRAINER")
                self._fallback_text(run, data)
        elif self.log_text:
            self._fallback_text(run, data)
        else:
            # Se nessuna modalità è attiva logga solo la lista
            run.log({"hydra/num_conf_files": len(data)}, commit=True)
            ok("Loggata lista file di configurazione in W&B.", prefix="TRAINER")

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
        ok(f"Loggati contenuti YAML (fallback) su W&B ({len(data)} files).", prefix="TRAINER")

class TableOnlyModelSummary(pl.Callback):
    """Custom model summary that only prints the parameters table, discarding verbose stats."""
    def __init__(self, max_depth: int = 2):
        super().__init__()
        self.max_depth = max_depth

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        model_summary = summarize(pl_module, max_depth=self.max_depth)
        summary_str = str(model_summary)
        parts = summary_str.split("--------------------------------------------------------------------------")
        if len(parts) >= 3:
            table_str = "--------------------------------------------------------------------------".join(parts[:2]) + "\n--------------------------------------------------------------------------"
            console.print(table_str)
        else:
            console.print(summary_str)



@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    """Hydra entrypoint using native instantiate API.
    
    All datasets, models, and components are instantiated via hydra.utils.instantiate
    with _target_ configuration format.
    """

    console.rule("[bold cyan]Training[/bold cyan]")
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
    ok(f"Reproducibility configured with seed={seed}", prefix="SEED")

    # ─────────────────────────────────────────────────────────────────────────
    # Directory setup
    # ─────────────────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1) Get the run name (user provided via wandb.name or resolved)
    user_run_name = cfg.trainer.get("wandb", {}).get("name") if cfg.get("trainer") else None
    if not user_run_name or user_run_name == "FMA_autoencoder_KL":
        run_name = resolve_run_name(cfg)
    else:
        run_name = user_run_name
        
    ok(f"Resolved run name: {run_name}", prefix="MODEL")
    
    # 2) Create unique, nested run directory
    run_dir = Path(get_original_cwd()) / "runs" / run_name / now
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    profiler_dir = run_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset instantiation via Hydra
    # ─────────────────────────────────────────────────────────────────────────
    train_ds = instantiate(cfg.data.train_dataset, seed=seed)
    ok(f"Train dataset instantiated: {len(train_ds)} samples", prefix="DATA")

    eval_ds = None
    if cfg.data.get("eval_dataset") is not None:
        eval_ds = instantiate(cfg.data.eval_dataset, seed=seed)
        ok(f"Eval dataset instantiated: {len(eval_ds)} samples", prefix="DATA")

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
        timeout=config.DEFAULT_DATALOADER_TIMEOUT,  # 1 min timeout per batch to detect stuck workers
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

    # ─────────────────────────────────────────────────────────────────────────
    # Model instantiation via Hydra
    # ─────────────────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.models.model, resolve=True)
    autoencoder = AutoEncoder.from_config(model_cfg)

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
    
    loss_config_dict = OmegaConf.to_container(trainer_cfg.get("loss_config", {}), resolve=True) or {}
    kl_beta_target = float((loss_config_dict.get("bottleneck") or {}).get("weights", {}).get("kl", 0.0))

    wrapper = AutoencoderTrainingWrapper(
        autoencoder=autoencoder,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        model_channels=model_channels,
        loss_config=loss_config_dict or None,
        eval_loss_config=OmegaConf.to_container(trainer_cfg.get("eval_loss_config", {}), resolve=True) or None,
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
        accumulate_grad_batches=int(trainer_cfg.trainer.get("accumulate_grad_batches", 1)),
        clip_grad_norm=float(trainer_cfg.trainer.get("clip_grad_norm", 0.0)),
        kl_annealing_epochs=int(trainer_cfg.trainer.get("kl_annealing_epochs", 0)),
        kl_beta_target=kl_beta_target,
    )

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
        if _is_rank0():
            logger = WandbLogger(
                project=wandb_cfg.get("project", config.DEFAULT_WANDB_PROJECT),
                name=run_name,
                save_dir=str(run_dir),
                log_model=wandb_cfg.get("log_model", "all"),
                config=OmegaConf.to_container(cfg, resolve=True),
                settings=wandb.Settings(_service_wait=7),
            )
            try:
                run = logger.experiment
                conf_root = Path(get_original_cwd()) / "config"
                WandbConfigLogger(conf_root, use_artifact=True, log_text=False).log_to_wandb(run)
            except Exception as e:
                warn(f"Upload dir conf on W&B failed ({type(e).__name__}: {e})", prefix="TRAINER")
        else:
            # Evita l'inizializzazione di run W&B sugli altri rank, ma mantieni un logger compatibile
            logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs", version=None)
    else:
        logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs", version=None)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks setup
    # ─────────────────────────────────────────────────────────────────────────
    pl_trainer_cfg = trainer_cfg.trainer
    
    callbacks = [
        ModelInfoLogger(
            filename=str(run_dir / "model_info.json"),
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
        TableOnlyModelSummary(max_depth=2),
        TQDMProgressBar(refresh_rate=1),
        DatasetEpochSetter(),
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
        warn("bf16 + complex detected: convolutions will use torch.complex64 (complex-bfloat16 not supported).", prefix="TRAINER")

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch Lightning Trainer
    # ─────────────────────────────────────────────────────────────────────────
    req_device = str(trainer_cfg.get("device", "auto")).lower()
    req_accelerator = req_device.split(":")[0]  # strip index, e.g. "cuda:0" -> "cuda"

    trainer = Trainer(
        default_root_dir=str(run_dir),
        accelerator=req_accelerator,
        devices=int(pl_trainer_cfg.get("num_gpus", 1)),
        strategy=pl_trainer_cfg.get("strategy", "auto"),
        max_epochs=int(pl_trainer_cfg.get("epochs", 50)),
        precision=requested_precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(pl_trainer_cfg.get("log_interval", 1)),
        num_sanity_val_steps=int(pl_trainer_cfg.get("num_sanity_val_steps", 0)),
        gradient_clip_val=float(pl_trainer_cfg.get("gradient_clip_val", 0.0)),
        detect_anomaly=bool(pl_trainer_cfg.get("detect_anomaly", False)),
        profiler=profiler,
        check_val_every_n_epoch=int(pl_trainer_cfg.get("check_val_every_n_epoch", 1)),
        val_check_interval=pl_trainer_cfg.get("val_check_interval", None),
        limit_train_batches=pl_trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=pl_trainer_cfg.get("limit_val_batches", 1.0),
        deterministic=deterministic_flag,
        enable_model_summary=False,
    )

    ok(f"{req_accelerator}", prefix="DEVICE")
    
    # Extracts and visualizes latent shape and compression rate
    if _is_rank0():
        log_compression_stats(wrapper, train_dl, console)

    ckpt_path = OmegaConf.select(cfg, "ckpt_path", default=None)
    if ckpt_path:
        ok(f"Resuming from checkpoint: {ckpt_path}", prefix="TRAINER")

    try:
        trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        warn("Training interrupted by user (Ctrl+C). Exiting gracefully...", prefix="TRAINER")
        sys._exit(0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
            err("="*80, prefix="TRAINER")
            err("🚨 OUT OF MEMORY ERROR DETECTED 🚨", prefix="TRAINER")
            err("="*80, prefix="TRAINER")
            err(f"Error details: {e}", prefix="TRAINER")
            err("To fix this, you can:", prefix="TRAINER")
            err("1. Decrease batch size (e.g., `data.train_dataloader.batch_size=8` instead of 32)", prefix="TRAINER")
            err("2. Decrease model size (e.g., lower `block_out_channels` in `conf/model/hf_autoencoder_kl.yaml`)", prefix="TRAINER")
            err("3. Use a smaller dataset or shorter audio segments", prefix="TRAINER")
            err("4. Disable profilers or decrease `accumulate_grad_batches`", prefix="TRAINER")
            err("="*80 + "\n", prefix="TRAINER")
        elif "is killed by signal: interrupt" in str(e).lower():
            warn("Training interrupted by user (Ctrl+C). Force exiting.", prefix="TRAINER")
            sys._exit(0)
        else:
            raise e

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        warn("Script terminated by user (Ctrl+C). Exiting gracefully.", prefix="TRAINER")
        sys.exit(0)
