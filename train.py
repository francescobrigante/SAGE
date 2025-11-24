import json
from pathlib import Path
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch.profiler as torch_profiler
from pytorch_lightning.profilers import PyTorchProfiler
from ar_spectra.models.autoencoder import AutoEncoder, instantiate_from_spec
from ar_spectra.training_utils.autoencoders import AutoencoderTrainingWrapper, AutoencoderValDemoCallback
from ar_spectra.training_utils.initialization import (
    build_datasets_and_loaders,
    build_training_wrapper_from_cfg,
)
from ar_spectra.training_utils.reproducibility import configure_reproducibility
from tqdm import tqdm
from rich.console import Console
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ar_spectra.training_utils.get_model_config import extract_model_config
import wandb
import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, List
console = Console()  

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")

def collate_stft(batch):
    Ss, wavs = zip(*batch)
    s0 = Ss[0].shape
    w0 = wavs[0].shape
    assert all(x.shape == s0 for x in Ss), f"STFT shapes differ: {[x.shape for x in Ss]}"
    assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
    return torch.stack(Ss, 0), torch.stack(wavs, 0)

# Auto pin-memory helpers 
def _max_pinnable_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    mb_list = [8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256, 384, 512]
    last_ok = 0
    for mb in mb_list:
        try:
            x = torch.empty((mb * 1024 * 1024) // 4, dtype=torch.float32)
            x.pin_memory()
            last_ok = mb
        except Exception:
            break
    return last_ok

def _sample_size_bytes(dataset) -> int:
    # Estimate item size by reading a single sample (S, W)
    try:
        s0, w0 = dataset[0]
        return s0.numel() * s0.element_size() + w0.numel() * w0.element_size()
    except Exception as e:
        warn(f"Could not estimate item size from dataset[0] ({type(e).__name__}: {e}); falling back to 0.")
        return 0

def _parse_pin_flag(value):
    # Accept bools or strings: "auto"|"true"|"false"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
        if v == "auto":
            return "auto"
    # default to auto if unspecified
    return "auto"

def _decide_pin_memory(requested, dataset, batch_size, num_workers: int = 0, prefetch_factor: int = None) -> bool:
    req = _parse_pin_flag(requested)
    if req is True or req is False:
        info(f"pin_memory set from config: {req}")
        return bool(req)
    # auto mode
    if not torch.cuda.is_available():
        info("CUDA not available; pin_memory disabled.")
        return False

    pinnable_mb = _max_pinnable_mb()
    item_bytes = _sample_size_bytes(dataset)
    if item_bytes <= 0:
        warn("Could not estimate item size; enabling pin_memory conservatively.")
        return True

    batch_mb = (item_bytes * batch_size) / (1024 * 1024)
    if num_workers > 0:
        pf = prefetch_factor if (prefetch_factor is not None) else 2  # default PyTorch
        pinned_batches = (num_workers * pf) + 2
    else:
        pf = None
        pinned_batches = 1

    effective_mb = batch_mb * pinned_batches
    use_pin = effective_mb <= pinnable_mb
    msg = ("enabled" if use_pin else "disabled")
    if pf is None:
        info(f"Auto pin_memory {msg}: batch≈{batch_mb:.2f} MB, pipelined_batches≈{pinned_batches}, "
             f"effective≈{effective_mb:.2f} MB, pinnable≈{pinnable_mb} MB")
    else:
        info(f"Auto pin_memory {msg}: batch≈{batch_mb:.2f} MB, pipelined_batches≈{pinned_batches} "
             f"(workers={num_workers}, prefetch={pf}), effective≈{effective_mb:.2f} MB, "
             f"pinnable≈{pinnable_mb} MB")
    return use_pin
# --- End auto pin-memory helpers ---

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

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
    """Hydra entrypoint. Falls back to a monolithic JSON when trainer.use_json=true."""
    # Determine configuration mode (Hydra vs legacy JSON)
    is_json_mode = cfg.trainer.get("use_json", False)
    if is_json_mode:
        legacy_path = cfg.trainer.get("json_path")
        if not legacy_path:
            raise ValueError("trainer.use_json=true but trainer.json_path is empty")
        legacy_cfg = load_json(legacy_path)
        ok(f"Loaded legacy JSON: {legacy_path}")
        unified = legacy_cfg
    else:
        # Rebuild unified experiment dictionary
        unified = {
            "seed": int(cfg.trainer.seed),
            "device": cfg.device if hasattr(cfg, "device") else cfg.trainer.get("device", "cuda"),
            "train_dataset": OmegaConf.to_container(cfg.data.train_dataset, resolve=True),
            "train_dataloader": OmegaConf.to_container(cfg.data.train_dataloader, resolve=True),
            "eval_dataset": OmegaConf.to_container(cfg.data.get("eval_dataset", {}), resolve=True),
            "eval_dataloader": OmegaConf.to_container(cfg.data.get("eval_dataloader", {}), resolve=True),
            "demo": OmegaConf.to_container(cfg.data.get("demo", {}), resolve=True),
            "model": OmegaConf.to_container(cfg.model.model, resolve=True),
            "optimizer": OmegaConf.to_container(cfg.trainer.get("optimizer", {}), resolve=True),
            "scheduler": OmegaConf.to_container(cfg.trainer.get("scheduler", {}), resolve=True),
            "trainer": OmegaConf.to_container(cfg.trainer.trainer, resolve=True),
            "wandb": OmegaConf.to_container(cfg.trainer.get("wandb", {}), resolve=True),
            "eval_loss_config": OmegaConf.to_container(cfg.trainer.get("eval_loss_config", {}), resolve=True),
            "loss_config": OmegaConf.to_container(cfg.trainer.get("loss_config", {}), resolve=True),
            "pre_transform": OmegaConf.to_container(cfg.trainer.get("pre_transform", {}), resolve=True),
        }
        ok("Hydra composition complete")

    cfg = unified
    seed = int(cfg.get("seed", 42))
    deterministic_flag = bool(cfg.get("trainer", {}).get("deterministic", True))
    configure_reproducibility(seed, deterministic=deterministic_flag, warn=warn)
    seed_everything(seed, workers=True)

    # Base directory centralized
    runs_dir = Path(get_original_cwd()) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Standard subdirectories
    # Checkpoints fuori da runs (richiesta: cartella root 'checkpoints')
    ckpt_dir = Path(get_original_cwd()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = runs_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    media_dir = runs_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Datasets, dataloaders and wrapper initialization (single source of truth)
    wrapper, data_init = build_training_wrapper_from_cfg(cfg)
    train_ds = data_init.train_dataset
    train_dl = data_init.train_dataloader
    eval_dl = data_init.eval_dataloader
    audio_channels = data_init.audio_channels
    ok("Instantiated datasets, dataloaders, AutoEncoder and Lightning wrapper.")

    # Provide eval STFT params to engine for validation phase
    eval_stft_params = (cfg.get("eval_dataset", {}) or {}).get("kwargs", {}) or {}
    wrapper.engine.val_stft_params = eval_stft_params

    # W&B logger 
    wandb_cfg = (cfg.get("wandb", {}) or {})
    use_wandb = bool(wandb_cfg.get("use_wandb", False))
    # Logger (W&B o TensorBoard) 
    logger = None
    if use_wandb:
        # NB: sezione W&B omessa; assicurarsi che WandbLogger usi save_dir=runs_dir
        logger = WandbLogger(
            project=wandb_cfg.get("project", "ICML_2026"),
            name=wandb_cfg.get("name", "default_name"),
            save_dir=str(runs_dir),
            log_model=wandb_cfg.get("log_model", "all"),  # log checkpoints to W&B with same cadence as local saves
            settings=wandb.Settings(_service_wait=7),
        )
        # Upload immediato dell'intera cartella di configurazione Hydra su W&B (senza copie locali)
        try:
            run = logger.experiment
            conf_root = Path(get_original_cwd()) / "conf"
            WandbConfigLogger(conf_root, use_artifact=True, log_text=False).log_to_wandb(run)
        except Exception as e:
            warn(f"Upload dir conf on W&B failed ({type(e).__name__}: {e})")
    else:
        logger = TensorBoardLogger(save_dir=str(runs_dir), name="lightning_logs", version=None)
        try:
            run = logger.experiment
            # Update run config with parsed configuration
            run.config.update({"parsed_config": cfg}, allow_val_change=True)
            if is_json_mode:
                # Legacy JSON mode: save original experiment JSON + parsed unified config
                cfg_path = Path("wandb_parsed_config.json")
                cfg_path.write_text(json.dumps(cfg, indent=2))
                run.save(str(cfg_path), base_path=str(cfg_path.parent))
                legacy_path_obj = Path(legacy_path)
                if legacy_path_obj.exists():
                    # Copy original JSON into run directory
                    legacy_copy = Path(f"wandb_legacy_experiment.json")
                    legacy_copy.write_text(legacy_path_obj.read_text())
                    run.save(str(legacy_copy), base_path=str(legacy_copy.parent))
            else:
                # Hydra mode: save core YAML configuration files
                original_cwd = Path(get_original_cwd())
                hydra_files = [
                    original_cwd / "conf" / "config.yaml",
                    original_cwd / "conf" / "data" / "data.yaml",
                    original_cwd / "conf" / "model" / "model.yaml",
                    original_cwd / "conf" / "trainer" / "trainer.yaml",
                ]
                for f in hydra_files:
                    if f.exists():
                        # Copy YAML into run directory
                        dst = Path(f"wandb_{f.name}")
                        dst.write_text(f.read_text())
                        run.save(str(dst), base_path=str(dst.parent))
        except Exception as e:
            warn(f"W&B config upload skipped ({type(e).__name__}: {e})")

    # Callbacks
    callbacks = [
        ModelInfoLogger(
            filename="model_info.json",
            max_module_lines=768,
            log_structure=cfg.get("trainer", {}).get("log_model_structure", False)
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch_{epoch:03d}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=int(cfg.get("trainer", {}).get("save_every_n_epochs", 3)),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
        TQDMProgressBar(refresh_rate=1),
        DatasetEpochSetter(train_ds),
    ]

    # Optional validation demo callback
    demo_cfg = cfg.get("demo", {}) or {}
    if eval_dl is not None:
        # Passa direttamente i parametri del dataset di training
        callbacks.append(
            AutoencoderValDemoCallback(
                every_n_epochs=int(demo_cfg.get("every_n_epochs", 1)),
                max_demos=int(demo_cfg.get("max_demos", 8)),
                sample_rate=int(cfg["train_dataset"]["kwargs"].get("sample_rate", 44100)),
                istft_params=demo_cfg.get("istft_params", {}),
                target_seconds=float(demo_cfg.get("target_seconds", 1.0)),
                save_basename=str(demo_cfg.get("save_basename", "recon_val")),
            )
        )

    
    # Profiler
    use_profiler = bool(cfg.get("trainer", {}).get("profile", False))
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

    # Warning for  bf16 precision with complex model: we will force conv to complex64
    try:
        requested_precision = str(cfg.get("trainer", {}).get("trainer", {}).get("precision", "32-true")).lower()
    except Exception:
        requested_precision = "32-true"
    is_bf16 = ("bf16" in requested_precision)
    try:
        has_complex_params = any(p.is_complex() for p in wrapper.autoencoder.parameters())
    except Exception:
        has_complex_params = False
    if is_bf16 and has_complex_params:
        warn("bf16 + complex rilevato: la convoluzione userà torch.complex64 (complex-bfloat16 non supportato).")

    # Trainer (sets default_root_dir -> runs_dir)
    trainer = Trainer(
        default_root_dir=str(runs_dir),
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=int(cfg.get("trainer", {}).get("num_gpus", 1)),
        strategy=cfg.get("trainer", {}).get("strategy", "auto"),
        max_epochs=int(cfg.get("trainer", {}).get("epochs", 50)),
        precision="32-true",
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=int(cfg.get("trainer", {}).get("log_interval", 1)),
        num_sanity_val_steps=int(cfg.get("trainer", {}).get("num_sanity_val_steps", 0)),
        gradient_clip_val=0.0,
        detect_anomaly=False,
        profiler=profiler,
        check_val_every_n_epoch=int(cfg.get("trainer", {}).get("check_val_every_n_epoch", 1500)),
        val_check_interval=cfg.get("trainer", {}).get("val_check_interval", None),
        deterministic=deterministic_flag,
    )

    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl if eval_dl is not None else None)

if __name__ == "__main__":
    main()