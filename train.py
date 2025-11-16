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
from tqdm import tqdm
from rich.console import Console
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ar_spectra.training_utils.get_model_config import extract_model_config
import wandb
import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
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

    torch.backends.cudnn.benchmark = True

    # Dataset
    train_ds = instantiate_from_spec(cfg["train_dataset"])
    dl_cfg = cfg.get("train_dataloader", {}) or {}
    num_workers = int(dl_cfg.get("num_workers", 8))

    # Decide pin_memory for train
    train_batch_size = int(dl_cfg.get("batch_size", 8))
    train_pin_req = dl_cfg.get("pin_memory", "auto")
    _train_pf = int(dl_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None
    train_pin_memory = _decide_pin_memory(
        train_pin_req, train_ds, train_batch_size, num_workers=num_workers, prefetch_factor=_train_pf
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=train_pin_memory,
        shuffle=bool(dl_cfg.get("shuffle", True)),
        drop_last=True,
        persistent_workers=(dl_cfg.get("persistent_workers", False) if num_workers > 0 else False),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
        collate_fn=collate_stft,
    )
    
    eval_ds = instantiate_from_spec(cfg.get("eval_dataset", None))
    dl_eval_cfg = cfg.get("eval_dataloader", {}) or {}
    if eval_ds is not None:
        eval_batch_size = int(dl_eval_cfg.get("batch_size", train_batch_size))
        eval_pin_req = dl_eval_cfg.get("pin_memory", train_pin_req)
        _eval_pf = int(dl_eval_cfg.get("prefetch_factor", dl_cfg.get("prefetch_factor", 8))) if num_workers > 0 else None
        eval_pin_memory = _decide_pin_memory(
            eval_pin_req, eval_ds, eval_batch_size, num_workers=num_workers, prefetch_factor=_eval_pf
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=eval_pin_memory,
            shuffle=bool(dl_eval_cfg.get("shuffle", False)),
            drop_last=False,
            persistent_workers=(dl_eval_cfg.get("persistent_workers", False) if num_workers > 0 else False),
            prefetch_factor=int(dl_eval_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
            collate_fn=collate_stft,
        )
    else:
        eval_dl = None
        warn("Eval dataset and dataloader not provided; training will proceed without evaluation.")

    # Infer channel counts from dataset or first batch (for 'auto' placeholders)
    ds_spec_ch = getattr(train_ds, "spec_channels", None)
    ds_audio_ch = getattr(train_ds, "audio_channels", None)
    cac = bool(cfg.get("train_dataset", {}).get("kwargs", {}).get("cac", False))
    if ds_spec_ch is not None and ds_audio_ch is not None:
        model_channels = int(ds_spec_ch)
        audio_channels = int(ds_audio_ch)
    else:
        sample = next(iter(train_dl))
        sp_reals, orig = sample
        if sp_reals.dim() == 3:
            Cx, F, T = sp_reals.shape
        elif sp_reals.dim() == 4:
            _, Cx, F, T = sp_reals.shape
        else:
            raise RuntimeError(f"Forma inattesa per sp_reals: {tuple(sp_reals.shape)}")
        model_channels = int(Cx)
        audio_channels = (model_channels // 2) if cac else model_channels

    # Resolve 'auto' placeholders in model encoder/decoder configuration
    model_cfg = cfg["model"]
    enc_kwargs = model_cfg["encoder"].setdefault("kwargs", {})
    dec_kwargs = model_cfg["decoder"].setdefault("kwargs", {})
    def set_auto(d: dict, key: str, value: int):
        v = d.get(key, None)
        if (v is None) or (isinstance(v, str) and v.lower() == "auto"):
            d[key] = int(value)
    set_auto(enc_kwargs, "input_size", model_channels)
    set_auto(dec_kwargs, "channels", model_channels)
    if "out_channels" in dec_kwargs:
        set_auto(dec_kwargs, "out_channels", model_channels)

    # Build model and Lightning wrapper
    autoenc = AutoEncoder.from_config(model_cfg)
    # Optimizer/scheduler specs are passed through to the wrapper
    optimizer_spec = cfg.get("optimizer", None)
    scheduler_spec = cfg.get("scheduler", None)

    # Instantiate Lightning training wrapper
    wrapper = AutoencoderTrainingWrapper(
        autoencoder=autoenc,
        sample_rate=int(cfg["train_dataset"]["kwargs"].get("sample_rate", 44100)),
        audio_channels=int(audio_channels),
        loss_config=cfg.get("loss_config", None),
        eval_loss_config=cfg.get("eval_loss_config", None),
        optimizer_configs=None,
        warmup_steps=int(cfg.get("trainer", {}).get("warmup_steps", 0)),
        warmup_mode=str(cfg.get("trainer", {}).get("warmup_mode", "adv")),
        encoder_freeze_on_warmup=bool(cfg.get("trainer", {}).get("encoder_freeze_on_warmup", False)),
        force_input_mono=bool(cfg.get("model", {}).get("autoencoder", {}).get("force_input_mono", False)),
        latent_mask_ratio=float(cfg.get("model", {}).get("autoencoder", {}).get("latent_mask_ratio", 0.0)),
        teacher_model=None,
        stft_params=cfg.get("train_dataset", {}).get("kwargs", {}),
        optimizer_spec=optimizer_spec,
        scheduler_spec=scheduler_spec,
        pre_transform_spec=cfg.get("pre_transform", None),
    )
    ok(f"Instantiated AutoEncoder and Lightning wrapper.")

    # Provide eval STFT params to engine for validation phase
    eval_stft_params = (cfg.get("eval_dataset", {}) or {}).get("kwargs", {}) or {}
    wrapper.engine.val_stft_params = eval_stft_params

    # W&B logger 
    wandb_cfg = (cfg.get("wandb", {}) or {})
    use_wandb = bool(wandb_cfg.get("use_wandb", False))
    # Logger (W&B o TensorBoard) 
    logger = None
    if use_wandb:

        logger = WandbLogger(project=wandb_cfg.get("project", "ICML_2026"), name=wandb_cfg.get("name", "default_name"), 
                             save_dir=str(runs_dir), log_model=False, settings=wandb.Settings(_service_wait=7))
        # Log YAML configuration files used for the run (Hydra mode) or legacy JSON.
        try:
            run = logger.experiment
            # Always attach parsed unified configuration
            run.config.update({"parsed_config": cfg}, allow_val_change=True)

            original_cwd = Path(get_original_cwd())
            if is_json_mode:
                # Legacy JSON mode: save the original JSON and the parsed unified config
                legacy_path_obj = Path(legacy_path) if 'legacy_path' in locals() and legacy_path else None
                # Opzione per salvare localmente anche nella cartella media (solo se richiesto)
                store_local = bool(wandb_cfg.get("store_local_configs", False))
                configs_local_dir = media_dir / f"configs_{run.id}" if store_local else None
                if store_local:
                    configs_local_dir.mkdir(parents=True, exist_ok=True)
                # Parsed unified config JSON
                parsed_cfg_path = (configs_local_dir / "parsed_config.json") if store_local else (original_cwd / "conf" / "parsed_config.json")
                parsed_cfg_path.write_text(json.dumps(cfg, indent=2))
                run.save(str(parsed_cfg_path), base_path=str(parsed_cfg_path.parent))
                # Original legacy JSON
                if legacy_path_obj is not None and legacy_path_obj.exists():
                    legacy_copy_path = (configs_local_dir / "legacy_experiment.json") if store_local else legacy_path_obj
                    if store_local:
                        legacy_copy_path.write_text(legacy_path_obj.read_text())
                    run.save(str(legacy_copy_path), base_path=str(legacy_copy_path.parent))
                ok("W&B: uploaded legacy JSON configuration files (no duplicate outside media dir).")
            else:
                # Hydra mode: save top-level config plus data/model/trainer YAMLs
                hydra_root = original_cwd / "conf"
                yaml_targets = []
                # Main composed config (config.yaml)
                yaml_targets.append(hydra_root / "config.yaml")
                # Data
                yaml_targets.append(hydra_root / "data" / "data.yaml")
                # Trainer
                yaml_targets.append(hydra_root / "trainer" / "trainer.yaml")
                # All model yaml variants (upload all to capture selected + alternatives)
                model_dir = hydra_root / "model"
                if model_dir.exists():
                    for mf in sorted(model_dir.glob("*.yaml")):
                        yaml_targets.append(mf)
                # Decide if we create local copies under media/ or only register originals
                store_local = bool(wandb_cfg.get("store_local_configs", False))
                configs_local_dir = media_dir / f"configs_{run.id}" if store_local else None
                if store_local:
                    configs_local_dir.mkdir(parents=True, exist_ok=True)
                uploaded = 0
                for src in yaml_targets:
                    if not src.exists():
                        continue
                    if store_local:
                        # Copy into media/<run-id>/configs
                        dst = configs_local_dir / src.name
                        dst.write_text(src.read_text())
                        run.save(str(dst), base_path=str(dst.parent))
                    else:
                        # Register original file directly without duplicating in CWD
                        run.save(str(src), base_path=str(src.parent))
                    uploaded += 1
                mode_msg = "(stored locally under media/)" if store_local else "(only on W&B; originals referenced)"
                ok(f"W&B: uploaded {uploaded} Hydra YAML file(s) {mode_msg}.")
        except Exception as e:
            warn(f"W&B YAML/JSON config upload failed ({type(e).__name__}: {e})")
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
        has_complex_params = any(p.is_complex() for p in autoenc.parameters())
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
    )

    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl if eval_dl is not None else None)

if __name__ == "__main__":
    main()