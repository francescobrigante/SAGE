import json
from pathlib import Path
import torch
import hydra
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

# --- Auto pin-memory helpers ---
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
    """
    Logga info e struttura del modello all'inizio del training usando extract_model_config:
    - stampa a terminale
    - salva un JSON su disco
    - logga su W&B (se attivo)
    """
    def __init__(self, filename: str = "model_info.json", max_module_lines: int = 512, log_structure: bool = True ):
        super().__init__()
        self.filename = filename
        self.max_module_lines = int(max_module_lines)
        self.log_structure = bool(log_structure)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # estrae info dal modello core (autoencoder dentro il wrapper)
        model = getattr(pl_module, "autoencoder", pl_module)
        info = extract_model_config(model)

        # limita quante righe della struttura stampare
        modules = info.get("modules", [])[: self.max_module_lines]

        # stampa su console (info + struttura)
        console.rule("[bold cyan]Model info")
        console.print(f"params total/trainable: {info.get('num_parameters_total')}/{info.get('num_parameters_trainable')}")
        console.rule("[bold cyan]Model structure")
        model_cfg = extract_model_config(model)
        if self.log_structure:
            console.print("[MODEL SUMMARY]\n", model_cfg["repr"])
        console.rule()

        # salva JSON su disco
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
                # aggiorna la config con l'info completa
                run.config.update({"model_info": info}, allow_val_change=True)
                # logga anche la struttura come testo preformattato
                run.log(
                    {"model/structure": model_cfg["repr"]},
                    step=int(getattr(trainer, "global_step", 0)),
                    commit=False,  # do not create a new step yet
                )
                # carica il JSON come file del run
                if out_path is not None:
                    run.save(str(out_path), base_path=str(base_dir))
            except Exception as e:
                warn(f"ModelInfoLogger: W&B log skipped ({type(e).__name__}: {e})")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entrypoint. Fallback a JSON se trainer.use_json=true."""
    # Fallback legacy JSON
    if cfg.trainer.get("use_json", False):
        legacy_path = cfg.trainer.get("json_path")
        if not legacy_path:
            raise ValueError("trainer.use_json=true ma trainer.json_path è vuoto")
        legacy_cfg = load_json(legacy_path)
        ok(f"Caricato legacy JSON: {legacy_path}")
        unified = legacy_cfg
    else:
        # Ricostruisci dizionario unico come prima
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
        }
        ok("Composizione Hydra completata")

    cfg = unified
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, workers=True)

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

    # Inferisci canali dal dataset/prime batch (per settare 'auto' in config modello)
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

    # Patching config modello 'auto'
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

    # build modello e wrapper Lightning
    autoenc = AutoEncoder.from_config(model_cfg)
    # optimizer/scheduler specs will be consumed inside the LightningModule
    optimizer_spec = cfg.get("optimizer", None)
    scheduler_spec = cfg.get("scheduler", None)

    # build wrapper Lightning
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
    )
    ok(f"Instantiated AutoEncoder and Lightning wrapper.")

    # Send eval params to the engine for validation only
    eval_stft_params = (cfg.get("eval_dataset", {}) or {}).get("kwargs", {}) or {}
    wrapper.engine.val_stft_params = eval_stft_params

    # logger (W&B opzionale)
    wandb_cfg = (cfg.get("wandb", {}) or {})
    use_wandb = bool(wandb_cfg.get("use_wandb", False))
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=wandb_cfg.get("project", "ICML_2026"),
            name=wandb_cfg.get("name", None),
            log_model=False,  # evita upload pesanti
            settings=wandb.Settings(_service_wait=7)
        )
        try:
            run = logger.experiment
            # aggiungi la config esatta del parser al config del run
            run.config.update({"parsed_config": cfg}, allow_val_change=True)
            # salva anche il JSON del config nel run (file upload)
            cfg_path = Path("wandb_parsed_config.json")
            cfg_path.write_text(json.dumps(cfg, indent=2))
            run.save(str(cfg_path), base_path=str(cfg_path.parent))
        except Exception as e:
            warn(f"W&B config upload skipped ({type(e).__name__}: {e})")

    # Callback e checkpoint
    ckpt_dir = Path(cfg.get("trainer", {}).get("ckpt_dir", "checkpoints/seanet_stft"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelInfoLogger(filename="model_info.json", max_module_lines=768, log_structure=cfg.get("trainer", {}).get("log_model_structure", False)),
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

    # Demo (validation) dal config opzionale
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

    
    # Profiler: PyTorch Profiler -> TensorBoard
    prof_logdir = Path(cfg.get("trainer", {}).get("profiler_dir", "lightning_profiler"))
    prof_logdir.mkdir(parents=True, exist_ok=True)
    use_profiler = bool(cfg.get("trainer", {}).get("profile", False))
    profiler = None
    if use_profiler:
        profiler = PyTorchProfiler(
            dirpath=str(prof_logdir),
            filename="pl_profile",
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            schedule=torch_profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=torch_profiler.tensorboard_trace_handler(str(prof_logdir)),
            record_shapes=False,          
            profile_memory=False,       
            with_stack=False,             
            profile_dataloader=True,     # evita profiling DataLoader
        )

    # Trainer
    #Set multi-GPU strategy if specified
    hydra_strategy = cfg.get("trainer", {}).get("strategy", "auto")
    hydra_num_gpus = int(cfg.get("trainer", {}).get("num_gpus", 1))
    if hydra_strategy:
        if hydra_strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = hydra_strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if hydra_num_gpus > 1 else "auto"
        
    epochs = int(cfg.get("trainer", {}).get("epochs", 50))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = hydra_num_gpus
    precision = "32-true"  


    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=epochs,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=int(cfg.get("trainer", {}).get("log_interval", 1)),
        num_sanity_val_steps=0,
        gradient_clip_val=0.0,
        detect_anomaly=False,
        profiler=profiler,
        check_val_every_n_epoch=int(cfg.get("trainer", {}).get("check_val_every_n_epoch", 1500)),
        val_check_interval=cfg.get("trainer", {}).get("val_check_interval", None),
    )

    # passa anche val_dataloaders
    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl if eval_dl is not None else None)

if __name__ == "__main__":
    main()