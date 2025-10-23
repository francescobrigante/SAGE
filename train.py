import argparse, json
from pathlib import Path
import torch
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
    def __init__(self, filename: str = "model_info.json", max_module_lines: int = 512, log_structure: bool = False ):
        super().__init__()
        self.filename = filename
        self.max_module_lines = int(max_module_lines)
        self.log_structure = bool(log_structure)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # 1) Estrai info dal modello core (autoencoder dentro il wrapper)
        model = getattr(pl_module, "autoencoder", pl_module)
        info = extract_model_config(model)

        # Limita quante righe della struttura stampare
        modules = info.get("modules", [])[: self.max_module_lines]

        # 2) Stampa su console (info + struttura)
        console.rule("[bold cyan]Model info")
        console.print(f"params total/trainable: {info.get('num_parameters_total')}/{info.get('num_parameters_trainable')}")
        console.rule("[bold cyan]Model structure")
        model_cfg = extract_model_config(model)
        if self.log_structure:
            print("[MODEL SUMMARY]\n", model_cfg["repr"])
        console.rule()

        # Salva JSON su disco
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

        # Logga su W&B (se presente)
        if isinstance(trainer.logger, WandbLogger):
            try:
                run = trainer.logger.experiment
                # aggiorna la config con l'info completa
                run.config.update({"model_info": info}, allow_val_change=True)
                # logga anche la struttura come testo preformattato
                run.log({"model/structure": model_cfg["repr"]})
                # carica il JSON come file del run
                if out_path is not None:
                    run.save(str(out_path), base_path=str(base_dir))
            except Exception as e:
                warn(f"ModelInfoLogger: W&B log skipped ({type(e).__name__}: {e})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment JSON")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (e.g. deepspeed)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (used to pick default strategy)")
    args = parser.parse_args()

    cfg = load_json(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, workers=True)

    # Enable kernel autotuner for more stable conv performance
    torch.backends.cudnn.benchmark = True

    # Dataset
    train_ds = instantiate_from_spec(cfg["train_dataset"])
    dl_cfg = cfg.get("train_dataloader", {}) or {}
    num_workers = int(dl_cfg.get("num_workers", 8))
    train_dl = DataLoader(
        train_ds,
        batch_size=dl_cfg.get("batch_size", 8),
        num_workers=num_workers,
        pin_memory=bool(dl_cfg.get("pin_memory", True)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        drop_last=True,
        persistent_workers=(dl_cfg.get("persistent_workers", False) if num_workers > 0 else False),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 14)) if num_workers > 0 else None,
        collate_fn=collate_stft,
        pin_memory_device=dl_cfg.get("pin_memory_device"),

    )
    
    
    eval_ds = instantiate_from_spec(cfg.get("eval_dataset", None))
    if eval_ds is not None:
        eval_dl = DataLoader(
            eval_ds,
            batch_size=dl_cfg.get("batch_size", 8),
            num_workers=num_workers,
            pin_memory=bool(dl_cfg.get("pin_memory", True)),
            shuffle=False,
            drop_last=False,
            persistent_workers=(dl_cfg.get("persistent_workers", False) if num_workers > 0 else False),
            prefetch_factor=int(dl_cfg.get("prefetch_factor", 14)) if num_workers > 0 else None,
            collate_fn=collate_stft,
            pin_memory_device=dl_cfg.get("pin_memory_device"),
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

    # logger (W&B opzionale)
    wandb_cfg = (cfg.get("wandb", {}) or {})
    use_wandb = bool(wandb_cfg.get("use_wandb", False))
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=wandb_cfg.get("project", "ICML_2026"),
            name=wandb_cfg.get("name", None),
            log_model=False,  # evita upload pesanti
        )

    # Callback e checkpoint
    ckpt_dir = Path(cfg.get("trainer", {}).get("ckpt_dir", "checkpoints/seanet_stft"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelInfoLogger(filename="model_info.json", max_module_lines=768, log_structure=False), 
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch_{epoch:03d}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1,
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
        callbacks.append(
            AutoencoderValDemoCallback(
                every_n_epochs=int(demo_cfg.get("every_n_epochs", 1)),
                max_demos=int(demo_cfg.get("max_demos", 8)),
                sample_rate=int(cfg["train_dataset"]["kwargs"].get("sample_rate", 44100)),
                istft_params=demo_cfg.get("istft_params", None),
                target_seconds=demo_cfg.get("target_seconds", None),
                target_samples=demo_cfg.get("target_samples", None),
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
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"
        
    epochs = int(cfg.get("trainer", {}).get("epochs", 50))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.num_gpus 
    precision = 32  # tensori complessi -> niente AMP

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=epochs,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=int(cfg.get("trainer", {}).get("log_interval", 500)),
        num_sanity_val_steps=0,
        gradient_clip_val=0.0,
        detect_anomaly=False,
        profiler=profiler,  # None se non richiesto
    )

    # passa anche val_dataloaders
    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=eval_dl if eval_dl is not None else None)

if __name__ == "__main__":
    main()