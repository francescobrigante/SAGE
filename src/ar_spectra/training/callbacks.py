import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
import json
import torch
import copy
from pathlib import Path
from rich.console import Console

from ar_spectra.utils.console import ok, warn
from ar_spectra.utils.model_info import extract_model_config, log_compression_stats

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
        # AutoencoderTrainingWrapper.on_load_checkpoint strips ema_autoencoder.* from state_dict
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


class ValFADCallback(Callback):
    """Compute a CLAP-based FAD ("val/fad_clap") during validation, on a FIXED
    corpus of full FMA test-split files, and log a single scalar to W&B.

    Why a separate corpus instead of the val batch: FAD is a *distributional*
    metric (one Frechet over all embeddings, not a per-sample average), and CLAP
    is tuned for ~10 s windows while the val loader serves ~1.5 s chunks. So this
    callback runs its own pass over `num_files` full test files, embeds the
    reconstructions with CLAP, and compares against the *cached* target
    embeddings under ``cache_dir/<clap_name>/target/`` (reused from the offline
    eval pipeline — no target recompute).

    Registration: MUST be appended BEFORE ``EMACallback`` so this hook's
    ``on_validation_epoch_end`` runs while EMA weights are still swapped into
    ``pl_module.autoencoder`` (EMACallback restores live weights in *its* own
    ``on_validation_epoch_end``, which runs after by list order). So the FAD is
    computed on the EMA weights — consistent with the offline eval.

    Heavy deps (fadtk / torchaudio / evaluation utils) are imported LAZILY inside
    the hooks so importing this module stays free for CPU unit tests. Every
    failure path warns and no-ops — it must never crash training.
    """

    def __init__(
        self,
        cache_dir: str,
        fma_csv_path: str,
        audio_root: str,
        num_files: int = 2000,
        fad_model: str = "clap-laion-music",
        num_downsamples=None,
        audio_exts=(".mp3", ".wav", ".flac"),
        enabled: bool = False,
    ):
        super().__init__()
        self.cache_dir = str(cache_dir)
        self.fma_csv_path = str(fma_csv_path) if fma_csv_path else None
        self.audio_root = str(audio_root)
        self.num_files = int(num_files)
        self.fad_model = str(fad_model)
        self.num_downsamples = num_downsamples
        self.audio_exts = tuple(audio_exts)
        self.enabled = bool(enabled)
        self._ready = False

    # ── Pure helpers (no torch/fadtk → unit-testable on CPU) ──────────────────
    @staticmethod
    def _shard(files, rank, world_size):
        """Deterministic disjoint partition of `files` across ranks."""
        ws = max(int(world_size), 1)
        return files[int(rank)::ws]

    @staticmethod
    def _build_file_list(all_files, tgt_dir):
        """Keep only files that have a cached target embedding .npy."""
        tgt_dir = Path(tgt_dir)
        return [f for f in all_files if (tgt_dir / f"{f.stem}.npy").exists()]

    @staticmethod
    def _target_stats(files, tgt_dir):
        """Pool cached target embeddings → (mu[D], cov[D,D]), computed once."""
        import numpy as np
        tgt_dir = Path(tgt_dir)
        embs = [np.load(tgt_dir / f"{f.stem}.npy").astype(np.float32) for f in files]
        all_t = np.concatenate(embs, axis=0)
        return all_t.mean(0), np.cov(all_t, rowvar=False)

    @staticmethod
    def _frechet_from_embs(target_mu, target_cov, pred_list, calc_fd):
        """Single Frechet distance over the union of prediction embeddings."""
        import numpy as np
        all_p = np.concatenate(pred_list, axis=0).astype(np.float32)
        return float(calc_fd(target_mu, target_cov, all_p.mean(0), np.cov(all_p, rowvar=False)))

    # ── Hooks ─────────────────────────────────────────────────────────────────
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        self._ready = False
        try:
            import sys
            proj_root = Path(__file__).resolve().parents[3]      # .../C-VAE
            eval_dir = str(proj_root / "evaluation")
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)

            from compute_clap_score import embed_clap            # evaluation/compute_clap_score.py
            from utils import collect_fma_files                  # evaluation/utils.py
            from fadtk.fad import calc_frechet_distance
            from fadtk.model_loader import CLAPLaionModel
            import fadtk.model_loader as _ml_mod
            import numpy as np

            # Precheck offline CLAP weights (live in the active venv's fadtk, NOT the repo)
            ckpt = Path(_ml_mod.__file__).parent / ".model-checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"
            if not ckpt.exists():
                warn(f"ValFADCallback: CLAP weights missing at {ckpt}; disabling.", prefix="FAD")
                return

            clap = CLAPLaionModel("music")                       # .name=clap-laion-music, .sr=48000
            clap.load_model()
            clap.model.to(pl_module.device)

            self._clap = clap
            self._embed_clap = embed_clap
            self._calc_fd = calc_frechet_distance
            self._np = np

            # Fixed corpus (sorted, FMA-test-filtered, capped), then intersect with cached targets
            all_files = collect_fma_files(
                Path(self.audio_root), set(self.audio_exts), self.fma_csv_path, self.num_files
            )
            tgt_dir = Path(self.cache_dir) / clap.name / "target"
            files_with_target = self._build_file_list(all_files, tgt_dir)
            if not files_with_target:
                warn("ValFADCallback: no cached target embeddings intersect corpus; disabling.",
                     prefix="FAD")
                return

            # Precompute target mu/cov ONCE (avoids reloading every validation)
            self._target_mu, self._target_cov = self._target_stats(files_with_target, tgt_dir)

            ws = int(getattr(trainer, "world_size", 1) or 1)
            rk = int(getattr(trainer, "global_rank", 0) or 0)
            self._my_files = self._shard(files_with_target, rk, ws)

            depths = getattr(getattr(pl_module.autoencoder, "encoder", None), "depths", None)
            self._num_downsamples = (
                int(self.num_downsamples) if self.num_downsamples is not None
                else (len(depths) - 1 if depths else 2)
            )

            self._ready = True
            ok(f"ValFADCallback ready: {len(files_with_target)} target files "
               f"(rank {rk}/{ws} owns {len(self._my_files)}), "
               f"num_downsamples={self._num_downsamples}.", prefix="FAD")
        except Exception as e:
            warn(f"ValFADCallback on_fit_start failed ({type(e).__name__}: {e}); disabling.",
                 prefix="FAD")
            self._ready = False

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not getattr(self, "_ready", False):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        # Run on EVERY validation. Cadence is controlled by Lightning's
        # check_val_every_n_epoch (=2 → FAD every 2 epochs). Do NOT add an internal
        # epoch-parity gate: Lightning validates at (current_epoch+1) % check_val == 0
        # (odd epochs for check_val=2), so a `current_epoch % N == 0` test would skip
        # every validation.
        try:
            self._run_fad(trainer, pl_module)
        except Exception as e:
            warn(f"ValFADCallback validation pass failed ({type(e).__name__}: {e}); skipping.",
                 prefix="FAD")

    def _run_fad(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import sys
        import numpy as np
        eval_dir = str(Path(__file__).resolve().parents[3] / "evaluation")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        from evaluate_swin_varT import _AudioDataset, pad_audio_for_swin
        from ar_spectra.models.inference import encode_audio, decode_audio

        ae = pl_module.autoencoder              # EMA weights (swap-in still active)
        was_training = ae.training
        ae.eval()
        sr = int(pl_module.sample_rate)
        ch = int(pl_module.audio_channels)
        hop = int(ae._stft_config.hop_length)
        force_mono = bool(getattr(pl_module.engine, "force_input_mono", False))
        wav_cache = Path(self.cache_dir) / "wav_cache"

        ds = _AudioDataset(self._my_files, target_sr=sr, target_ch=ch, cache_dir=wav_cache)
        pred_embs = []
        with torch.no_grad():
            for i in range(len(ds)):
                wav, _stem = ds[i]
                if wav is None:
                    continue
                wav_in = wav.mean(0, keepdim=True) if (force_mono and wav.shape[0] > 1) else wav
                wav_pad, orig_len = pad_audio_for_swin(wav_in, hop, self._num_downsamples)
                x = wav_pad.unsqueeze(0).to(pl_module.device)            # [1, C, T_pad]
                latents = encode_audio(ae, x)
                pred = decode_audio(ae, latents, target_length=orig_len)  # [1, C, T]
                pred = pred[0].float().cpu()                             # [C, T] for embed_clap
                emb = self._embed_clap(self._clap, pred, sr, pl_module.device)  # (n_chunks, 512)
                pred_embs.append(np.asarray(emb, dtype=np.float32))
        if was_training:
            ae.train()
        torch.cuda.empty_cache()

        # Gather ragged per-rank embedding lists (counts differ → all_gather_object, not all_gather)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, pred_embs)
            all_pred = [e for shard in gathered for e in shard]
        else:
            all_pred = pred_embs

        if int(getattr(trainer, "global_rank", 0)) == 0:
            from ar_spectra.utils.console import log_metric
            if not all_pred:
                warn("ValFADCallback: no pred embeddings this validation; skipping log.", prefix="FAD")
                return
            score = self._frechet_from_embs(self._target_mu, self._target_cov, all_pred, self._calc_fd)
            log_metric(trainer.logger, "val/fad_clap", score)
            ok(f"val/fad_clap = {score:.4f} (epoch {int(trainer.current_epoch)})", prefix="FAD")
