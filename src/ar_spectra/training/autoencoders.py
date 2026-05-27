# =============================================================================
# PyTorch Lightning Wrappers and Callbacks for Model Training and Validation.
# =============================================================================
import os
import torch
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Literal
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from copy import deepcopy
from hydra.utils import instantiate as hydra_instantiate
import torch.nn.functional as F
from safetensors.torch import save_model
from einops import rearrange

# Optional audio libraries for fallback saving
try:
    import torchaudio as _ta
except ImportError:
    _ta = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.io.wavfile import write as wavwrite
except ImportError:
    wavwrite = None

from ..utils.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from .engine import AutoencoderEngine  
from ..models.autoencoder import AutoEncoder
from ..models.inference import encode_audio as inference_encode_audio, decode_audio as inference_decode_audio
from ar_spectra.utils.console import ok, warn, log_metric, log_histogram, log_point_cloud, log_image, log_audio, logger_project_name
from ar_spectra.utils.audio import trim_to_shortest

def _save_audio_with_fallback(path: str, wav_chxn: torch.Tensor, sr: int) -> bool:
    """
    Salva audio tentando nell'ordine:
      1) torchaudio.save (usa TorchCodec; può fallire per FFmpeg/TorchCodec)
      2) soundfile (libsndfile)
      3) scipy.io.wavfile.write
    wav_chxn: (C, N) float32 su CPU o GPU
    """
    # Assicurati di avere (C, N) float32 CPU
    wav = wav_chxn.detach().to(torch.float32).cpu().contiguous()
    # 1) torchaudio
    try:
        if _ta is None:
            raise ImportError("torchaudio not installed")
        _ta.save(path, wav, sr, encoding="PCM_F", bits_per_sample=32)
        return True
    except Exception as e:
        warn(f"torchaudio.save failed ({type(e).__name__}: {e}); trying soundfile...")

    # 2) soundfile
    try:
        if sf is None:
            raise ImportError("soundfile not installed")
        data = wav.transpose(0, 1).numpy()  # (N, C)
        sf.write(path, data, sr, subtype="FLOAT")
        return True
    except Exception as e:
        warn(f"soundfile write failed ({type(e).__name__}: {e}); trying scipy...")

    # 3) scipy
    try:
        if wavwrite is None:
            raise ImportError("scipy not installed")
        data = wav.transpose(0, 1).numpy().astype("float32")  # (N, C)
        wavwrite(path, sr, data)
        return True
    except Exception as e:
        warn(f"scipy.io.wavfile.write failed ({type(e).__name__}: {e})")
        warn(f"Failed to save audio to '{path}' with torchaudio/soundfile/scipy.")
        return False
    
    
class AutoencoderTrainingWrapper(pl.LightningModule):
    """
    Adapter Lightning che delega all'AutoencoderEngine:
    - nessuna logica di loss qui dentro
    - configure_optimizers: crea optimizer/scheduler usando utils e i fallback dal JSON
    - training_step chiama engine.compute() e fa backward/step/logging
    - validation_step chiama engine.compute_validation()
    """
    def __init__(
        self,
        autoencoder: AutoEncoder,
        sample_rate=44100,
        loss_config: Optional[dict] = None,
        eval_loss_config: Optional[dict] = None,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        warmup_mode: Literal["adv", "full"] = "adv",
        encoder_freeze_on_warmup: bool = False,
        force_input_mono: bool = False,
        latent_mask_ratio: float = 0.0,
        teacher_model: Optional[AutoEncoder] = None,
        clip_grad_norm: float = 0.0,
        kl_annealing_epochs: int = 0,
        kl_beta_target: float = 0.0,
        audio_channels: Optional[int] = None,
        model_channels: Optional[int] = None,
        stft_params: Optional[dict] = None,
        optimizer_spec: Optional[dict] = None,
        scheduler_spec: Optional[dict] = None,
        disc_optimizer_spec: Optional[dict] = None,  # if set, used for disc instead of optimizer_spec
        pre_transform_spec: Optional[dict] = None,
        accumulate_grad_batches: int = 1,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.clip_grad_norm = clip_grad_norm
        self.kl_annealing_epochs = kl_annealing_epochs  # 0 = disabled; N = linear warmup over N epochs
        self.kl_beta_target = kl_beta_target             # target β at end of warmup
        self.lr = lr
        self.audio_channels = audio_channels
        self.model_channels = model_channels
        self.stft_params = stft_params
        self.sample_rate = sample_rate

        # Store raw specs (may be None). We normalize in configure_optimizers.
        self._optimizer_spec = optimizer_spec
        self._scheduler_spec = scheduler_spec
        self._disc_optimizer_spec = disc_optimizer_spec  # None → falls back to _optimizer_spec

        self.engine = AutoencoderEngine(
            autoencoder=autoencoder,
            sample_rate=sample_rate,
            loss_config=loss_config,
            eval_loss_config=eval_loss_config,
            warmup_steps=warmup_steps,
            warmup_mode=warmup_mode,
            encoder_freeze_on_warmup=encoder_freeze_on_warmup,
            force_input_mono=force_input_mono,
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model,
            audio_channels=audio_channels,
            stft_params=stft_params
        )

        # Set optional spectrogram normalization on the core autoencoder
        try:
            if pre_transform_spec is not None:
                self.engine.autoencoder.configure_pre_transform(pre_transform_spec)
            ok(self.engine.autoencoder.pre_transform_description(), prefix="PRE TRANSFORM")
        except Exception as e:
            warn(f"Failed to create/apply pre_transform ({type(e).__name__}: {e})")

        # Buffer intermedio per valid
        self.validation_step_outputs = []
        # Track separate accumulation windows for gen/disc manual optimization.
        self._accum_steps_gen = 0
        self._accum_steps_disc = 0
        # Accumulate metrics to log once per optimizer step.
        self._log_accum_gen: Dict[str, float] = {}
        self._log_accum_disc: Dict[str, float] = {}
        self._log_accum_gen_count = 0
        self._log_accum_disc_count = 0

        # Make the model DDP-compatible if it contains ComplexFloat parameters
        from ar_spectra.utils.complex_ddp import make_complex_model_ddp_compatible
        make_complex_model_ddp_compatible(self.engine)

    # Accessori utili in callback esistenti
    @property
    def autoencoder(self):
        return self.engine.autoencoder

    @property
    def discriminator(self):
        return self.engine.loss_manager.discriminator

    @property
    def use_disc(self):
        return self.discriminator is not None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        inference_cfg: Dict[str, Any] = {
            "version": 1,
            "model": self.autoencoder.export_model_config(),
            "in_channels": int(self.model_channels) if self.model_channels is not None else None,
            "audio_channels": int(self.audio_channels) if self.audio_channels is not None else None,
            "stft_config": self.autoencoder.stft_config_dict(),
            "sample_rate": int(self.sample_rate) if self.sample_rate is not None else None,
        }

        if self.stft_params:
            inference_cfg["train_stft_params"] = deepcopy(self.stft_params)

        checkpoint["inference_config"] = {k: v for k, v in inference_cfg.items() if v is not None}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # EMACallback injects ema_autoencoder.* into state_dict at save time.
        # Strip those keys here (before PL calls load_state_dict with strict=True)
        # and stash them so EMACallback.on_load_checkpoint can recover them.
        prefix = "ema_autoencoder."
        state = checkpoint.get("state_dict", {})
        ema_keys = [k for k in list(state) if k.startswith(prefix)]
        if ema_keys:
            checkpoint["_ema_state"] = {k[len(prefix):]: state.pop(k) for k in ema_keys}

    def configure_optimizers(self):
        """
        Create optimizers/schedulers using Hydra instantiate with _target_ format.
        Fallbacks to sensible defaults if specs are missing.
        """
        default_opt = {"_target_": "torch.optim.AdamW", "lr": 2e-4, "betas": [0.8, 0.99]}
        default_sched = {"_target_": "ar_spectra.training.schedulers.InverseLR", "inv_gamma": 200000, "power": 0.5, "warmup": 0.999}

        opt_spec = self._optimizer_spec if self._optimizer_spec else default_opt
        sched_spec = self._scheduler_spec if self._scheduler_spec else default_sched

        # Ensure spec is a mutable copy
        opt_spec = deepcopy(opt_spec) if isinstance(opt_spec, dict) else default_opt
        sched_spec = deepcopy(sched_spec) if isinstance(sched_spec, dict) else default_sched

        # Handle missing _target_ with fallback
        if "_target_" not in opt_spec:
            warn(f"optimizer: missing '_target_', using fallback {default_opt['_target_']}")
            opt_spec = default_opt
        if "_target_" not in sched_spec:
            warn(f"scheduler: missing '_target_', using fallback {default_sched['_target_']}")
            sched_spec = default_sched

        # --- Weight decay exclusion (opt-in via `weight_decay_exclude_1d: true` in YAML) ---
        # Excludes 1-D params (LayerNorm/BN scale+bias) and all .bias from weight decay.
        # Universal good practice for any transformer — not Swin-specific.
        wd_exclude = opt_spec.pop("weight_decay_exclude_1d", False)
        weight_decay = opt_spec.get("weight_decay", 0.0)

        def _build_param_groups(model: torch.nn.Module):
            if not wd_exclude:
                return list(model.parameters())
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # make_complex_model_ddp_compatible stores formerly-1D complex params as
                # 2-D float32 tensors with a "_real_view" suffix (view_as_real adds 1 dim).
                # Treat them the same as 1-D params for weight-decay exclusion so that the
                # complex model sees the same optimisation regime as its real counterpart.
                local_name = name.rsplit(".", 1)[-1]  # param name within its module
                is_complex_ddp_1d = p.ndim == 2 and local_name.endswith("_real_view")
                if p.ndim == 1 or name.endswith(".bias") or is_complex_ddp_1d:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]

        # --- Scheduler interval (opt-in via `interval: step` in YAML) ---
        # Pop before hydra_instantiate so it is not forwarded to the scheduler class.
        sched_interval = sched_spec.pop("interval", "epoch")

        # Create optimizers via Hydra instantiate
        opt_gen = hydra_instantiate(opt_spec, params=_build_param_groups(self.autoencoder), _convert_="all")

        opt_disc = None
        if self.use_disc and self.discriminator is not None:
            # Use disc-specific spec if provided (e.g. different lr/betas/wd), else fall back to gen spec.
            disc_spec_raw = self._disc_optimizer_spec if self._disc_optimizer_spec else opt_spec
            disc_spec = deepcopy(disc_spec_raw) if isinstance(disc_spec_raw, dict) else deepcopy(opt_spec)
            if "_target_" not in disc_spec:
                disc_spec["_target_"] = opt_spec.get("_target_", default_opt["_target_"])
            disc_spec.pop("weight_decay_exclude_1d", None)  # consumed for gen; disc always uses flat param list
            # Discriminator always uses a flat parameter list so that the optimizer-level weight_decay
            # from disc_spec is applied uniformly — no per-group override that would inherit gen's wd.
            opt_disc = hydra_instantiate(disc_spec, params=list(self.discriminator.parameters()), _convert_="all")
            disc_lr_log = disc_spec.get("lr", "?")
        else:
            disc_lr_log = "n/a"

        # Create schedulers via Hydra instantiate
        sched_gen = hydra_instantiate(sched_spec, optimizer=opt_gen, _convert_="all") if sched_spec else None
        sched_disc = hydra_instantiate(sched_spec, optimizer=opt_disc, _convert_="all") if (sched_spec and opt_disc is not None) else None

        ok(
            f"Optimizer: {opt_spec['_target_']} gen_lr={opt_spec.get('lr','?')} disc_lr={disc_lr_log} (wd_exclude_1d={wd_exclude}), "
            f"Scheduler: {sched_spec['_target_']} (interval={sched_interval})",
            prefix="TRAINER",
        )

        # --- Build Lightning return value ---
        # Step-based: return dict format so Lightning calls scheduler.step() every batch.
        # Epoch-based (default): return list format, unchanged from before.
        if sched_interval == "step":
            def _lr_cfg(sched):
                return {"scheduler": sched, "interval": "step", "frequency": 1}
            if self.use_disc and opt_disc is not None and sched_disc is not None:
                return (
                    {"optimizer": opt_gen, "lr_scheduler": _lr_cfg(sched_gen)},
                    {"optimizer": opt_disc, "lr_scheduler": _lr_cfg(sched_disc)},
                )
            return {"optimizer": opt_gen, "lr_scheduler": _lr_cfg(sched_gen)}
        else:
            if self.use_disc and opt_disc is not None:
                if sched_gen is not None and sched_disc is not None:
                    return [opt_gen, opt_disc], [sched_gen, sched_disc]
                return [opt_gen, opt_disc]
            else:
                if sched_gen is not None:
                    return [opt_gen], [sched_gen]
                return [opt_gen]
    def forward(self, reals):
        enc_out = self.engine.autoencoder.encode(reals, return_info=True)
        latents = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        decoded = self.engine.autoencoder.decode(latents)
        return decoded

    def on_train_epoch_start(self) -> None:
        if self.kl_annealing_epochs <= 0:
            return
        epoch = self.current_epoch
        kl_module = self.engine.loss_manager.get_kl_loss_module()
        if kl_module is None:
            return
        if epoch > self.kl_annealing_epochs:
            annealed = self.kl_beta_target
        else:
            annealed = self.kl_beta_target * min(epoch / self.kl_annealing_epochs, 1.0)
            kl_module.update_weight(annealed)
        self.log("train/kl_weight", annealed, on_epoch=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        out = self.engine.compute(batch, global_step=int(self.global_step))
        phase = out["phase"]
        loss_info = out["loss_info"]
        stats = out["stats"]

        accum_batches = self.accumulate_grad_batches
        num_train_batches = int(getattr(self.trainer, "num_training_batches", 0) or 0)
        is_last_batch = (num_train_batches > 0 and (batch_idx + 1) >= num_train_batches)

        # Prendi optimizer e scheduler da Lightning (gestione robusta lista/singolo)
        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, (list, tuple)):
                sched_gen = schedulers[0] if len(schedulers) > 0 else None
                sched_disc = schedulers[1] if len(schedulers) > 1 else None
            else:
                sched_gen, sched_disc = schedulers, None
        else:
            opt_gen = self.optimizers()
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, (list, tuple)):
                sched_gen = schedulers[0] if len(schedulers) > 0 else None
            else:
                sched_gen = schedulers
            opt_disc = sched_disc = None

        # DISC step
        if phase == "disc" and self.use_disc:
            disc_loss = out["disc_total"]

            if self._accum_steps_disc == 0:
                opt_disc.zero_grad()
            self.manual_backward(disc_loss / accum_batches)
            self._accum_steps_disc += 1
            if (self._accum_steps_disc % accum_batches == 0) or is_last_batch:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
                opt_disc.step()
                if sched_disc is not None:
                    sched_disc.step()
                opt_disc.zero_grad()
                self._accum_steps_disc = 0
                # Log once per optimizer step.
                log_dict = {}
                log_dict["train/disc_lr"] = opt_disc.param_groups[0]["lr"]
                if self._log_accum_disc_count > 0:
                    for key, value in self._log_accum_disc.items():
                        log_dict[key] = value / self._log_accum_disc_count
                self.log_dict(log_dict, prog_bar=True, on_step=True)
                self._log_accum_disc = {}
                self._log_accum_disc_count = 0

            # breakdown disc
            for name, value in out["disc_breakdown"].items():
                key = f"train/{name}"
                self._log_accum_disc[key] = self._log_accum_disc.get(key, 0.0) + value.detach().item()
            self._log_accum_disc_count += 1

        # GEN step
        else:
            gen_loss = out["gen_total"]

            if self._accum_steps_gen == 0:
                opt_gen.zero_grad()
            self.manual_backward(gen_loss / accum_batches)
            self._accum_steps_gen += 1
            if (self._accum_steps_gen % accum_batches == 0) or is_last_batch:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.clip_grad_norm)
                opt_gen.step()
                if sched_gen is not None:
                    sched_gen.step()
                opt_gen.zero_grad()
                self._accum_steps_gen = 0
                # Log once per optimizer step.
                log_dict = {}
                log_dict["train/gen_lr"] = opt_gen.param_groups[0]["lr"]
                if self._log_accum_gen_count > 0:
                    for key, value in self._log_accum_gen.items():
                        log_dict[key] = value / self._log_accum_gen_count
                self.log_dict(log_dict, prog_bar=True, on_step=True)
                self._log_accum_gen = {}
                self._log_accum_gen_count = 0

            # logging richiesto: loss, latent_std, data_std
            self._log_accum_gen["train/loss"] = self._log_accum_gen.get("train/loss", 0.0) + gen_loss.detach().item()
            self._log_accum_gen["train/latent_std"] = self._log_accum_gen.get("train/latent_std", 0.0) + stats["latent_std"].detach().item()
            self._log_accum_gen["train/data_std"] = self._log_accum_gen.get("train/data_std", 0.0) + stats["data_std"].detach().item()

            # breakdown gen
            for name, value in out["gen_breakdown"].items():
                key = f"train/{name}"
                self._log_accum_gen[key] = self._log_accum_gen.get(key, 0.0) + value.detach().item()
            self._log_accum_gen_count += 1
        # Ritorna sempre la loss gen per compatibilità
        return out["gen_total"]

    def validation_step(self, batch, batch_idx):
        val_loss_dict = self.engine.compute_validation(batch)
        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

    def test_step(self, batch, batch_idx):
        test_loss_dict = self.engine.compute_validation(batch)
        for k, v in test_loss_dict.items():
            self.log(f"test/{k}", v, prog_bar=True, sync_dist=True)
        return test_loss_dict

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for k, v in loss_dict.items():
                sum_loss_dict[k] = sum_loss_dict.get(k, 0.0) + v

        for k, v in sum_loss_dict.items():
            avg = v / len(self.validation_step_outputs)
            avg = self.all_gather(torch.tensor(avg, device=self.device)).mean().item()
            log_metric(self.logger, f"val/{k}", avg)
        self.validation_step_outputs.clear()

    def export_model(self, path, use_safetensors=False):
        model = self.autoencoder_ema.ema_model if getattr(self, "autoencoder_ema", None) is not None else self.autoencoder
        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)


class AutoencoderValDemoCallback(pl.Callback):
    """
    Log demo su validation:
    - usa il batch di validation (no grad)
    - decode -> istft con override opzionale dei parametri
    - opzionale: forzare lunghezza target dei segnali demo
    """
    def __init__(
        self,
        every_n_epochs: int = 1,
        max_demos: int = 8,
        sample_rate: Optional[int] = None,
        istft_params: Optional[dict] = None,
        target_seconds: Optional[float] = None,
        target_samples: Optional[int] = None,
        save_basename: str = "recon_val",
    ):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.max_demos = int(max_demos)
        self.sample_rate = sample_rate
        self.istft_params = istft_params or {}
        self.target_seconds = target_seconds
        self.target_samples = target_samples
        self.save_basename = save_basename
        self._last_logged_epoch = -1

    @staticmethod
    def _crop_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B,C,N)
        n = x.shape[-1]
        if n == target_len:
            return x
        if n > target_len:
            start = (n - target_len) // 2
            return x[..., start:start+target_len]
        pad = target_len - n
        left = pad // 2
        right = pad - left
        return F.pad(x, (left, right), mode="constant", value=0.0)

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # logga solo sul primo batch di ogni epoca, con frequenza every_n_epochs
        epoch = int(trainer.current_epoch)
        is_sanity = getattr(trainer, "sanity_checking", False)
        if batch_idx != 0 or (epoch % self.every_n_epochs) != 0 or (self._last_logged_epoch == epoch and not is_sanity):
            return
        # non marca l'epoca come "già loggata" se siamo nel sanity-check,
        # così il logging reale alla fine dell'epoca può ancora avvenire.
        if not is_sanity:
            self._last_logged_epoch = epoch

        sp_reals, reals_wav = batch
        sp_reals = sp_reals.to(pl_module.device, non_blocking=True)
        reals_wav = reals_wav.to(pl_module.device, non_blocking=True)

        # limita il numero di esempi
        if sp_reals.shape[0] > self.max_demos:
            sp_reals = sp_reals[:self.max_demos, ...]
            reals_wav = reals_wav[:self.max_demos, ...]

        with torch.no_grad():
            force_mono = bool(getattr(pl_module.engine, "force_input_mono", False))
            waveform_input = reals_wav
            if force_mono and waveform_input.shape[1] > 1:
                waveform_input = waveform_input.mean(dim=1, keepdim=True)
            waveform_input = waveform_input.contiguous()

            try:
                latents = inference_encode_audio(
                    pl_module.autoencoder,
                    waveform_input,
                    debug=False,
                )
                decoded = inference_decode_audio(
                    pl_module.autoencoder,
                    latents,
                    debug=False,
                )
            except Exception as exc:
                warn(
                    f"Validation demo inference pipeline failed ({type(exc).__name__}: {exc}). "
                    "Skipping demo logging for this batch."
                )
                return

            decoded = decoded.detach()
            reference_audio = waveform_input.detach()

            # allinea a reals e applica target length opzionale
            decoded, reference_audio = trim_to_shortest(decoded, reference_audio)

            sr = int(self.sample_rate or getattr(pl_module.engine, "sample_rate", 44100))
            if self.target_seconds is not None and self.target_seconds > 0:
                target_len = int(round(self.target_seconds * sr))
            elif self.target_samples is not None and self.target_samples > 0:
                target_len = int(self.target_samples)
            else:
                target_len = decoded.shape[-1]

            if target_len != decoded.shape[-1]:
                decoded = self._crop_or_pad(decoded, target_len)
                reference_audio = self._crop_or_pad(reference_audio, target_len)

            input_encoder_audio = reference_audio

            reals_fakes = rearrange([reference_audio, decoded], 'i b d n -> (b i) d n')
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
            encoder_input_istft = rearrange([reference_audio, input_encoder_audio], 'i b d n -> (b i) d n')
            encoder_input_istft = rearrange(encoder_input_istft, 'b d n -> d (b n)')

            # path di salvataggio
            try:
                data_dir = os.path.join(
                    trainer.logger.save_dir, logger_project_name(trainer.logger),
                    getattr(getattr(trainer.logger, "experiment", None), "id", "offline"), "media")
                os.makedirs(data_dir, exist_ok=True)
                filename = os.path.join(data_dir, f'{self.save_basename}_ep{epoch:04d}.wav')
                filename_input_encoder_istft = os.path.join(data_dir, f'{self.save_basename}_input_encoder_istft_ep{epoch:04d}.wav')

            except Exception:
                filename = f'{self.save_basename}_ep{epoch:04d}.wav'
                filename_input_encoder_istft = f'{self.save_basename}_input_encoder_istft_ep{epoch:04d}.wav'
                

            # Salva in float32 per evitare clipping
            # Pre-normalize per evitare clipping: scala se il picco supera 1.0
            eps = 1e-9
            # assicurati che tutti i tensori stiano sullo stesso device prima di fare torch.stack()
            device = getattr(pl_module, "device", None) or (reference_audio.device if torch.is_tensor(reference_audio) else torch.device("cpu"))
            peak_real = (reference_audio.abs().max().detach().to(device) if torch.is_tensor(reference_audio) else torch.tensor(0.0, device=device))
            peak_dec = (decoded.abs().max().detach().to(device) if torch.is_tensor(decoded) else torch.tensor(0.0, device=device))
            peak_enc = (encoder_input_istft.abs().max().detach().to(device) if torch.is_tensor(encoder_input_istft) else torch.tensor(0.0, device=device))
            global_peak = float(torch.max(torch.stack([peak_real, peak_dec, peak_enc, torch.tensor(eps, device=device)])).item())
            if global_peak > 1.0:
                scale = 1.0 / global_peak
                reals_fakes = reals_fakes * scale
                encoder_input_istft = encoder_input_istft * scale

            # Converti a float32 per il salvataggio
            wav_reals_fakes_f32 = reals_fakes.detach().to(torch.float32).cpu()
            wav_input_encoder_istft_f32 = encoder_input_istft.detach().to(torch.float32).cpu()

            # Prova prima torchaudio (TorchCodec), poi fallback a soundfile/scipy
            saved_enc = _save_audio_with_fallback(filename_input_encoder_istft, wav_input_encoder_istft_f32, sr)
            saved_rec = _save_audio_with_fallback(filename, wav_reals_fakes_f32, sr)

            if saved_rec:
                log_audio(trainer.logger, 'val/recon', filename, sr)
            if saved_enc:
                log_audio(trainer.logger, 'val/input_encoder_istft', filename_input_encoder_istft, sr)
            lat_to_log = latents[0] if isinstance(latents, (tuple, list)) else latents
            log_point_cloud(trainer.logger, 'val/embeddings_3dpca', lat_to_log)
            log_image(trainer.logger, 'val/embeddings_spec', tokens_spectrogram_image(lat_to_log))
            log_image(trainer.logger, 'val/recon_melspec_left', audio_spectrogram_image(wav_reals_fakes_f32))
