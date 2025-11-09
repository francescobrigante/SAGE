import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any, Tuple

from ..models.autoencoder import AutoEncoder
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator
from ..models.bottlenecks import VAEBottleneck
from .losses import (
    MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, TargetValueLoss,
    L1Loss, LossWithTarget, MSELoss, HubertLoss, 
)
from .losses import auraloss as auraloss
from .losses.ar_spectra_losses import ComplexSpectralConvergence, MultiResSpectralConvergence, ComplexMSE
from .utils import create_optimizer_from_config, create_scheduler_from_config
from rich.console import Console
console = Console()  

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")


def trim_to_shortest(a, b):
    if a.shape[-1] > b.shape[-1]:
        return a[..., :b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[..., :a.shape[-1]]
    return a, b

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if isinstance(bottleneck, VAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)


    return losses

class AutoencoderEngine(nn.Module):
    """
    - Costruisce discriminator, loss e metriche eval (se richieste)
    - compute(batch, global_step) -> dict con phase, loss totali e breakdown, loss_info
    - compute_validation(batch) -> dict metriche validation (CPU scalari)
    - configure_optimizers() -> ottimizzatori + scheduler dal config
    Nessun backward/step/logging qui dentro.
    """
    def __init__(self, 
                 autoencoder: AutoEncoder,
                 sample_rate: int = 48000,
                 loss_config: Optional[dict] = None,
                 eval_loss_config: Optional[dict] = None,
                 optimizer_configs: Optional[dict] = None,
                 warmup_steps: int = 0,
                 warmup_mode: Literal["adv", "full"] = "adv",
                 encoder_freeze_on_warmup: bool = False,
                 force_input_mono: bool = False,
                 latent_mask_ratio: float = 0.0,
                 teacher_model: Optional[AutoEncoder] = None,
                 audio_channels: Optional[int] = None,
                 stft_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.autoencoder = autoencoder
        self.teacher_model = teacher_model
        self.sample_rate = sample_rate
        self.stft_params = stft_params or {}   # training params
        self.val_stft_params = None            # eval params injected by train.py

        # training policy
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.force_input_mono = force_input_mono
        self.latent_mask_ratio = latent_mask_ratio

        # optimizer configs per adapters
        if optimizer_configs is None:
            optimizer_configs = {
                "autoencoder": {"optimizer": {"type": "AdamW", "config": {"lr": 2e-4, "betas": (0.8, 0.99)}}},
                "discriminator": {"optimizer": {"type": "AdamW", "config": {"lr": 2e-4, "betas": (0.8, 0.99)}}},
            }
        # NOTE: DEPRECATED, mantenuto per retrocompatibilità
        self.optimizer_configs = optimizer_configs 

        # Numero di canali del segnale audio (mono/stereo)
        if audio_channels is None:
            audio_channels = 2  # default conservative
        self.audio_channels = int(audio_channels)

        # usa il loss_config passato dal chiamante
        if loss_config is None:
            loss_config = {}
        if not isinstance(loss_config, dict):
            warn("AutoencoderEngine: provided loss_config is not a dict — treating as empty config")
            loss_config = {}
        self.loss_config = loss_config
        # use discriminator only if present and truthy
        self.use_disc = bool(self.loss_config.get("discriminator"))
 
         # reconstruction losses: spectral può non essere presente nella config -> protegge l'accesso
        spectral_cfg = self.loss_config.get("spectral")
        if spectral_cfg and isinstance(spectral_cfg, dict):
            # JSON corrente: "spectral": { "stft_mse": { "config": {...} }, "weights": {...} }
            stft_mse_block = spectral_cfg.get("stft_mse", {}) or {}
            configs_stft_mse = stft_mse_block.get("config", {}) or {}
            self.stft_mse = ComplexMSE(**configs_stft_mse)
        else:

            self.stft_mse = None

        if spectral_cfg and isinstance(spectral_cfg, dict):
            # JSON: "spectral": { "<one-of> mrstft_stable_audio | mrstft | mrstft_sc": { "config": {...} }, "weights": {...} }
            # select one of the possible mrstft losses
            mr_keys = [k for k in ("mrstft_stable_audio", "mrstft", "mrstft_sc") if k in spectral_cfg]
            assert len(mr_keys) <= 1, (
                "loss_config.spectral: you need to specify at most one of the keys:"
                "'mrstft_stable_audio', 'mrstft', 'mrstft_sc'. "
                f"Found: {mr_keys}"
            )
            if len(mr_keys) == 0:
                self.mrstft = None
            else:
                chosen = mr_keys[0]
                mrstft_block = spectral_cfg.get(chosen, {}) or {}
                # accetta sia {"config": {...}} sia dizionario piatto
                configs_mrstft = mrstft_block.get("config", mrstft_block) or {}
                if chosen in ("mrstft_stable_audio", "mrstft"):
                    # Variante auraloss classica
                    self.mrstft = auraloss.MultiResolutionSTFTLoss(**configs_mrstft)
                elif chosen == "mrstft_sc":
                    # Variante basata su Spectral Convergence
                    self.mrstft = MultiResSpectralConvergence(**configs_mrstft)
        # per evitare AttributeError in rami opzionali
        self.lrstft = None

        # Discriminator
        self.discriminator = None
        if self.use_disc:
            disc_type = self.loss_config['discriminator']['type']
            disc_cfg = self.loss_config['discriminator']['config']
            if disc_type == 'oobleck':
                self.discriminator = OobleckDiscriminator(**disc_cfg)
            elif disc_type == 'encodec':
                self.discriminator = EncodecDiscriminator(in_channels=self.audio_channels, **disc_cfg)
            elif disc_type == 'dac':
                self.discriminator = DACGANLoss(channels=self.audio_channels, sample_rate=sample_rate, **disc_cfg)
            elif disc_type == 'big_vgan':
                self.discriminator = BigVGANDiscriminator(channels=self.audio_channels, sample_rate=sample_rate, **disc_cfg)

        # Generator composite loss
        gen_loss_modules = []
        if self.use_disc:
            gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss'),
            ]

        stft_loss_decay = spectral_cfg.get('decay', 1.0) if spectral_cfg else 1.0
        # if spectral is available, we add the spectral loss 
        if spectral_cfg:
            if self.stft_mse is not None:
                stft_mse_weight = spectral_cfg['weights'].get('stft_mse', 0.0)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.stft_mse,
                        input_key='sp_decoded',      # spettrogramma predetto
                        target_key='encoder_input',  # spettrogramma target
                        name='pwc_mse_loss',
                        weight=stft_mse_weight,
                        decay=stft_loss_decay,
                    )
                )
            if self.mrstft is not None:
                stft_mse_weight = spectral_cfg['weights'].get('mrstft', 0.0)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.mrstft,
                        target_key='reals',
                        input_key='decoded',
                        name='mrstft_loss',
                        weight=stft_mse_weight,
                        decay=stft_loss_decay
                    )
                )
                

        if "mrmel" in self.loss_config:
             mrmel_weight = self.loss_config["mrmel"]["weights"]["mrmel"]
             if mrmel_weight > 0:
                 mrmel_config = self.loss_config["mrmel"]["config"]
                 self.mrmel = MelSpectrogramLoss(self.sample_rate,
                     n_mels=mrmel_config["n_mels"],
                     window_lengths=mrmel_config["window_lengths"],
                     pow=mrmel_config["pow"],
                     log_weight=mrmel_config["log_weight"],
                     mag_weight=mrmel_config["mag_weight"],
                 )
                 gen_loss_modules.append(LossWithTarget(self.mrmel, "reals", "decoded", name="mrmel_loss", weight=mrmel_weight))

        if "hubert" in self.loss_config:
            hubert_weight = self.loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = self.loss_config["hubert"].get("config", {})
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)
                gen_loss_modules.append(LossWithTarget(self.hubert, target_key="reals", input_key="decoded", name="hubert_loss", weight=hubert_weight, decay=self.loss_config["hubert"].get("decay", 1.0)))

        if "time" in self.loss_config:
            if self.loss_config["time"]["weights"].get("l1", 0.0) > 0.0:
                gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded', weight=self.loss_config["time"]["weights"]["l1"], name='l1_time_loss', decay=self.loss_config["time"].get('decay', 1.0)))
            if self.loss_config["time"]["weights"].get("l2", 0.0) > 0.0:
                gen_loss_modules.append(MSELoss(key_a='reals', key_b='decoded', weight=self.loss_config["time"]["weights"]["l2"], name='l2_time_loss', decay=self.loss_config["time"].get('decay', 1.0)))

        if self.autoencoder.bottleneck is not None:
            gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(gen_loss_modules)

        # Disc losses
        self.losses_disc = None
        if self.use_disc:
            self.losses_disc = MultiLoss([ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss')])

        # Eval losses
        self.eval_losses = nn.ModuleDict()
        if eval_loss_config is not None:
            if "stft" in eval_loss_config:
                self.eval_losses["stft"] = auraloss.STFTLoss(**eval_loss_config["stft"])
            if "sisdr" in eval_loss_config:
                self.eval_losses["sisdr"] = auraloss.SISDRLoss(**eval_loss_config["sisdr"])
            if "mel" in eval_loss_config:
                self.eval_losses["mel"] = auraloss.MelSTFTLoss(self.sample_rate, **eval_loss_config["mel"])
                
        ok("AutoencoderEngine: initialized with train losses: " +
           f"gen: {[type(l).__name__ for l in gen_loss_modules]}, " +
           (f"disc: {[type(l).__name__ for l in self.losses_disc.modules()]}" if self.use_disc else "no disc"))
        ok(f"AutoencoderEngine: initialized with eval losses: {list(self.eval_losses.keys())}")
        # Log dettagliato dei parametri delle loss
        self._log_losses_summary()

    @torch.no_grad()
    def _encode_teacher_if_needed(self, encoder_input):
        if self.teacher_model is None:
            return None
        return self.teacher_model.encode(encoder_input, return_info=False)

    def _resolve_istft_kwargs(self, override: Optional[dict] = None) -> dict:
        # training path (unchanged): base = training, override replaces
        # Do not require/expect 'length' here anymore. If present, it will be ignored.
        allowed = {"n_fft", "hop_length", "win_length", "center", "normalized", "window", "onesided"}
        base = {k: v for k, v in (self.stft_params or {}).items() if k in allowed}
        if override:
            for k, v in override.items():
                if k not in allowed:
                    continue
                if isinstance(v, str) and v.lower() == "auto":
                    if k not in base:
                        base.pop(k, None)
                    continue
                base[k] = v
        for k in list(base.keys()):
            if isinstance(base[k], str) and base[k].lower() == "auto":
                base.pop(k, None)
        return base

    def _resolve_eval_istft_kwargs(self) -> dict:
        """
        Validation: usa come base i parametri dell'eval; se un campo è 'auto' o mancante,
        fallback ai parametri di training.
        """
        # Keep eval/training ISTFT keys minimal; 'length' is not required and will be ignored if supplied
        allowed = {"n_fft", "hop_length", "win_length", "center", "normalized", "window", "onesided"}
        train_base = {k: v for k, v in (self.stft_params or {}).items() if k in allowed}
        eval_raw = getattr(self, "val_stft_params", None) or {}
        eval_clean = {}
        for k, v in eval_raw.items():
            if k not in allowed:
                continue
            if isinstance(v, str) and v.lower() == "auto":
                continue  # lascia il fallback al training
            eval_clean[k] = v
        # fallback prima, poi override con eval pulito (eval ha priorità)
        out = dict(train_base)
        out.update(eval_clean)
        # ripulisci eventuali 'auto' avanzati (da training unlikely)
        for k in list(out.keys()):
            if isinstance(out[k], str) and out[k].lower() == "auto":
                out.pop(k, None)
        return out

    def compute(self, batch: Tuple[torch.Tensor, torch.Tensor], global_step: int) -> Dict[str, Any]:
        """
        batch: (sp_reals, orig_waveforms)
        Ritorna:
          - phase: "gen" | "disc"
          - gen_total (tensor), gen_breakdown (dict[str,tensor])
          - disc_total, disc_breakdown (se presente)
          - loss_info: dict con tensori utili (encoder_input, latents, decoded, reals, ...)
          - stats: {data_std, latent_std}
        """
        sp_reals, orig_waveforms = batch
        loss_info: Dict[str, Any] = {}

        encoder_input = sp_reals
        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)
        loss_info["encoder_input"] = encoder_input  # spettrogramma GT
        loss_info["reals"] = orig_waveforms         # waveform GT

        warmed_up = (global_step >= self.warmup_steps)

        # Encode
        if warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        loss_info["latents"] = latents
        loss_info.update(encoder_info)

        # Distillation
        teacher_latents = self._encode_teacher_if_needed(encoder_input)
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)
            loss_info["latents"] = latents

        # Decode STFT -> waveform
        sp_decoded = self.autoencoder.decode(latents)   # spettrogramma predetto
        # Prima proviamo senza params (se l'istft interna è in grado di gestire spec complessi)

        # fallback: usa parametri di training risolti/filtrati
        istft_kwargs = self._resolve_istft_kwargs()
        if istft_kwargs:
            decoded = self.autoencoder.istft(sp_decoded, **istft_kwargs)
        else:
            raise ValueError(
                "autoencoder.istft failed and no usable istft params available. "
                "Ensure 'n_fft' (e correlati) are present in the dataset kwargs or resolved from config."
            )

        # allinea alle waveform reali (non usare l’inversione dell’input)
        decoded, orig_waveforms = trim_to_shortest(decoded, orig_waveforms)

        loss_info["decoded"] = decoded              # waveform predetta
        loss_info["reals"] = orig_waveforms         # waveform GT
        loss_info["sp_decoded"] = sp_decoded        # spettrogramma predetto

        # decoded/reals shape attesa: (B, C_audio, N)
        if self.audio_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = orig_waveforms[:, 0:1, :]
            loss_info["reals_right"] = orig_waveforms[:, 1:2, :]

        if teacher_latents is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents)
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents)
            loss_info['teacher_latents'] = teacher_latents
            loss_info['teacher_decoded'] = teacher_decoded
            loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
            loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        # Discriminator (solo computo loss)
        disc_total = None
        disc_breakdown: Dict[str, torch.Tensor] = {}
        if self.use_disc:
            if warmed_up:
                loss_dis, loss_adv, feat_match = self.discriminator.loss(reals=orig_waveforms, fakes=decoded)
            else:
                if self.warmup_mode == "adv":
                    loss_dis, _, _ = self.discriminator.loss(reals=orig_waveforms, fakes=decoded)
                else:
                    loss_dis = torch.tensor(0.0, device=decoded.device)
                loss_adv = torch.tensor(0.0, device=decoded.device)
                feat_match = torch.tensor(0.0, device=decoded.device)

            loss_info["loss_dis"] = loss_dis
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = feat_match

            disc_total, disc_breakdown = self.losses_disc(loss_info)

        gen_total, gen_breakdown = self.losses_gen(loss_info)

        # Stats per logging
        data_std = loss_info["encoder_input"].std()
        latent_std = loss_info["latents"].std()
        stats = {"data_std": data_std, "latent_std": latent_std}

        # Alternanza fase
        use_disc_phase = (
            self.use_disc
            and (global_step % 2 == 1)
            and ((self.warmup_mode == "full" and warmed_up) or self.warmup_mode == "adv")
        )

        return {
            "phase": "disc" if use_disc_phase else "gen",
            "gen_total": gen_total,
            "gen_breakdown": gen_breakdown,
            "disc_total": disc_total,
            "disc_breakdown": disc_breakdown,
            "loss_info": loss_info,
            "stats": stats,
        }

    @torch.no_grad()
    def compute_validation(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        sp_reals, orig_waveforms = batch
        encoder_input = sp_reals
        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        latents, _ = self.autoencoder.encode(encoder_input, return_info=True)
        sp_decoded = self.autoencoder.decode(latents)
        
        istft_kwargs = self._resolve_eval_istft_kwargs()
        if istft_kwargs:
            decoded = self.autoencoder.istft(sp_decoded, **istft_kwargs)
        else:
            raise ValueError(
                "autoencoder.istft failed and no usable istft params available. "
                "Ensure ISTFT params are present in eval/train kwargs."
            )
        decoded, orig_waveforms = trim_to_shortest(decoded, orig_waveforms)

        val_loss_dict: Dict[str, float] = {}
        for eval_key, eval_fn in self.eval_losses.items():
            value = eval_fn(decoded, orig_waveforms)
            if eval_key == "sisdr":
                value = -value
            if isinstance(value, torch.Tensor):
                value = value.item()
            val_loss_dict[eval_key] = value
        return val_loss_dict

    def configure_optimizers(self):
        """
        Ritorna gli stessi formati attesi da Lightning:
        - solo gen: [opt_gen] o ([opt_gen],[sched_gen])
        - con disc: [opt_gen, opt_disc] o ([opt_gen, opt_disc],[sched_gen, sched_disc])
        Gli adapter Fabric possono usare lo stesso metodo e gestire i ritorni.
        """
        gen_params = list(self.autoencoder.parameters())
        if self.discriminator is not None:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())
            sched_gen = sched_disc = None
            if ("scheduler" in self.optimizer_configs['autoencoder']) and ("scheduler" in self.optimizer_configs['discriminator']):
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['autoencoder']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def _extract_hparams(self, module: nn.Module) -> dict:
        # Estrae solo attributi semplici stampabili
        simple = {}
        for k, v in vars(module).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, bool, str, type(None))):
                simple[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, bool, str)) for x in v):
                simple[k] = v
        return simple

    def _log_losses_summary(self):
        ok("===== Loss setup summary =====")
        # Generator losses
        spectral_cfg = self.loss_config.get("spectral", {}) or {}
        if self.stft_mse is not None:
            ok(f"- PerceptuallyWeightedComplexMSE: {self._extract_hparams(self.stft_mse)}")
        if self.mrstft is not None:
            try:
                params = self._extract_hparams(self.mrstft)
            except Exception:
                params = {}
            ok(f"- {type(self.mrstft).__name__}: {params}")
        if spectral_cfg:
            ok(f"- Spectral weights: {spectral_cfg.get('weights', {})}")

        if "time" in self.loss_config:
            ok(f"- Time-domain weights: {self.loss_config['time'].get('weights', {})}")
        if "mrmel" in self.loss_config:
            ok(f"- MR-Mel weight: {self.loss_config['mrmel'].get('weights', {})}")
        if "hubert" in self.loss_config:
            ok(f"- HuBERT weight/decay: {self.loss_config['hubert'].get('weights', {})}, decay={self.loss_config['hubert'].get('decay', 1.0)}")

        if self.use_disc:
            dcfg = self.loss_config.get("discriminator", {})
            ok(f"- Discriminator: type={dcfg.get('type')}, weights={dcfg.get('weights', {})}")

        # Eval losses
        if len(self.eval_losses) > 0:
            for name, mod in self.eval_losses.items():
                ok(f"- Eval {name}: {type(mod).__name__}")
        ok("===== End loss summary =====")