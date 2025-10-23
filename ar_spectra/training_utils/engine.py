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
    def __init__(
        self,
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
        # Params STFT/istft fallback (es. n_fft, hop_length, win_length, center, normalized)
        self.stft_params = stft_params or {}

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
        self.optimizer_configs = optimizer_configs

        # Numero di canali del segnale audio (mono/stereo)
        if audio_channels is None:
            audio_channels = 2  # default conservative
        self.audio_channels = int(audio_channels)

        # default loss config
        if loss_config is None:
            warn(f"AutoencoderEngine: loss config is None, using default MRSTFT + L1")
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes, win_lengths = [], []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)
            loss_config = {
                "spectral": {
                    "type": "mrstft",
                    "config": {"fft_sizes": scales, "hop_sizes": hop_sizes, "win_lengths": win_lengths, "perceptual_weighting": True},
                    "weights": {"mrstft": 1.0}
                },
                "time": {"type": "l1", "config": {}, "weights": {"l1": 0.0}}
            }
        # ensure we have a dict (guarda contro input non validi)
        if not isinstance(loss_config, dict):
            warn("AutoencoderEngine: provided loss_config is not a dict — treating as empty config")
            loss_config = {}
        self.loss_config = loss_config
        # use discriminator only if present and truthy
        self.use_disc = bool(self.loss_config.get("discriminator"))

        # reconstruction losses: spectral può non essere presente nella config -> protegge l'accesso
        spectral_cfg = self.loss_config.get("spectral")
        if spectral_cfg and isinstance(spectral_cfg, dict):
            stft_loss_args = spectral_cfg.get("config", {}) or {}
            # usa SD-STFT solo per segnali stereo
            if self.audio_channels == 2:
                self.sdstft = auraloss.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
                self.lrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:

            self.sdstft = None
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
        # se ho spectral configurata, aggiungo i corrispondenti loss (altrimenti salto)
        if spectral_cfg and self.sdstft is not None:
            mrstft_weight = spectral_cfg.get('weights', {}).get('mrstft', 0.0)
            if self.teacher_model is not None:
                if mrstft_weight > 0:
                    stft_w = mrstft_weight * 0.25
                    gen_loss_modules += [
                        MSELoss(key_a='teacher_latents', key_b='latents', weight=stft_w, name='latent_distill_loss', decay=stft_loss_decay),
                        AuralossLoss(self.sdstft, target_key='reals', input_key='decoded', name='mrstft_loss', weight=stft_w, decay=stft_loss_decay),
                        AuralossLoss(self.sdstft, input_key='decoded', target_key='teacher_decoded', name='mrstft_loss_distill', weight=stft_w, decay=stft_loss_decay),
                        AuralossLoss(self.sdstft, target_key='reals', input_key='own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_w, decay=stft_loss_decay),
                        AuralossLoss(self.sdstft, target_key='reals', input_key='teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_w, decay=stft_loss_decay),
                    ]
            else:
                if mrstft_weight > 0:
                    gen_loss_modules.append(AuralossLoss(self.sdstft, target_key='reals', input_key='decoded', name='mrstft_loss', weight=mrstft_weight, decay=stft_loss_decay))
                    if self.audio_channels == 2 and self.lrstft is not None:
                        half_w = mrstft_weight / 2.0
                        gen_loss_modules += [
                            AuralossLoss(self.lrstft, target_key='reals_left',  input_key='decoded_left', name='stft_loss_left', weight=half_w, decay=stft_loss_decay),
                            AuralossLoss(self.lrstft, target_key='reals_right', input_key='decoded_right', name='stft_loss_right', weight=half_w, decay=stft_loss_decay),
                        ]

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

    @torch.no_grad()
    def _encode_teacher_if_needed(self, encoder_input):
        if self.teacher_model is None:
            return None
        return self.teacher_model.encode(encoder_input, return_info=False)

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
        loss_info["encoder_input"] = encoder_input
        loss_info["reals"] = orig_waveforms

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
        sp_decoded = self.autoencoder.decode(latents)
        # Prima proviamo senza params (se l'istft interna è in grado di gestire spec complessi)
        try:
            #print("the shape of sp_decoded is:", sp_decoded.shape)
            decoded = self.autoencoder.istft(sp_decoded)
        except ValueError as e:
            # fallback: prova a passare i parametri STFT presi dalla configurazione del dataset
            if self.stft_params:
                decoded = self.autoencoder.istft(sp_decoded, **self.stft_params)
            else:
                # rialza con messaggio più informativo
                raise ValueError(
                    "autoencoder.istft failed and no stft params available for fallback. "
                    "Pass 'stft_params' (containing at least 'n_fft') to AutoencoderEngine "
                    "or include them in the dataset config."
                ) from e
        decoded, orig_waveforms = trim_to_shortest(decoded, orig_waveforms)

        loss_info["decoded"] = decoded
        loss_info["reals"] = orig_waveforms

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
        decoded = self.autoencoder.istft(sp_decoded)
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