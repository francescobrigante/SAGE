# =============================================================================
# Manages parsing, instantiating, and aggregating generic loss functions and discriminators.
# =============================================================================
import torch.nn as nn
from typing import Optional

from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator
from ..models.bottlenecks import VAEBottleneck
from .losses.base import MultiLoss, ValueLoss, L1Loss, LossWithTarget, MSELoss
from .losses.perceptual import MelSpectrogramLoss, HubertLoss
from .losses import signal
from .losses.spectral import MultiResSpectralConvergence, ComplexMSE, MultiResolutionSpectrogramLoss, PhaseCosineDistance
from ar_spectra.utils.console import ok, warn

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []
    if isinstance(bottleneck, VAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except (KeyError, TypeError):
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)
    return losses


class LossManager(nn.Module):
    """
    Parses configuration and instantiates generator losses, discriminator losses,
    and evaluation metrics. Encapsulates the monolithic initialization logic previously
    stored in the training engine.
    """
    def __init__(self, 
                 autoencoder,
                 sample_rate: int = 48000,
                 loss_config: Optional[dict] = None,
                 eval_loss_config: Optional[dict] = None,
                 audio_channels: int = 2):
        super().__init__()
        # self.autoencoder = autoencoder -> Removed to avoid PL duplicating params in ModelSummary
        self.sample_rate = sample_rate
        self.audio_channels = int(audio_channels)

        if loss_config is None:
            loss_config = {}
        if not isinstance(loss_config, dict):
            warn("LossManager: provided loss_config is not a dict — treating as empty config")
            loss_config = {}
        self.loss_config = loss_config

        self.use_disc = bool(self.loss_config.get("discriminator"))

        # reconstruction losses
        spectral_cfg = self.loss_config.get("spectral")
        self.apply_pre_transform_to_wave_losses = False
        if spectral_cfg and isinstance(spectral_cfg, dict):
            self.apply_pre_transform_to_wave_losses = bool(spectral_cfg.get("apply_pre_transform_to_wave_losses", False))
        if self.apply_pre_transform_to_wave_losses and not getattr(autoencoder, "has_pre_transform", False):
            warn(
                "apply_pre_transform_to_wave_losses requested but the autoencoder has no pre_transform; disabling this option."
            )
            self.apply_pre_transform_to_wave_losses = False

        if spectral_cfg and isinstance(spectral_cfg, dict):
            stft_mse_block = spectral_cfg.get("stft_mse", {}) or {}
            configs_stft_mse = stft_mse_block.get("config", {}) or {}
            self.stft_mse = ComplexMSE(**configs_stft_mse)
        else:
            self.stft_mse = None

        self.mrstft = None
        self.phase_loss = None

        if spectral_cfg and isinstance(spectral_cfg, dict):
            mr_keys = [k for k in ("mrstft_stable_audio", "mrstft", "mrstft_sc",) if k in spectral_cfg]
            phase_keys = ["cosine_phase_loss"] if "cosine_phase_loss" in spectral_cfg else []
            assert len(mr_keys) <= 1, (
                "loss_config.spectral: you need to specify at most one of the keys:"
                "'mrstft_stable_audio', 'mrstft', 'mrstft_sc'. "
                f"Found: {mr_keys}"
            )
            if len(mr_keys) > 0:
                chosen = mr_keys[0]
                mrstft_block = spectral_cfg.get(chosen, {}) or {}
                configs_mrstft = mrstft_block.get("config", mrstft_block) or {}
                if not isinstance(configs_mrstft, dict):
                    raise TypeError(f"Expected dict for spectral.{chosen}.config, got {type(configs_mrstft).__name__}")
                configs_mrstft = dict(configs_mrstft)
                
                if chosen == "mrstft_stable_audio":
                    if self.apply_pre_transform_to_wave_losses:
                        warn("apply_pre_transform_to_wave_losses is not supported with 'mrstft_stable_audio'.")
                    self.mrstft = signal.MultiResolutionSTFTLoss(**configs_mrstft)
                elif chosen == "mrstft_sc":
                    extra_kwargs = {}
                    if self.apply_pre_transform_to_wave_losses:
                        extra_kwargs = {"apply_pre_transform": True, "pre_transform": autoencoder.pre_transform}
                    self.mrstft = MultiResSpectralConvergence(**configs_mrstft, **extra_kwargs)
                elif chosen == "mrstft":
                    extra_kwargs = {}
                    if self.apply_pre_transform_to_wave_losses:
                        extra_kwargs = {"apply_pre_transform": True, "pre_transform": autoencoder.pre_transform}
                    self.mrstft = MultiResolutionSpectrogramLoss(**configs_mrstft, **extra_kwargs)

            if len(phase_keys) > 0:
                phase_chosen = phase_keys[0]
                phase_block = spectral_cfg.get(phase_chosen, {}) or {}
                configs_phase = phase_block.get("config", phase_block) or {}
                self.phase_loss = PhaseCosineDistance(**configs_phase)

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

        gen_loss_modules = []
        if self.use_disc:
            gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss'),
            ]

        stft_loss_decay = spectral_cfg.get('decay', 1.0) if spectral_cfg else 1.0
        if spectral_cfg:
            if self.stft_mse is not None:
                stft_mse_weight = spectral_cfg['weights'].get('stft_mse', 0.0)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.stft_mse,
                        input_key='sp_decoded', target_key='encoder_input',
                        name='pwc_mse_loss', weight=stft_mse_weight, decay=stft_loss_decay,
                    )
                )
            if self.mrstft is not None:
                stft_mse_weight = spectral_cfg['weights'].get('mrstft', 0.0)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.mrstft,
                        target_key='reals', input_key='decoded',
                        name='mrstft_loss', weight=stft_mse_weight, decay=stft_loss_decay
                    )
                )
            if self.phase_loss is not None:
                phase_weight = spectral_cfg['weights'].get('cosine_phase_loss', 0.0)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.phase_loss,
                        target_key='encoder_input', input_key='sp_decoded',
                        name='phase_cosine_loss', weight=phase_weight, decay=stft_loss_decay   
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
                gen_loss_modules.append(
                    L1Loss(key_a='decoded', key_b='reals', weight=self.loss_config["time"]["weights"]["l1"], name='l1_time_loss', decay=self.loss_config["time"].get('decay', 1.0))
                )
            if self.loss_config["time"]["weights"].get("l2", 0.0) > 0.0:
                gen_loss_modules.append(
                    MSELoss(key_a='decoded', key_b='reals', weight=self.loss_config["time"]["weights"]["l2"], name='l2_time_loss', decay=self.loss_config["time"].get('decay', 1.0))
                )

        if getattr(autoencoder, "bottleneck", None) is not None:
            gen_loss_modules += create_loss_modules_from_bottleneck(autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(gen_loss_modules)

        self.losses_disc = None
        if self.use_disc:
            self.losses_disc = MultiLoss([ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss')])

        self.eval_losses = nn.ModuleDict()
        if eval_loss_config is not None:
            if "stft" in eval_loss_config:
                self.eval_losses["stft"] = signal.STFTLoss(**eval_loss_config["stft"])
            if "sisdr" in eval_loss_config:
                self.eval_losses["sisdr"] = signal.SISDRLoss(**eval_loss_config["sisdr"])
            if "mel" in eval_loss_config:
                self.eval_losses["mel"] = signal.MelSTFTLoss(self.sample_rate, **eval_loss_config["mel"])
                
        ok("initialized with train losses: " +
           f"gen: {[type(l).__name__ for l in gen_loss_modules]}, " +
           (f"disc: {[type(l).__name__ for l in self.losses_disc.modules()]}" if self.use_disc else "no disc"), prefix="LOSS MANAGER")
        ok(f"initialized with eval losses: {list(self.eval_losses.keys())}", prefix="LOSS MANAGER")
        self._log_losses_summary()

    def _extract_hparams(self, module: nn.Module) -> dict:
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
        spectral_cfg = self.loss_config.get("spectral", {}) or {}
        if self.stft_mse is not None:
            ok(f"PerceptuallyWeightedComplexMSE: {self._extract_hparams(self.stft_mse)}", prefix="LOSS")
        if self.mrstft is not None:
            try:
                params = self._extract_hparams(self.mrstft)
            except Exception:
                params = {}
                warn(f"Failed to extract hparams from {type(self.mrstft).__name__}", prefix="LOSS")
            ok(f"{type(self.mrstft).__name__}: {params}", prefix="LOSS")
            if self.apply_pre_transform_to_wave_losses:
                ok("pre_transform applied to waveform spectral losses", prefix="LOSS")
        if self.phase_loss is not None:
            try:
                params = self._extract_hparams(self.phase_loss)
            except Exception:
                params = {}
                warn(f"Failed to extract hparams from {type(self.phase_loss).__name__}", prefix="LOSS")
            ok(f"{type(self.phase_loss).__name__}: {params}", prefix="LOSS")
        if spectral_cfg:
            ok(f"Spectral weights: {spectral_cfg.get('weights', {})}", prefix="LOSS")

        if "time" in self.loss_config:
            ok(f"Time-domain weights: {self.loss_config['time'].get('weights', {})}", prefix="LOSS")
        if "mrmel" in self.loss_config:
            ok(f"MR-Mel weight: {self.loss_config['mrmel'].get('weights', {})}", prefix="LOSS")
        if "hubert" in self.loss_config:
            ok(f"HuBERT weight/decay: {self.loss_config['hubert'].get('weights', {})}, decay={self.loss_config['hubert'].get('decay', 1.0)}", prefix="LOSS")

        if self.use_disc:
            dcfg = self.loss_config.get("discriminator", {})
            ok(f"Discriminator: type={dcfg.get('type')}, weights={dcfg.get('weights', {})}", prefix="LOSS")

        if len(self.eval_losses) > 0:
            for name, mod in self.eval_losses.items():
                ok(f"Eval {name}: {type(mod).__name__}", prefix="LOSS")
