# =============================================================================
# Manages parsing, instantiating, and aggregating generic loss functions and discriminators.
# =============================================================================
import torch.nn as nn
from typing import Optional

from ..models.discriminators import (
    EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator,
    MultiTransformerDiscriminator, HILDiscriminator, WavTokenizerGANLoss,
)
from ..models.bottlenecks import VAEBottleneck
from .losses.base import MultiLoss, ValueLoss, L1Loss, LossWithTarget, MSELoss, SelfLoss
from .losses.perceptual import MelSpectrogramLoss, HubertLoss
from .losses.generative import LatentFlowMatchingLoss
from .losses.semantic import (
    CLAPTeacher, MERTTeacher, LatentVFLoss, LatentCosineDistillLoss,
    OctaveChromaTarget, ILDTarget, LatentChromaILDLoss,
    LatentContrastiveLoss,
)
from ..models.latent_dit import LatentDiT
from .losses import signal
from .losses.spectral import (
    MultiResSpectralConvergence,
    MultiResolutionSpectrogramLoss,
    PhaseCosineDistance,
    ComplexMSE,
    PerceptualComplexMSE,
    SideComplexMSE,
    STFTConsistencyLoss,
    InstantaneousFrequencyGroupDelayLoss,
    NormalizedComplexDistanceLoss,
    SpectralContrastLoss,
    MRSTFTSame,
)
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
            if "perceptual_mse" in spectral_cfg:
                pmse_block = spectral_cfg.get("perceptual_mse", {}) or {}
                configs_pmse = pmse_block.get("config", {}) or {}
                configs_pmse.setdefault("sample_rate", self.sample_rate)
                alpha = 1.0
                if getattr(autoencoder, "pre_transform", None) is not None:
                    alpha = getattr(autoencoder.pre_transform, "alpha", 1.0)
                configs_pmse.setdefault("power_norm_alpha", alpha)
                self.stft_mse = PerceptualComplexMSE(**configs_pmse)
                self._stft_mse_name = 'perceptual_mse_loss'
                self._stft_mse_key = 'perceptual_mse'
            elif "stft_mse" in spectral_cfg:
                stft_mse_block = spectral_cfg.get("stft_mse", {}) or {}
                configs_stft_mse = stft_mse_block.get("config", {}) or {}
                self.stft_mse = ComplexMSE(**configs_stft_mse)
                self._stft_mse_name = 'pwc_mse_loss'
                self._stft_mse_key = 'stft_mse'
            else:
                self.stft_mse = None
                self._stft_mse_key = 'stft_mse'
        else:
            self.stft_mse = None
            self._stft_mse_key = 'stft_mse'

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

        self.consistency_loss = None
        if spectral_cfg and isinstance(spectral_cfg, dict) and "stft_consistency" in spectral_cfg:
            cons_block = spectral_cfg.get("stft_consistency", {}) or {}
            cons_cfg = cons_block.get("config", cons_block) or {}
            if not isinstance(cons_cfg, dict):
                raise TypeError(f"Expected dict for spectral.stft_consistency.config, got {type(cons_cfg).__name__}")
            self.consistency_loss = STFTConsistencyLoss(**cons_cfg)

        self.if_gd_loss = None
        if spectral_cfg and isinstance(spectral_cfg, dict) and "if_gd" in spectral_cfg:
            if_gd_cfg = spectral_cfg.get("if_gd", {}).get("config", {}) or {}
            if not isinstance(if_gd_cfg, dict):
                if_gd_cfg = dict(if_gd_cfg)
            self.if_gd_loss = InstantaneousFrequencyGroupDelayLoss(**if_gd_cfg)

        self.ncd_loss = None
        if spectral_cfg and isinstance(spectral_cfg, dict) and "normalized_complex_distance" in spectral_cfg:
            ncd_cfg = spectral_cfg.get("normalized_complex_distance", {}).get("config", {}) or {}
            if not isinstance(ncd_cfg, dict):
                ncd_cfg = dict(ncd_cfg)
            self.ncd_loss = NormalizedComplexDistanceLoss(**ncd_cfg)

        self.scl_loss = None
        if spectral_cfg and isinstance(spectral_cfg, dict) and "spectral_contrast" in spectral_cfg:
            scl_cfg = spectral_cfg.get("spectral_contrast", {}).get("config", {}) or {}
            if not isinstance(scl_cfg, dict):
                scl_cfg = dict(scl_cfg)
            self.scl_loss = SpectralContrastLoss(**scl_cfg)

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
            elif disc_type == 'transformer':
                self.discriminator = MultiTransformerDiscriminator(in_channels=self.audio_channels, **disc_cfg)
            elif disc_type == 'hil':
                self.discriminator = HILDiscriminator(in_channels=self.audio_channels, sample_rate=sample_rate, **disc_cfg)
            elif disc_type == 'wavtokenizer':
                self.discriminator = WavTokenizerGANLoss(channels=self.audio_channels, sample_rate=sample_rate, **disc_cfg)

        gen_loss_modules = []
        if self.use_disc:
            gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss'),
            ]

        stft_loss_decay = spectral_cfg.get('decay', 1.0) if spectral_cfg else 1.0
        if spectral_cfg:
            if self.stft_mse is not None:
                stft_mse_weight = spectral_cfg['weights'].get(self._stft_mse_key, 0.0)
                if stft_mse_weight > 0.0:
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.stft_mse,
                            input_key='sp_decoded', target_key='encoder_input',
                            name=self._stft_mse_name, weight=stft_mse_weight, decay=stft_loss_decay,
                        )
                    )
            if self.mrstft is not None:
                stft_mse_weight = spectral_cfg['weights'].get('mrstft', 0.0)
                if stft_mse_weight > 0.0:
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.mrstft,
                            target_key='reals', input_key='decoded',
                            name='mrstft_loss', weight=stft_mse_weight, decay=stft_loss_decay
                        )
                    )
            if self.phase_loss is not None:
                phase_weight = spectral_cfg['weights'].get('cosine_phase_loss', 0.0)
                if phase_weight > 0.0:
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.phase_loss,
                            target_key='encoder_input', input_key='sp_decoded',
                            name='phase_cosine_loss', weight=phase_weight, decay=stft_loss_decay
                        )
                    )
            if self.consistency_loss is not None:
                cons_weight = spectral_cfg['weights'].get('stft_consistency', 0.0)
                if cons_weight > 0.0:
                    gen_loss_modules.append(
                        SelfLoss(
                            self.consistency_loss,
                            input_key='sp_decoded_linear',  # must use linear (un-normed) spectrogram: power_norm breaks STFT consistency
                            name='stft_consistency_loss', weight=cons_weight, decay=stft_loss_decay,
                        )
                    )
            if self.if_gd_loss is not None:
                if_gd_weight = spectral_cfg['weights'].get('if_gd', 0.0)
                if if_gd_weight > 0.0:
                    in_key = 'sp_decoded' if getattr(self.if_gd_loss, 'is_complex', False) else 'decoded'
                    tgt_key = 'encoder_input' if getattr(self.if_gd_loss, 'is_complex', False) else 'reals'
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.if_gd_loss,
                            input_key=in_key, target_key=tgt_key,
                            name='if_gd_loss', weight=if_gd_weight, decay=stft_loss_decay,
                        )
                    )
            if self.ncd_loss is not None:
                ncd_weight = spectral_cfg['weights'].get('normalized_complex_distance', 0.0)
                if ncd_weight > 0.0:
                    in_key = 'sp_decoded' if getattr(self.ncd_loss, 'is_complex', False) else 'decoded'
                    tgt_key = 'encoder_input' if getattr(self.ncd_loss, 'is_complex', False) else 'reals'
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.ncd_loss,
                            input_key=in_key, target_key=tgt_key,
                            name='normalized_complex_distance_loss', weight=ncd_weight, decay=stft_loss_decay,
                        )
                    )
            if self.scl_loss is not None:
                scl_weight = spectral_cfg['weights'].get('spectral_contrast', 0.0)
                if scl_weight > 0.0:
                    in_key = 'sp_decoded'
                    tgt_key = 'encoder_input'
                    gen_loss_modules.append(
                        LossWithTarget(
                            self.scl_loss,
                            input_key=in_key, target_key=tgt_key,
                            name='spectral_contrast_loss', weight=scl_weight, decay=stft_loss_decay,
                        )
                    )


        if "mrmel" in self.loss_config:
             mrmel_weight = self.loss_config["mrmel"]["weights"]["mrmel"]
             if mrmel_weight > 0.0:
                 mrmel_config = self.loss_config["mrmel"]["config"]
                 self.mrmel = MelSpectrogramLoss(self.sample_rate,
                     n_mels=mrmel_config["n_mels"],
                     window_lengths=mrmel_config["window_lengths"],
                     pow=mrmel_config["pow"],
                     log_weight=mrmel_config["log_weight"],
                     mag_weight=mrmel_config["mag_weight"],
                 )
                 gen_loss_modules.append(LossWithTarget(self.mrmel, "reals", "decoded", name="mrmel_loss", weight=mrmel_weight))

        if "mrstft_same" in self.loss_config:
            mrstft_same_weight = self.loss_config["mrstft_same"]["weights"]["mrstft_same"]
            if mrstft_same_weight > 0.0:
                mrstft_same_cfg = self.loss_config["mrstft_same"].get("config", {}) or {}
                mrstft_same_cfg = dict(mrstft_same_cfg)
                mrstft_same_cfg.setdefault("sample_rate", self.sample_rate)
                self.mrstft_same = MRSTFTSame(**mrstft_same_cfg)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.mrstft_same, target_key="reals", input_key="decoded",
                        name="mrstft_same_loss", weight=mrstft_same_weight,
                        decay=self.loss_config["mrstft_same"].get("decay", 1.0),
                    )
                )

        if "mrstft_sd" in self.loss_config:
            mrstft_sd_weight = self.loss_config["mrstft_sd"]["weights"]["mrstft_sd"]
            if mrstft_sd_weight > 0.0:
                mrstft_sd_cfg = dict(self.loss_config["mrstft_sd"].get("config", {}) or {})
                mrstft_sd_cfg.setdefault("sample_rate", self.sample_rate)
                self.mrstft_sd = signal.SumAndDifferenceSTFTLoss(**mrstft_sd_cfg)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.mrstft_sd, target_key="reals", input_key="decoded",
                        name="mrstft_sd_loss", weight=mrstft_sd_weight,
                        decay=self.loss_config["mrstft_sd"].get("decay", 1.0),
                    )
                )

        # A — normalised complex MSE on the Side alone. Same tensors as stft_mse
        # (sp_decoded vs encoder_input), so it inherits the power-norm CAC domain
        # and the absolute-phase anchoring rather than duplicating them.
        if "mse_side" in self.loss_config:
            mse_side_weight = self.loss_config["mse_side"]["weights"]["mse_side"]
            if mse_side_weight > 0.0:
                mse_side_cfg = dict(self.loss_config["mse_side"].get("config", {}) or {})
                self.mse_side = SideComplexMSE(**mse_side_cfg)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.mse_side, input_key="sp_decoded", target_key="encoder_input",
                        name="mse_side_loss", weight=mse_side_weight,
                        decay=self.loss_config["mse_side"].get("decay", 1.0),
                    )
                )

        # L6 — inter-channel coherence, the differentiable surrogate of d_pan.
        # Registered separately from mrstft_sd so it can be enabled on its own:
        # the magnitude term and the phase term address different failures and
        # cost different things, and the ablation needs to separate them.
        if "stereo_coh" in self.loss_config:
            stereo_coh_weight = self.loss_config["stereo_coh"]["weights"]["stereo_coh"]
            if stereo_coh_weight > 0.0:
                stereo_coh_cfg = dict(self.loss_config["stereo_coh"].get("config", {}) or {})
                self.stereo_coh = signal.StereoCoherenceLoss(**stereo_coh_cfg)
                gen_loss_modules.append(
                    LossWithTarget(
                        self.stereo_coh, target_key="reals", input_key="decoded",
                        name="stereo_coh_loss", weight=stereo_coh_weight,
                        decay=self.loss_config["stereo_coh"].get("decay", 1.0),
                    )
                )

        if "hubert" in self.loss_config:
            hubert_weight = self.loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0.0:
                hubert_cfg = self.loss_config["hubert"].get("config", {})
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)
                gen_loss_modules.append(LossWithTarget(self.hubert, target_key="reals", input_key="decoded", name="hubert_loss", weight=hubert_weight, decay=self.loss_config["hubert"].get("decay", 1.0)))

        if "semantic_distill" in self.loss_config:
            distill_weight = self.loss_config["semantic_distill"]["weights"]["distill"]
            if distill_weight > 0.0:
                import config as _root_config
                distill_cfg = dict(self.loss_config["semantic_distill"].get("config", {}) or {})
                warmup = self.loss_config["semantic_distill"].get("detach_warmup_steps", 25000)
                latent_dim = int(distill_cfg.pop("latent_dim"))
                proj_dim = int(distill_cfg.pop("proj_dim", 512))
                # teacher_type selects the frozen semantic teacher (default "clap" →
                # unchanged legacy behavior; "mert" → framewise MERT-v1-95M layer 4).
                teacher_type = str(
                    self.loss_config["semantic_distill"].get("teacher_type", "clap")
                ).lower()
                self.distill_proj = nn.Linear(latent_dim, proj_dim)   # → aux_parameters() → opt_aux
                if teacher_type == "mert":
                    self.mert_teacher = MERTTeacher(
                        str(_root_config.MODELS_DIR / "MERT-v1-95M"),
                        src_sr=self.sample_rate
                    )
                    distill_teacher = self.mert_teacher               # (B,768,T) → loss time-pools
                else:
                    self.clap_teacher = CLAPTeacher(                  # attr name unchanged → resume-safe
                        str(_root_config.MODELS_DIR / "LAION_CLAP" / "music_audioset_epoch_15_esc_90.14.pt"),
                        src_sr=self.sample_rate
                    )
                    distill_teacher = self.clap_teacher              # (B,512) global embedding
                gen_loss_modules.append(
                    LatentCosineDistillLoss(
                        self.distill_proj, distill_teacher, weight=distill_weight,
                        detach_warmup_steps=warmup, **distill_cfg
                    )
                )

        if "semantic_regression" in self.loss_config:
            reg_weight = self.loss_config["semantic_regression"]["weights"].get("regression", 0.0)
            if reg_weight > 0.0:
                reg_cfg = dict(self.loss_config["semantic_regression"].get("config", {}) or {})
                warmup = self.loss_config["semantic_regression"].get("detach_warmup_steps", 25000)
                D = int(reg_cfg.get("latent_dim", 64))
                _OCTAVES = [(1, 1.0), (5, 1.5), (9, 1.0)]                # (center_octave, width)
                chroma_heads = nn.ModuleList([nn.Conv1d(D, 128, 1) for _ in _OCTAVES])
                ild_head = nn.Conv1d(D, 32, 1)
                chroma_targets = nn.ModuleList([
                    OctaveChromaTarget(oct, w, sr=self.sample_rate) for oct, w in _OCTAVES
                ])
                ild_target = ILDTarget(sr=self.sample_rate)
                # chroma_ild registers learnable heads → aux_parameters() → opt_aux
                self.chroma_ild = nn.ModuleDict({
                    "chroma": chroma_heads,
                    "ild": nn.ModuleList([ild_head]),
                })
                gen_loss_modules.append(
                    LatentChromaILDLoss(
                        chroma_heads=chroma_heads,
                        ild_head=ild_head,
                        chroma_targets=chroma_targets,
                        ild_target=ild_target,
                        weight=reg_weight,
                        detach_warmup_steps=int(warmup),
                    )
                )

        if "latent_flow" in self.loss_config:
            flow_weight = self.loss_config["latent_flow"]["weights"]["flow"]
            if flow_weight > 0.0:
                flow_cfg = dict(self.loss_config["latent_flow"].get("config", {}) or {})
                warmup = self.loss_config["latent_flow"].get("detach_warmup_steps", 10000)
                scale_invariant = self.loss_config["latent_flow"].get("scale_invariant", True)
                latent_dim = flow_cfg.pop("latent_dim")          # channels of featurized latent
                # DiT registered as self.flow_dit → aux_parameters() collects it into opt_aux
                # (opt_gen only sees the autoencoder, LATENT_ALIGNMENT_PLAN.md §0.4).
                self.flow_dit = LatentDiT(latent_dim=latent_dim, **flow_cfg)
                gen_loss_modules.append(
                    LatentFlowMatchingLoss(
                        self.flow_dit, weight=flow_weight, detach_warmup_steps=warmup,
                        scale_invariant=scale_invariant,
                    )
                )

        if "contrastive" in self.loss_config:
            contr_weight = self.loss_config["contrastive"]["weights"].get("contr", 0.0)
            if contr_weight > 0.0:
                contr_cfg = dict(self.loss_config["contrastive"].get("config", {}) or {})
                warmup = self.loss_config["contrastive"].get("detach_warmup_steps", 20000)
                latent_dim = int(contr_cfg.pop("latent_dim", 64))
                proj_dim = int(contr_cfg.pop("proj_dim", 256))
                tau = float(contr_cfg.pop("tau", 0.1))
                
                # Proiettore MLP: Linear -> SiLU -> Linear
                self.contr_proj = nn.Sequential(
                    nn.Linear(latent_dim, proj_dim),
                    nn.SiLU(),
                    nn.Linear(proj_dim, proj_dim)
                )
                gen_loss_modules.append(
                    LatentContrastiveLoss(
                        self.contr_proj, tau=tau,
                        weight=contr_weight, detach_warmup_steps=warmup
                    )
                )
                self.contrastive_enabled = True

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

        # Names of learnable latent-alignment submodules (set by per-phase loss blocks,
        # LATENT_ALIGNMENT_PLAN.md Fasi 1-4/6). Collected by aux_parameters() so they
        # land in opt_aux — opt_gen only sees the autoencoder, so without this any
        # learnable loss-module weights would never be updated.
        self._aux_module_names = ("vf_proj", "distill_proj", "chroma_ild", "contr_proj", "flow_dit")

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

    def get_kl_loss_module(self):
        """Return the KL ValueLoss module, or None if not present."""
        for loss_module in self.losses_gen.losses:
            if getattr(loss_module, 'name', '') == 'kl_loss':
                return loss_module
        return None

    def aux_parameters(self) -> list:
        """Learnable params of latent-alignment submodules, for the aux optimizer.

        opt_gen is built from the autoencoder only, so these modules need their own
        optimizer (LATENT_ALIGNMENT_PLAN.md §0.4). Returns [] when no alignment block
        is active — today's behavior is then unchanged (opt_aux is not created).
        """
        params = []
        for name in self._aux_module_names:
            module = getattr(self, name, None)
            if module is not None:
                params += [p for p in module.parameters() if p.requires_grad]
        return params

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
        if self.consistency_loss is not None:
            ok(f"STFTConsistencyLoss: {self._extract_hparams(self.consistency_loss)}", prefix="LOSS")
        if getattr(self, "if_gd_loss", None) is not None:
            ok(f"InstantaneousFrequencyGroupDelayLoss: {self._extract_hparams(self.if_gd_loss)}", prefix="LOSS")
        if getattr(self, "ncd_loss", None) is not None:
            ok(f"NormalizedComplexDistanceLoss: {self._extract_hparams(self.ncd_loss)}", prefix="LOSS")
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
