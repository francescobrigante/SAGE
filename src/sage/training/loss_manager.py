# =============================================================================
# Builds the generator loss of the paper (eq. 2), the discriminator and the
# validation metrics from `trainer.loss_config` / `trainer.eval_loss_config`.
#
#   L_G = λ_STFT·L_STFT + λ_mel·L_mel + λ_SD·L_SD + λ_KL·L_KL + λ_sem·L_sem
#         + λ_adv·L_adv + λ_fm·L_fm
#
# Only these seven terms are recognised in loss_config. Losses explored during
# development live in sage.nn.losses.experimental and are added to a run through
# `loss_config.extra` (appended after the paper terms).
# =============================================================================
from typing import Optional

import torch.nn as nn
from hydra.utils import instantiate

from sage.nn.bottleneck import VAEBottleneck
from sage.nn.discriminators import (
    EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator,
    MultiTransformerDiscriminator, HILDiscriminator, WavTokenizerGANLoss,
)
from sage.nn.losses import signal
from sage.nn.losses.base import MultiLoss, ValueLoss, LossWithTarget, SelfLoss
from sage.nn.losses.perceptual import MelSpectrogramLoss
from sage.nn.losses.semantic import CLAPTeacher, LatentCosineDistillLoss
from sage.nn.losses.spectral import ComplexMSE
from sage.utils.console import ok

# Keys accepted in loss_config, one block per term of eq. 2 (plus the extra list).
# None = any value (e.g. constructor kwargs, validated by the loss class itself).
LOSS_SCHEMA = {
    "spectral": {"stft_mse": {"config": None}, "weights": {"stft_mse": None}},               # L_STFT
    "mrmel": {"weights": {"mrmel": None},                                                    # L_mel
              "config": {"n_mels": None, "window_lengths": None, "pow": None, "log_weight": None, "mag_weight": None}},
    "mrstft_sd": {"weights": {"mrstft_sd": None}, "config": None},                           # L_SD
    "bottleneck": {"weights": {"kl": None}},                                                 # L_KL
    "semantic_distill": {"weights": {"distill": None}, "config": None,                       # L_sem
                         "teacher_type": None, "detach_warmup_steps": None},
    "discriminator": {"type": None, "config": None,                                          # L_adv, L_fm
                      "weights": {"adversarial": None, "feature_matching": None}},
    "extra": None,                                                                           # experimental losses
}
_EXTRA_KEYS = {"name", "weight", "loss", "input_key", "target_key", "decay"}


def _unknown_keys(cfg, schema, path=""):
    if schema is None or not isinstance(cfg, dict):
        return []
    unknown = []
    for key, value in cfg.items():
        where = f"{path}.{key}" if path else str(key)
        unknown += [where] if key not in schema else _unknown_keys(value, schema[key], where)
    return unknown


def _check_loss_config(loss_config: dict) -> None:
    """Reject every key the manager would otherwise ignore silently: typos, and the
    pre-release blocks and options (e.g. `mrstft_same`, `spectral.mrstft`, `decay`)."""
    unknown = _unknown_keys(loss_config, LOSS_SCHEMA)
    if unknown:
        raise ValueError(
            f"loss_config: unknown entries {sorted(unknown)}. Only the terms of the paper are "
            f"built from their own block ({', '.join(k for k in LOSS_SCHEMA if k != 'extra')}); add other "
            "losses through loss_config.extra (see sage.nn.losses.experimental).")


def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    """The KL term of a VAE bottleneck. It is built even at weight 0 (decoder fine-tuning):
    the paper checkpoint has it in its loss layout (losses_gen.losses.5.weight)."""
    if not isinstance(bottleneck, VAEBottleneck):
        return []
    kl_weight = ((loss_config.get("bottleneck") or {}).get("weights") or {}).get("kl")
    if kl_weight is None:
        raise ValueError("loss_config.bottleneck.weights.kl is required with a VAE bottleneck (0 disables it)")
    return [ValueLoss(key='kl', weight=kl_weight, name='kl_loss')]


class LossManager(nn.Module):
    """Generator losses, discriminator losses and evaluation metrics of a training run.

    A term is built only when its weight is > 0 (except KL, see
    create_loss_modules_from_bottleneck). The order of the generator terms is
    fixed (adv, fm, STFT, mel, SD, sem, KL, then the extra losses) because their weights
    are saved in the checkpoint by position (``losses_gen.losses.<i>.weight``).
    """
    def __init__(self,
                 autoencoder,
                 sample_rate: int = 48000,
                 loss_config: Optional[dict] = None,
                 eval_loss_config: Optional[dict] = None,
                 audio_channels: int = 2):
        super().__init__()
        # the autoencoder is not stored: Lightning would count its parameters twice
        self.sample_rate = sample_rate
        self.audio_channels = int(audio_channels)
        if loss_config is None:
            loss_config = {}
        if not isinstance(loss_config, dict):
            raise TypeError(f"loss_config must be a dict, got {type(loss_config).__name__}")
        _check_loss_config(loss_config)
        self.loss_config = loss_config
        self.use_disc = bool(loss_config.get("discriminator"))

        # The discriminator is built before the distillation head: both draw from the
        # RNG, and this order keeps seeded runs identical to the paper code.
        self.discriminator = self._build_discriminator(loss_config["discriminator"]) if self.use_disc else None

        gen_loss_modules = []

        # L_adv, L_fm: computed by the discriminator in the engine, weighted here
        if self.use_disc:
            disc_w = loss_config["discriminator"]["weights"]
            gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=disc_w['adversarial'], name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=disc_w['feature_matching'], name='feature_matching_loss'),
            ]

        # L_STFT: squared error on the power-compressed complex STFT (the model's own domain)
        spectral_cfg = loss_config.get("spectral") or {}
        self.stft_mse = None
        stft_weight = (spectral_cfg.get("weights") or {}).get("stft_mse", 0.0)
        if stft_weight > 0.0:
            self.stft_mse = ComplexMSE(**((spectral_cfg.get("stft_mse") or {}).get("config") or {}))
            gen_loss_modules.append(LossWithTarget(
                self.stft_mse, input_key='sp_decoded', target_key='encoder_input',
                name='pwc_mse_loss', weight=stft_weight))

        # L_mel: multi-resolution mel loss on the waveform
        self.mrmel = None
        if loss_config.get("mrmel") and loss_config["mrmel"]["weights"]["mrmel"] > 0.0:
            mel_cfg = loss_config["mrmel"]["config"]
            self.mrmel = MelSpectrogramLoss(
                self.sample_rate,
                n_mels=mel_cfg["n_mels"],
                window_lengths=mel_cfg["window_lengths"],
                pow=mel_cfg["pow"],
                log_weight=mel_cfg["log_weight"],
                mag_weight=mel_cfg["mag_weight"],
            )
            gen_loss_modules.append(LossWithTarget(       # (reals, decoded): order of the paper code, kept
                self.mrmel, input_key="reals", target_key="decoded",
                name="mrmel_loss", weight=loss_config["mrmel"]["weights"]["mrmel"]))

        # L_SD: sum-and-difference MR-STFT on M, S, L, R (A-weighted)
        self.mrstft_sd = None
        if loss_config.get("mrstft_sd") and loss_config["mrstft_sd"]["weights"]["mrstft_sd"] > 0.0:
            sd_cfg = dict(loss_config["mrstft_sd"].get("config") or {})
            sd_cfg.setdefault("sample_rate", self.sample_rate)
            self.mrstft_sd = signal.SumAndDifferenceSTFTLoss(**sd_cfg)
            gen_loss_modules.append(LossWithTarget(
                self.mrstft_sd, input_key="decoded", target_key="reals",
                name="mrstft_sd_loss", weight=loss_config["mrstft_sd"]["weights"]["mrstft_sd"]))

        # L_sem: clip-level cosine distillation of the latent onto LAION-CLAP
        sem_cfg = loss_config.get("semantic_distill")
        if sem_cfg and sem_cfg["weights"]["distill"] > 0.0:
            import config as _root_config
            distill_cfg = dict(sem_cfg.get("config") or {})
            warmup = sem_cfg.get("detach_warmup_steps", 8334)   # generator updates (paper s0)
            latent_dim = int(distill_cfg.pop("latent_dim"))
            proj_dim = int(distill_cfg.pop("proj_dim", 512))
            teacher_type = str(sem_cfg.get("teacher_type", "clap")).lower()
            if teacher_type != "clap":
                raise ValueError(f"semantic_distill.teacher_type={teacher_type!r}: only 'clap' is supported")
            self.distill_proj = nn.Linear(latent_dim, proj_dim)   # → aux_parameters() → opt_aux
            self.clap_teacher = CLAPTeacher(
                str(_root_config.MODELS_DIR / "LAION_CLAP" / "music_audioset_epoch_15_esc_90.14.pt"),
                src_sr=self.sample_rate
            )
            gen_loss_modules.append(LatentCosineDistillLoss(      # (B,512) global CLAP embedding
                self.distill_proj, self.clap_teacher, weight=sem_cfg["weights"]["distill"],
                detach_warmup_steps=warmup, **distill_cfg))

        # L_KL: from the VAE bottleneck
        if getattr(autoencoder, "bottleneck", None) is not None:
            gen_loss_modules += create_loss_modules_from_bottleneck(autoencoder.bottleneck, loss_config)

        # experimental losses (sage.nn.losses.experimental), after every paper term
        gen_loss_modules += self._build_extra(loss_config.get("extra") or [], gen_loss_modules)

        self.losses_gen = MultiLoss(gen_loss_modules)

        # Learnable loss submodules (the CLAP distillation head). Collected by
        # aux_parameters() so they land in opt_aux — opt_gen only sees the autoencoder,
        # so without this the head would never be updated.
        self._aux_module_names = ("distill_proj",)

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

        self._log_losses_summary()

    def _build_discriminator(self, cfg: dict) -> nn.Module:
        disc_type, disc_cfg = cfg["type"], cfg["config"]
        ch, sr = self.audio_channels, self.sample_rate
        if disc_type == 'oobleck':
            disc_cfg = dict(disc_cfg)
            return OobleckDiscriminator(in_channels=disc_cfg.pop('in_channels', ch), **disc_cfg)
        if disc_type == 'encodec':
            return EncodecDiscriminator(in_channels=ch, **disc_cfg)
        if disc_type == 'dac':
            return DACGANLoss(channels=ch, sample_rate=sr, **disc_cfg)
        if disc_type == 'big_vgan':
            return BigVGANDiscriminator(channels=ch, sample_rate=sr, **disc_cfg)
        if disc_type == 'transformer':
            return MultiTransformerDiscriminator(in_channels=ch, **disc_cfg)
        if disc_type == 'hil':
            return HILDiscriminator(in_channels=ch, sample_rate=sr, **disc_cfg)
        if disc_type == 'wavtokenizer':   # the paper's discriminator
            return WavTokenizerGANLoss(channels=ch, sample_rate=sr, **disc_cfg)
        raise ValueError(f"unknown discriminator type {disc_type!r}")

    @staticmethod
    def _build_extra(entries: list, existing: list) -> list:
        """Wrap each ``loss_config.extra`` entry (see sage.nn.losses.experimental)."""
        names = {m.name for m in existing}
        modules = []
        for entry in entries:
            unknown = set(entry) - _EXTRA_KEYS
            missing = {"name", "weight", "loss", "input_key"} - set(entry)
            if unknown or missing:
                raise ValueError(f"loss_config.extra entry {entry.get('name')!r}: "
                                 f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
            if entry["name"] in names:
                raise ValueError(f"loss_config.extra: duplicate loss name {entry['name']!r}")
            names.add(entry["name"])
            if float(entry["weight"]) <= 0.0:
                continue
            loss = instantiate(entry["loss"], _convert_="all")
            common = dict(name=entry["name"], weight=float(entry["weight"]), decay=float(entry.get("decay", 1.0)))
            if entry.get("target_key") is None:
                modules.append(SelfLoss(loss, input_key=entry["input_key"], **common))
            else:
                modules.append(LossWithTarget(loss, input_key=entry["input_key"],
                                              target_key=entry["target_key"], **common))
        return modules

    def get_kl_loss_module(self):
        """Return the KL ValueLoss module, or None if not present."""
        for loss_module in self.losses_gen.losses:
            if getattr(loss_module, 'name', '') == 'kl_loss':
                return loss_module
        return None

    def aux_parameters(self) -> list:
        """Learnable params of the loss submodules (CLAP distillation head), for the aux optimizer.

        opt_gen is built from the autoencoder only, so these modules need their own
        optimizer. Returns [] when distillation is off (opt_aux is then not created).
        """
        params = []
        for name in self._aux_module_names:
            module = getattr(self, name, None)
            if module is not None:
                params += [p for p in module.parameters() if p.requires_grad]
        return params

    def _log_losses_summary(self):
        for m in self.losses_gen.losses:
            inner = getattr(m, "loss_module", None)
            what = f" ({type(inner).__name__})" if inner is not None else ""
            ok(f"{m.name}{what}: weight {float(m.weight):g}", prefix="LOSS")
        if self.use_disc:
            ok(f"discriminator: {type(self.discriminator).__name__}", prefix="LOSS")
        if len(self.eval_losses) > 0:
            ok(f"eval: {', '.join(f'{k} ({type(v).__name__})' for k, v in self.eval_losses.items())}", prefix="LOSS")
