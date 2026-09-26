# =============================================================================
# Core autoencoder training logic, computing forwards passes and aggregating losses.
# =============================================================================
import contextlib
import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any, Tuple

from sage.model.autoencoder import SAGEAutoencoder
from sage.utils.console import ok, warn, err
from sage.utils.audio import trim_to_shortest
from sage.utils.tensors import align_freq_bins, align_time_frames

from sage.training.loss_manager import LossManager


def select_training_phase(use_disc: bool, warmup_mode: str, disc_phase: bool, warmed_up: bool) -> str:
    """Decide whether this batch is a 'disc' or 'gen' step.

    ``disc_phase`` is a per-batch alternation flag owned by the training wrapper (a
    boolean toggled once per batch). It is deliberately NOT derived from
    ``global_step``: with an auxiliary optimizer present (e.g. CLAP distillation), the
    gen phase calls TWO ``optimizer.step()`` (gen + aux) so ``global_step`` advances by
    2 on gen batches and 1 on disc batches — using its parity then keeps it even and
    starves the disc phase. A per-batch toggle alternates correctly regardless of how
    many optimizers step. The disc phase is additionally gated by ``use_disc`` and the
    warmup schedule.
    """
    is_disc = bool(
        use_disc
        and disc_phase
        and ((warmup_mode == "full" and warmed_up) or warmup_mode == "adv")
    )
    return "disc" if is_disc else "gen"


class AutoencoderEngine(nn.Module):
    """
    - Costruisce discriminator, loss e metriche eval (se richieste)
    - compute(batch, global_step) -> dict con phase, loss totali e breakdown, loss_info
    - compute_validation(batch) -> dict metriche validation (CPU scalari)
    - configure_optimizers() -> ottimizzatori + scheduler dal config
    Nessun backward/step/logging qui dentro.
    """
    def __init__(self, 
                 autoencoder: SAGEAutoencoder,
                 sample_rate: int = 48000,
                 loss_config: Optional[dict] = None,
                 eval_loss_config: Optional[dict] = None,
                 warmup_steps: int = 0,
                 warmup_mode: Literal["adv", "full"] = "adv",
                 encoder_freeze_on_warmup: bool = False,
                 freeze_encoder: bool = False,
                 force_input_mono: bool = False,
                 latent_mask_ratio: float = 0.0,
                 teacher_model: Optional[SAGEAutoencoder] = None,
                 audio_channels: Optional[int] = None,
                 stft_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.autoencoder = autoencoder
        self.teacher_model = teacher_model
        self.sample_rate = sample_rate
        self.stft_params = stft_params or {}   # training params
        if not self.stft_params:
            raise ValueError("stft_params must be provided to AutoencoderEngine for audio reconstruction")

        self.autoencoder.set_stft_config(self.stft_params)
        if self.teacher_model is not None and hasattr(self.teacher_model, "set_stft_config"):
            try:
                self.teacher_model.set_stft_config(self.stft_params)
            except Exception as exc:
                warn(f"Failed to propagate STFT config to teacher model ({type(exc).__name__}: {exc})", prefix="MODEL")

        # training policy
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup

        # freeze_encoder: always-on (decoupled from warmup). Decoder-finetune mode —
        # freezes the encoder so only the decoder trains. The VAE bottleneck of the
        # real model is parameter-free (μ/logvar split + reparam), so freezing the
        # encoder freezes everything up to the latent.
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            self.autoencoder.encoder.requires_grad_(False)
            n = sum(p.numel() for p in self.autoencoder.encoder.parameters())
            ok(f"freeze_encoder=True → encoder frozen ({n/1e6:.2f}M params, requires_grad=False); "
               "only the decoder will train.", prefix="MODEL")

        self.force_input_mono = force_input_mono
        self.latent_mask_ratio = latent_mask_ratio

        # Numero di canali del segnale audio (mono/stereo)
        if audio_channels is None:
            audio_channels = 2  # default conservative
        self.audio_channels = int(audio_channels)

        # usa il loss_config passato dal chiamante
        if loss_config is None:
            loss_config = {}
        if not isinstance(loss_config, dict):
            warn("AutoencoderEngine: provided loss_config is not a dict — treating as empty config", prefix="MODEL")
            loss_config = {}
        self.loss_config = loss_config
        self.loss_manager = LossManager(
            autoencoder=self.autoencoder,
            sample_rate=self.sample_rate,
            loss_config=self.loss_config,
            eval_loss_config=eval_loss_config,
            audio_channels=self.audio_channels
        )
        self.use_disc = self.loss_manager.use_disc

    @torch.no_grad()
    def _encode_teacher_if_needed(self, encoder_input):
        if self.teacher_model is None:
            return None
        return self.teacher_model.encode(encoder_input, return_info=False)

    def compute(self, batch: Tuple[torch.Tensor, torch.Tensor], global_step: int,
                disc_phase: Optional[bool] = None) -> Dict[str, Any]:
        """Compute forward and loss breakdown for a training batch.

        The method orchestrates the end-to-end path ``spectrogram -> encoder ->
        bottleneck -> decoder`` and prepares the tensors required by every
        generator/discriminator loss.

        Pre-transform handling follows the configuration stored on
        ``self.autoencoder``:

        * ``apply_encoder``: encoder inputs are normalized before being fed to
            the network.
        * ``apply_target``: when enabled, the same normalized representation is
            used as loss target (``loss_info["encoder_input"]``), ensuring
            spectrogram losses compare tensors in the transformed domain.
        * ``apply_inverse``: regardless of this flag, waveform-domain losses
            always receive inverse-transformed spectrograms through
            ``loss_info["decoded"]`` so that audio reconstruction happens in the
            linear domain.

        Args:
                batch: Tuple ``(sp_reals, orig_waveforms)`` containing the reference
                        spectrograms and waveforms produced by the dataset.
                global_step: Current optimization step, used to control warm-up and
                        adversarial phase alternation.

        Returns:
                Dict[str, Any]: A payload that includes the selected training phase,
                scalar losses, detailed loss breakdowns, cached tensors required by
                the loss modules, and auxiliary statistics (``data_std`` and
                ``latent_std``).
        """
        sp_reals, orig_waveforms = batch
        loss_info: Dict[str, Any] = {}

        encoder_input = sp_reals
        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        # we apply pre-transform if needed
        spectral_target = encoder_input
        if self.autoencoder.pre_transform_applies_to_target:
            spectral_target = self.autoencoder.apply_pre_transform_to_target(spectral_target)

        # we store transformed target and original waveforms
        loss_info["encoder_input"] = spectral_target
        loss_info["reals"] = orig_waveforms
        loss_info["global_step"] = global_step  # consumed by latent-alignment losses for detached-warmup

        warmed_up = (global_step >= self.warmup_steps)

        # Resolve the phase up-front: gen losses only on gen steps, disc loss only on
        # disc steps, and the generator forward under no_grad on disc steps.
        if disc_phase is None:
            disc_phase = (global_step % 2 == 1)
        phase = select_training_phase(self.use_disc, self.warmup_mode, bool(disc_phase), warmed_up)
        disc_step = (phase == "disc")

        gen_fwd_ctx = torch.no_grad() if disc_step else contextlib.nullcontext()
        enc_ctx = (
            torch.no_grad()
            if (disc_step or (warmed_up and self.encoder_freeze_on_warmup))
            else contextlib.nullcontext()
        )

        with gen_fwd_ctx:
            # Encode
            with enc_ctx:
                enc_out = self.autoencoder.encode(encoder_input, return_info=True)

            bottleneck_info: Dict[str, Any] = {}
            
            if isinstance(enc_out, tuple) and len(enc_out) == 3:
                latents, encoder_info, bottleneck_info = enc_out
            elif isinstance(enc_out, tuple):
                latents, encoder_info = enc_out
            else:
                latents, encoder_info = enc_out, {}

            loss_info["latents"] = latents
            loss_info.update(encoder_info)
            loss_info.update(bottleneck_info)

            # Optional latent masking (denoising-VAE augmentation), before decode.
            if self.latent_mask_ratio > 0.0:
                mask = torch.rand_like(latents) < self.latent_mask_ratio
                latents = torch.where(mask, torch.zeros_like(latents), latents)
                loss_info["latents"] = latents

            sp_decoded = self.autoencoder.decode(latents, encoder_info=encoder_info, apply_inverse=False)

            # if has pre-transform, invert it otherwise use decoded as is (linear)
            sp_decoded_linear = (
                self.autoencoder.apply_inverse_pre_transform(sp_decoded)
                if self.autoencoder.has_pre_transform
                else sp_decoded
            )

            # select spectrogram for losses
            if self.autoencoder.pre_transform_applies_to_target:
                sp_decoded_for_losses = sp_decoded
            elif self.autoencoder.pre_transform_applies_inverse:
                sp_decoded_for_losses = sp_decoded_linear
            else:
                sp_decoded_for_losses = sp_decoded

            # Align freq + time for the spectral losses
            try:
                sp_decoded_aligned = align_freq_bins(sp_decoded_for_losses, spectral_target)
                sp_decoded_aligned = align_time_frames(sp_decoded_aligned, spectral_target)
            except Exception as e:
                err(f"Failed to align spectrogram F dimension ({e}).", prefix="MODEL")
                sp_decoded_aligned = sp_decoded_for_losses

            # Align freq + time for waveform reconstruction
            try:
                sp_decoded_linear_aligned = align_freq_bins(sp_decoded_linear, encoder_input)
                sp_decoded_linear_aligned = align_time_frames(sp_decoded_linear_aligned, encoder_input)
            except Exception as e:
                err(f"Failed to align spectrogram for waveform losses ({e}).", prefix="MODEL")
                sp_decoded_linear_aligned = sp_decoded_linear

            decoded = self.autoencoder.istft(sp_decoded_linear_aligned, target_length=orig_waveforms.shape[-1])
            # align to the real waveforms (do not use the inverse of the input)
            decoded, orig_waveforms = trim_to_shortest(decoded, orig_waveforms)

        loss_info["decoded"] = decoded              # waveform pred
        loss_info["reals"] = orig_waveforms         # waveform GT
        loss_info["sp_decoded"] = sp_decoded_aligned  # use aligned tensor for losses
        loss_info["sp_decoded_linear"] = sp_decoded_linear_aligned # linear spectra for waveform losses

        # decoded/reals expected shape: (B, C_audio, N)
        if self.audio_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = orig_waveforms[:, 0:1, :]
            loss_info["reals_right"] = orig_waveforms[:, 1:2, :]

        # Teacher (SAO-style AE latent distillation) -- gen step only.
        teacher_latents = None
        if not disc_step:
            teacher_latents = self._encode_teacher_if_needed(encoder_input)
        if teacher_latents is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents)
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents)
            loss_info['teacher_latents'] = teacher_latents
            loss_info['teacher_decoded'] = teacher_decoded
            loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
            loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        gen_total = None
        gen_breakdown: Dict[str, torch.Tensor] = {}
        disc_total = None
        disc_breakdown: Dict[str, torch.Tensor] = {}

        if disc_step:
            # decoded is detached (no_grad forward) → disc.loss() updates only the disc
            loss_dis, _, _ = self.loss_manager.discriminator.loss(reals=orig_waveforms, fakes=decoded)
            loss_info["loss_dis"] = loss_dis
            disc_total, disc_breakdown = self.loss_manager.losses_disc(loss_info)
        else:
            if self.use_disc:
                if warmed_up:
                    _, loss_adv, feat_match = self.loss_manager.discriminator.loss(reals=orig_waveforms, fakes=decoded)
                else:
                    loss_adv = torch.zeros((), device=decoded.device)
                    feat_match = torch.zeros((), device=decoded.device)
                loss_info["loss_adv"] = loss_adv
                loss_info["feature_matching_distance"] = feat_match

            gen_total, gen_breakdown = self.loss_manager.losses_gen(loss_info)

        # Stats per logging
        data_std = loss_info["encoder_input"].std()
        latent_std = loss_info["latents"].std()
        stats = {"data_std": data_std, "latent_std": latent_std}

        return {
            "phase": phase,
            "gen_total": gen_total,
            "gen_breakdown": gen_breakdown,
            "disc_total": disc_total,
            "disc_breakdown": disc_breakdown,
            "loss_info": loss_info,
            "stats": stats,
        }

    @torch.no_grad()
    def compute_validation(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        batch: (sp_reals, orig_waveforms)
        Same logic as compute(), but only for eval losses and no grad.
        """
        sp_reals, orig_waveforms = batch
        encoder_input = sp_reals
        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        # Feed the RAW spectrogram to encode(). SAGEAutoencoder.encode applies the
        # pre-transform internally (apply_encoder=True), so pre-transforming here
        # would apply it TWICE → the encoder would see an out-of-distribution input
        # and every val/* metric would be silently distorted. This mirrors compute(),
        # which feeds raw input to encode() and keeps the transformed tensor only as a
        # separate loss target. align_* below uses encoder_input for shape only.
        enc_out = self.autoencoder.encode(encoder_input, return_info=True)
        bottleneck_info: Dict[str, Any] = {}
        if isinstance(enc_out, tuple) and len(enc_out) == 3:
            latents, _enc_info, bottleneck_info = enc_out
        elif isinstance(enc_out, tuple):
            latents, _enc_info = enc_out
        else:
            latents = enc_out
        sp_decoded = self.autoencoder.decode(latents, apply_inverse=False)
        sp_decoded_linear = (
            self.autoencoder.apply_inverse_pre_transform(sp_decoded)
            if self.autoencoder.has_pre_transform
            else sp_decoded
        )

        try:
            sp_decoded_linear = align_freq_bins(sp_decoded_linear, encoder_input)
            sp_decoded_linear = align_time_frames(sp_decoded_linear, encoder_input)
        except Exception as exc:
            err(f"Validation spectrogram alignment failed ({type(exc).__name__}: {exc}).", prefix="MODEL")

        decoded = self.autoencoder.istft(sp_decoded_linear, target_length=orig_waveforms.shape[-1])
        decoded, orig_waveforms = trim_to_shortest(decoded, orig_waveforms)

        val_loss_dict: Dict[str, float] = {}
        for eval_key, eval_fn in self.loss_manager.eval_losses.items():
            value = eval_fn(decoded, orig_waveforms)
            if eval_key == "sisdr":
                value = -value
            if isinstance(value, torch.Tensor):
                value = value.item()
            val_loss_dict[eval_key] = value
        if "kl" in bottleneck_info:
            kl_val = bottleneck_info["kl"]
            if isinstance(kl_val, torch.Tensor):
                kl_val = kl_val.item()
            val_loss_dict["kl"] = kl_val
        return val_loss_dict

