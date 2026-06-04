#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/swin_encoder.py
# MTEB-compatible encoder for Swin C-VAE checkpoints (complex and real).
# Single forward pass per clip (variable-T padded), frame-masked mean pool.
# Embedding = deterministic VAE mean μ; complex models: Re+Im concatenated.
# =============================================================================
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

# Path inject MUST happen before ar_spectra import so maeb_dl can find src/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))  # put evaluation/ on path for compatibility
from maeb import compatibility                              # noqa: E402
compatibility.add_ar_spectra_to_path(_REPO_ROOT)          # inject src/ into sys.path

from ar_spectra.models.inference import EuleroEncodeDecode  # noqa: E402

import torchaudio                                          # noqa: E402
from tqdm.auto import tqdm                                 # noqa: E402

from mteb.models.abs_encoder import AbsEncoder            # noqa: E402
from mteb.models.model_meta import ModelMeta              # noqa: E402

from .audio_prep import AudioDecodeError, prepare_audio   # noqa: E402

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)

_SWIN_SR = 44100
_SWIN_CH = 2


# ---------------------------------------------------------------------------
# Padding helper (identical to evaluate_swin_varT.py)
# ---------------------------------------------------------------------------

def _pad_for_swin(
    wav: torch.Tensor,
    hop: int,
    num_downsamples: int,
) -> tuple[torch.Tensor, int]:
    """Pad waveform so STFT frame count is a multiple of 2^num_downsamples.

    Args:
        wav:             (C, T) float32 waveform.
        hop:             STFT hop length in samples.
        num_downsamples: Number of temporal downsampling stages in the encoder.

    Returns:
        (wav_padded, orig_samples): padded waveform and original sample count.
    """
    orig = wav.shape[-1]
    w_mul = 2 ** num_downsamples
    W = (orig // hop) + 1                            # current STFT frame count
    pad_w = (w_mul - W % w_mul) % w_mul              # extra frames needed
    target = max(orig, (W + pad_w - 1) * hop)
    if target > orig:
        wav = F.pad(wav, (0, target - orig))
    return wav, orig


# ===========================================================================
# MTEB encoder wrapper
# ===========================================================================

class SwinEncoder(AbsEncoder):
    """MTEB-compatible wrapper for Swin C-VAE checkpoints.

    Encodes each audio clip with a single variable-T forward pass (padded to
    the next 2^num_downsamples STFT-frame boundary), then mean-pools the VAE
    mean μ over valid (non-padding) latent frames.  Complex models return
    Re+Im concatenated → same embed_dim as real at equal compression.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 30.0,
        standardize_bottleneck: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.max_audio_length_seconds = max_audio_length_seconds

        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        loader = EuleroEncodeDecode(model_name, device=_device)
        self.autoencoder = loader.autoencoder
        self.device: torch.device = loader.device

        self.sampling_rate: int = loader.sample_rate or _SWIN_SR
        self.audio_channels: int = loader.audio_channels or _SWIN_CH
        self.is_complex: bool = getattr(self.autoencoder.encoder, "is_complex", False)

        # STFT hop length
        stft_cfg = getattr(self.autoencoder, "_stft_config", None)
        if stft_cfg is None:
            raise RuntimeError("Checkpoint has no _stft_config — cannot compute padding.")
        self.hop: int = stft_cfg.hop_length

        # Number of temporal downsampling stages (PatchMerging layers)
        try:
            self.num_downsamples: int = len(self.autoencoder.encoder.depths) - 1
        except AttributeError:
            log.warning("Cannot read encoder.depths; defaulting num_downsamples=2.")
            self.num_downsamples = 2

        # parameters_to_predict determines how many slots in pre_bottleneck_latents
        # correspond to μ (first slot) vs σ (second) vs optional third moment.
        self.parameters_to_predict: int = self.autoencoder.bottleneck.parameters_to_predict

        # Base embed_dim: Re+Im for complex, direct for real.
        encoder_dimension: int = self.autoencoder.encoder.dimension
        self._m: int = encoder_dimension // self.parameters_to_predict
        base_dim: int = 2 * self._m if self.is_complex else self._m

        # standardize_bottleneck: fold the latent freq axis into channels
        # (C × F_lat) and pool over time only, mirroring SAO's 1-D
        # (channels × time) protocol so both encoders are probed at the same
        # embedding width. F_lat is fixed by the architecture (Swin keeps the
        # freq grid constant), discovered once here via a silent probe.
        self.standardize_bottleneck: bool = standardize_bottleneck
        self.f_lat: int | None = self._probe_f_lat() if standardize_bottleneck else None
        self.embed_dim: int = base_dim * self.f_lat if self.f_lat else base_dim

        self._cache: dict[str, torch.Tensor] = {}

        log.info(
            "SwinEncoder: is_complex=%s  hop=%d  num_downsamples=%d  "
            "p2p=%d  standardize=%s  f_lat=%s  embed_dim=%d  device=%s",
            self.is_complex, self.hop, self.num_downsamples,
            self.parameters_to_predict, self.standardize_bottleneck,
            self.f_lat, self.embed_dim, self.device,
        )

    # ------------------------------------------------------------------
    # One-time latent freq-grid probe (only needed for standardize_bottleneck)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _probe_f_lat(self) -> int:
        """Run one tiny forward on silence to read the latent freq-grid size F_lat.

        F_lat is fixed by the architecture (Swin keeps the freq axis constant),
        so a short silent clip suffices and padding is irrelevant. Avoids
        hardcoding the STFT/patch/merge freq formula — robust to any ckpt config.

        Returns:
            The number of latent frequency bands F_lat (e.g. 4 for the ×64 real
            baseline).
        """
        n = self.hop * (2 ** self.num_downsamples) * 8            # safe >1-window length
        wav = torch.zeros(self.audio_channels, n)
        wav_padded, _ = _pad_for_swin(wav, self.hop, self.num_downsamples)
        spec = self.autoencoder.stft(wav_padded.unsqueeze(0).to(self.device))   # (1,2,F,T)
        if not self.is_complex:
            spec = self.autoencoder._pack_complex(spec)                          # (1,4,F,T)
        enc_info = self.autoencoder.encode(spec, return_info=True)[1]
        f_lat, _ = enc_info["feature_shape"]                                     # (freq, time)
        return int(f_lat)

    # ------------------------------------------------------------------
    # Core encoding: single variable-T forward pass per clip
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_item(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode one waveform → 1-D embedding tensor of shape (embed_dim,).

        Args:
            wav: (C, T) float32 tensor, already resampled and channel-normalized.

        Returns:
            Float32 tensor of shape (embed_dim,).
        """
        wav_padded, orig_samples = _pad_for_swin(wav, self.hop, self.num_downsamples)

        # Single forward pass
        spec = self.autoencoder.stft(wav_padded.unsqueeze(0).to(self.device))  # (1,2,F,T)
        if not self.is_complex:
            spec = self.autoencoder._pack_complex(spec)                         # (1,4,F,T)

        result = self.autoencoder.encode(spec, return_info=True)
        enc_info = result[1]                                                    # 2- or 3-tuple
        pre_bn = enc_info["pre_bottleneck_latents"]                             # (1, p2p*m, L)  L=F*T
        F_lat, T_lat = enc_info["feature_shape"]                                # (freq, time) latent grid

        mu = pre_bn[:, : self._m]                                               # (1, m, L)
        if torch.is_complex(mu):
            mu = torch.cat([mu.real, mu.imag], dim=1)                           # (1, 2m, L)
        C = mu.shape[1]                                                         # = embed_dim

        # Recover the 2-D latent grid. The encoder flattens freq-major
        # (token = f*T_lat + t), so reshape to (C, F, T) and pool over ALL
        # frequency bands and only the VALID (non-padding) time frames — the
        # flat slice mu[..., :valid_T] would wrongly keep freq-band 0 only.
        mu = mu.reshape(1, C, F_lat, T_lat)                                     # (1, C, F, T)
        W_orig = orig_samples // self.hop + 1                                   # STFT frames of orig audio
        valid_T = min(math.ceil(W_orig / (2 ** self.num_downsamples)), T_lat)   # valid time frames
        mu = mu[..., :valid_T]                                                  # (1, C, F, T_valid)

        if self.standardize_bottleneck:
            # Fold freq → channels (C × F_lat), pool over VALID time only — the
            # padding has already been sliced off above, so this mirrors SAO's
            # masked time pool exactly. Bijective reshape: no info lost/mixed.
            Cf = mu.shape[1] * mu.shape[2]                                      # C * F_lat (= embed_dim)
            mu = mu.reshape(1, Cf, mu.shape[-1])                               # (1, C*F, T_valid)
            return mu.float().mean(dim=-1).squeeze(0)                          # (C*F,) pooled over time
        return mu.float().mean(dim=(-1, -2)).squeeze(0)                        # (C,) pooled over freq+time

    # ------------------------------------------------------------------
    # Batch encoding (MTEB DataLoader loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: "DataLoader",
        show_progress_bar: bool = True,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode all audio items from an MTEB DataLoader → (N, D) array.

        Corrupt / unreadable clips are skipped (substituted with a zero
        embedding) so a single bad file never aborts the whole task — mirroring
        the skip-on-failure policy of the ar_spectra training dataloader, while
        preserving one embedding per dataset row (label/qrel alignment).
        """
        all_embeddings: list[torch.Tensor] = []
        n_skipped = 0

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Swin encode"):
            for audio_item in batch["audio"]:
                path = audio_item.get("path")
                if path and path in self._cache:
                    all_embeddings.append(self._cache[path].cpu())
                    continue
                try:
                    wav = prepare_audio(
                        audio_item,
                        self.sampling_rate,
                        self.audio_channels,
                        self.max_audio_length_seconds,
                    )
                    emb = self._encode_item(wav)
                    if path:
                        self._cache[path] = emb
                except AudioDecodeError as e:
                    n_skipped += 1
                    log.warning("Skipping undecodable clip (%s) → zero embedding.", e)
                    emb = torch.zeros(self.embed_dim)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM on a single clip. "
                        "Try --device cpu or reduce --max-audio-sec."
                    ) from None
                all_embeddings.append(emb.cpu())

        if n_skipped:
            log.warning("Swin encode: %d clip(s) skipped (zero embedding).", n_skipped)
        return torch.stack(all_embeddings).float().numpy()                      # (N, D)

    # ------------------------------------------------------------------
    # MTEB EncoderProtocol entry point
    # ------------------------------------------------------------------

    def encode(
        self,
        inputs: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: "PromptType | None" = None,
        **kwargs: Any,
    ) -> "Array":
        if any(m != "audio" for m in task_metadata.modalities):
            raise ValueError(
                f"SwinEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(
                f"Task '{task_metadata.name}' provided no audio inputs "
                f"for prompt_type={prompt_type!r}."
            )
        return self.get_audio_embeddings(inputs, **kwargs)


# ===========================================================================
# ModelMeta builder
# ===========================================================================

def build_swin_model_meta(ckpt_path: str, encoder: SwinEncoder) -> ModelMeta:
    """Build MTEB ModelMeta for a Swin C-VAE checkpoint."""
    model_type = "cvae-cplx" if encoder.is_complex else "cvae-real"
    # Standardized runs get a distinct name so their results never collide with
    # the default (freq+time pooled) run of the same checkpoint.
    suffix = "__std" if encoder.standardize_bottleneck else ""
    return ModelMeta(
        loader=lambda model_name, revision, **kw: SwinEncoder(
            model_name=model_name, revision=revision, **kw
        ),
        name=f"{model_type}/{Path(ckpt_path).stem.replace(' ', '_')}{suffix}",
        languages=["eng-Latn"],
        open_weights=False,
        revision="local",
        release_date=None,
        max_tokens=None,
        n_parameters=None,
        memory_usage_mb=None,
        embed_dim=encoder.embed_dim,
        license=None,
        reference=None,
        similarity_fn_name="cosine",
        framework=["PyTorch"],
        use_instructions=False,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
        modalities=["audio"],
        citation=None,
    )
