#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/sao_encoder.py
# MTEB encoder wrapper for the Stable Audio Open ACE autoencoder.
# Embedding = deterministic VAE mean, single forward (no overlap-add), pooled
# over the valid (non-padding) latent frames. Uses the model's ORIGINAL
# inference interface only (preprocess_audio_list_for_encoder + encode_audio).
# =============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from . import compatibility
from .audio_prep import AudioDecodeError, prepare_audio, valid_latent_frames

# Vendored stable_audio_tools needs its repo on sys.path + heavy deps stubbed
# BEFORE importing the factory. repo_root = .../C-VAE (parents: maeb, evaluation).
_REPO_ROOT = Path(__file__).resolve().parents[2]
compatibility.add_stable_audio_to_path(_REPO_ROOT)
compatibility.stub_stable_audio_deps()

from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402

from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)


# ===========================================================================
# MTEB encoder wrapper
# ===========================================================================
class SAOACEEncoder(AbsEncoder):
    """MTEB-compatible wrapper for the Stable Audio Open ACE encoder.

    Each clip is encoded in a single forward pass (no chunking / overlap-add);
    the deterministic VAE posterior mean is taken (via skip_bottleneck, which
    returns [mean | log_scale] pre-bottleneck features), then pooled over the
    valid latent frames so each clip's own silence-padding is excluded.
    """

    def __init__(
        self,
        model_name: str,                 # path to .ckpt (also the unique id)
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_config_path: str | None = None,
        max_audio_length_seconds: float = 30.0,
        pooling: str = "mean",           # "mean" | "max"
        **kwargs: Any,
    ):
        self.model_name = model_name                              # ckpt path / unique id
        self.device = device                                     # torch device string
        self.max_audio_length_seconds = max_audio_length_seconds  # clip-length cap (s)
        self.pooling = pooling                                   # temporal pooling mode
        self._sub_batch_size_hint: int | None = None             # last OK forward sub-batch
        self._cache: dict[str, np.ndarray] = {}                  # path -> embedding cache

        log.info("Loading SAO-ACE checkpoint: %s", model_name)
        ckpt = torch.load(model_name, map_location="cpu", weights_only=False)

        # Config source: prefer the one embedded in the ckpt (evaluate_sao.py
        # pattern); fall back to an explicit --model-config JSON.
        model_config = ckpt.get("model_config") if isinstance(ckpt, dict) else None
        if model_config is None:
            if not model_config_path:
                raise ValueError(
                    f"{model_name}: checkpoint has no embedded 'model_config'; "
                    "pass --model-config pointing at the ACE model JSON."
                )
            with open(model_config_path) as f:
                model_config = json.load(f)

        self.model = create_model_from_config(model_config)
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        # Diffusion training ckpts wrap the AE under "autoencoder.*"; strip the prefix
        # so sao_encoder works with both unwrapped AE ckpts and full diffusion ckpts.
        _PREFIX = "autoencoder."
        if any(k.startswith(_PREFIX) for k in state_dict):
            state_dict = {
                k[len(_PREFIX):]: v
                for k, v in state_dict.items()
                if k.startswith(_PREFIX) and not k.startswith("autoencoder_ema")
            }
            log.info("Stripped 'autoencoder.' prefix from state_dict (%d keys).", len(state_dict))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(
                f"Checkpoint missing {len(missing)} model keys; first: {missing[:5]}. "
                "Pass an UNWRAPPED autoencoder state_dict or a diffusion ckpt with "
                "'autoencoder.*' keys."
            )
        if unexpected:
            # Tolerated (matches evaluate_sao.py): extra buffers / non-autoencoder keys.
            log.warning("Checkpoint has %d unexpected keys; first: %s",
                        len(unexpected), unexpected[:5])

        self.sampling_rate: int = int(model_config.get("sample_rate") or 44100)
        self.audio_channels: int = int(
            model_config.get("audio_channels")
            or model_config.get("model", {}).get("io_channels")
            or 2
        )
        self.downsample: int = int(self.model.downsampling_ratio)   # samples per latent frame
        self.embed_dim: int = int(self.model.latent_dim)            # VAE mean dim

        self.model.eval().to(self.device)
        log.info(
            "SAOACEEncoder ready: sr=%d, ch=%d, downsample=%d, embed_dim=%d, "
            "pooling=%s, device=%s",
            self.sampling_rate, self.audio_channels, self.downsample,
            self.embed_dim, self.pooling, self.device,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_prepared(
        self, prepared: list[torch.Tensor], sub_batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode (C,T) waveforms → (N, embed_dim) via single forward + masked pool.

        Global padding to the batch length is done once by the model's own
        preprocess_audio_list_for_encoder (cheap, no forward), so embeddings stay
        invariant to how the batch is split for OOM recovery.
        """
        if sub_batch_size is None:
            sub_batch_size = self._sub_batch_size_hint or len(prepared)
        safe_bs = max(1, sub_batch_size)

        # Each clip's own valid (non-pad) latent-frame count, computed pre-padding.
        valid_frames = torch.tensor(
            [valid_latent_frames(int(w.shape[-1]), self.downsample) for w in prepared],
            dtype=torch.long,
        )

        padded = self.model.preprocess_audio_list_for_encoder(
            list(prepared), self.sampling_rate,
        )                                                       # (N, C, T_global) CPU

        parts: list[np.ndarray] = []
        i = 0
        while i < len(prepared):
            end = min(i + safe_bs, len(prepared))
            try:
                audio_sub = padded[i:end].to(self.device)       # (b, C, T_global)
                vf_sub = valid_frames[i:end].to(self.device)    # (b,)
                
                # single forward, no overlap-add; skip_bottleneck → [mean|log_scale]
                pre_bn = self.model.encode_audio(
                    audio_sub, chunked=False, skip_bottleneck=True,
                )                                               # (b, 2*D, T_lat)
                mean, _ = pre_bn.chunk(2, dim=1)                # (b, D, T_lat) det. mean
                vf_sub = vf_sub.clamp(max=mean.shape[-1])
                mask = (
                    torch.arange(mean.shape[-1], device=self.device).unsqueeze(0)
                    < vf_sub.unsqueeze(1)
                )                                               # (b, T_lat) valid-frame mask

                if self.pooling == "mean":
                    emb = (mean * mask.unsqueeze(1)).sum(-1) / vf_sub.clamp_min(1).unsqueeze(1)
                elif self.pooling == "max":
                    neg_inf = torch.finfo(mean.dtype).min
                    emb = mean.masked_fill(~mask.unsqueeze(1), neg_inf).max(-1).values
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")
                
                parts.append(emb.cpu().float().numpy())          # (b, D)
                i = end

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if safe_bs == 1:
                    raise RuntimeError(
                        "CUDA OOM with sub_batch_size=1. "
                        "Try --device cpu or a smaller --max-audio-sec."
                    ) from None
                safe_bs = max(1, safe_bs // 2)
                log.warning("CUDA OOM — retrying slice with sub_batch_size=%d.", safe_bs)

        self._sub_batch_size_hint = safe_bs
        return np.concatenate(parts, axis=0)

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: "DataLoader",
        show_progress_bar: bool = True,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        n_skipped = 0
        sub_bs = min(batch_size, self._sub_batch_size_hint or batch_size)
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="SAO-ACE encode"):
            prepared: list = []
            good_positions: list[int] = []
            cached_emb: dict[int, np.ndarray] = {}
            items = batch["audio"]
            for i, item in enumerate(items):
                path = item.get("path")
                if path and path in self._cache:
                    cached_emb[i] = self._cache[path]
                    continue
                try:
                    prepared.append(
                        prepare_audio(item, self.sampling_rate, self.audio_channels,
                                      self.max_audio_length_seconds)
                    )
                    good_positions.append(i)
                except AudioDecodeError as e:
                    n_skipped += 1
                    log.warning("Skipping undecodable clip (%s) → zero embedding.", e)

            full = np.zeros((len(items), self.embed_dim), dtype=np.float32)
            if prepared:
                emb = self._encode_prepared(prepared, sub_batch_size=sub_bs)   # (n_good, D)
                full[good_positions] = emb
                for idx, emb_row in zip(good_positions, emb):
                    path = items[idx].get("path")
                    if path:
                        self._cache[path] = emb_row
            for i, c_emb in cached_emb.items():
                full[i] = c_emb
                
            all_emb.append(full)
            sub_bs = min(batch_size, self._sub_batch_size_hint or batch_size)

        if n_skipped:
            log.warning("SAO-ACE encode: %d clip(s) skipped (zero embedding).", n_skipped)
        return np.concatenate(all_emb, axis=0)

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
                f"SAOACEEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(
                f"Task '{task_metadata.name}' provided no audio inputs "
                f"(prompt_type={prompt_type!r})."
            )
        return self.get_audio_embeddings(inputs, **kwargs)


# ===========================================================================
# ModelMeta builder
# ===========================================================================
def build_model_meta(ckpt_path: str, encoder: SAOACEEncoder) -> ModelMeta:
    name = Path(ckpt_path).stem
    return ModelMeta(
        loader=lambda model_name, revision, **kw: SAOACEEncoder(
            model_name=model_name, revision=revision, **kw,
        ),
        name=f"sao-ace/{name.replace(' ', '_')}",
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
