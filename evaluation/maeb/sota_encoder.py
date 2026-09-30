#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/sota_encoder.py
# MTEB encoder wrapper for SOTA baseline codecs (codicodec, music2latent, sao-vae,
# same) via the shared CodecAdapter.encode_latent. Embedding = deterministic latent,
# single per-clip forward, mean-pooled over latent time frames → the model's native
# latent width (64-d for the x64-class codecs, matching SAO; 256-d for SAME, which
# is NOT width-matched). Reuses maeb/audio_prep for robust decode + channel/sr.
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .audio_prep import AudioDecodeError, prepare_audio

# evaluation/codecs.py holds the CodecAdapter factory (lazy model imports).
from evaluation.codecs import build_adapter

from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)


class SOTACodecEncoder(AbsEncoder):
    """MTEB-compatible encoder for codicodec / music2latent.

    Each clip is encoded individually (no cross-clip batching) via
    adapter.encode_latent → [D, T_lat], then pooled over the time axis. The
    latent already has no padding (full 30-s clips), so a plain temporal pool
    matches SAO's valid-frame pool at this scale.
    """

    def __init__(
        self,
        model_name: str,                 # adapter key: "codicodec" | "music2latent"
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        pooling: str = "mean",           # "mean" | "max"
        adapter_kwargs: dict | None = None,   # e.g. {"model_dir": ...} for sao-vae
        **kwargs: Any,
    ):
        self.model_name = model_name                              # adapter key / unique id
        self.device = device                                     # torch device string
        self.max_audio_length_seconds = max_audio_length_seconds  # clip-length cap (s)
        self.pooling = pooling                                   # temporal pooling mode
        self._cache: dict[str, np.ndarray] = {}                  # path -> embedding cache

        log.info("Building adapter '%s' on %s", model_name, device)
        self.adapter = build_adapter(model_name, device=device, **(adapter_kwargs or {}))
        self.sampling_rate = int(self.adapter.sample_rate)
        self.audio_channels = int(self.adapter.audio_channels)

        # Probe the latent width once (codicodec/music2latent → 64).
        probe = torch.zeros(self.audio_channels, self.sampling_rate)
        self.embed_dim = int(self.adapter.encode_latent(probe).shape[0])
        log.info("SOTACodecEncoder ready: sr=%d ch=%d embed_dim=%d pooling=%s device=%s",
                 self.sampling_rate, self.audio_channels, self.embed_dim, self.pooling, self.device)

    @torch.no_grad()
    def _embed_one(self, wav: torch.Tensor) -> np.ndarray:
        """[C, T] waveform → [embed_dim] pooled latent embedding."""
        lat = self.adapter.encode_latent(wav)                    # [D, T_lat]
        if self.pooling == "mean":
            emb = lat.mean(dim=-1)
        elif self.pooling == "max":
            emb = lat.max(dim=-1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return emb.cpu().float().numpy()                         # [D]

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: "DataLoader",
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        n_skipped = 0
        for batch in tqdm(inputs, disable=not show_progress_bar, desc=f"{self.model_name} encode"):
            for item in batch["audio"]:
                path = item.get("path")
                if path and path in self._cache:
                    all_emb.append(self._cache[path]); continue
                try:
                    wav = prepare_audio(item, self.sampling_rate, self.audio_channels,
                                        self.max_audio_length_seconds)
                    emb = self._embed_one(wav)
                except AudioDecodeError as e:
                    n_skipped += 1
                    log.warning("Skipping undecodable clip (%s) → zero embedding.", e)
                    emb = np.zeros(self.embed_dim, dtype=np.float32)
                if path:
                    self._cache[path] = emb
                all_emb.append(emb)
        if n_skipped:
            log.warning("%s encode: %d clip(s) skipped (zero embedding).", self.model_name, n_skipped)
        return np.stack(all_emb, axis=0)

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
                f"SOTACodecEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(f"Task '{task_metadata.name}' provided no audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


def build_model_meta(model_name: str, encoder: SOTACodecEncoder) -> ModelMeta:
    return ModelMeta(
        loader=lambda model_name, revision, **kw: SOTACodecEncoder(
            model_name=model_name, revision=revision, **kw,
        ),
        name=f"sota/{model_name}",
        languages=["eng-Latn"],
        open_weights=True,
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
