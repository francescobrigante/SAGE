#!/usr/bin/env python3
# =============================================================================
# evaluation/maeb/clap_encoder.py
# MTEB encoder wrapper for the FROZEN LAION-CLAP teacher — the "CLAP-oracle"
# reference row (NOT a baseline: CLAP is not invertible, it cannot reconstruct
# audio, so it never competes on the fidelity axis).
#
# Purpose: bound how much of the distillation teacher's semantic power survives
# inside SAGE's 64-d invertible latent. The checkpoint MUST be the exact one
# distilled in training (music_audioset_epoch_15_esc_90.14.pt, see
# CLAPTeacher in src/sage/nn/losses/semantic.py) — a different CLAP would
# answer a different question. The loading mirrors CLAPTeacher; keep in sync.
#
# Two caveats to tabulate with the numbers (both make the oracle OPTIMISTIC):
#   * 512-d vs SAGE's 64-d — not width-matched (same convention as SAME's 256-d).
#   * CLAP is contrastively pretrained on AudioSet + captioned music, so GTZAN /
#     MusicGenre / NSynth are plausibly in-distribution for it and are NOT for
#     SAGE (trained label-free on FMA-full + Jamendo + M4Singer).
#
# Windowing: laion_clap with enable_fusion=False truncates via 'rand_trunc',
# which RANDOM-crops 10 s out of a longer clip (training/data.py:465-468) —
# non-deterministic and discards 2/3 of a 30-s clip. We therefore feed exact
# 10-s windows (len == max_len makes that crop branch unreachable) and mean the
# per-window embeddings, which is deterministic, uses the whole clip, and
# mirrors the mean-over-time pooling the codec encoders use.
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .audio_prep import AudioDecodeError, prepare_audio

from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)

CLAP_SR = 48000                 # LAION-CLAP operates natively at 48 kHz
CLAP_WINDOW = 480000            # 10 s @ 48 kHz — laion_clap's `max_len`
CLAP_EMBED_DIM = 512


class CLAPOracleEncoder(AbsEncoder):
    """MTEB-compatible encoder exposing the frozen distillation teacher.

    One clip → mean of the per-10-s-window CLAP embeddings → [512]. Mono, since
    CLAP downmixes internally anyway (semantic.py does the same before the
    teacher forward, so the training-time and eval-time inputs agree).
    """

    def __init__(
        self,
        ckpt_path: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        amodel: str = "HTSAT-base",      # architecture of the music_audioset ckpt
        l2_normalize: bool = True,       # re-normalize after window averaging
        **kwargs: Any,
    ):
        import laion_clap

        self.model_name = "clap-oracle-teacher"
        self.ckpt_path = str(ckpt_path)
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds
        self.l2_normalize = l2_normalize
        self.sampling_rate = CLAP_SR
        self.audio_channels = 1                       # mono: CLAP downmixes anyway
        self.embed_dim = CLAP_EMBED_DIM
        self._cache: dict[str, np.ndarray] = {}       # path -> embedding

        if not Path(self.ckpt_path).is_file():
            raise FileNotFoundError(f"CLAP teacher checkpoint not found: {self.ckpt_path}")

        log.info("Loading frozen CLAP teacher (%s) from %s", amodel, self.ckpt_path)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel)
        self.clap.load_ckpt(self.ckpt_path)
        self.clap.requires_grad_(False).eval()
        self.clap.to(device)

        n = sum(p.numel() for p in self.clap.parameters())
        log.info("CLAPOracleEncoder ready: sr=%d embed_dim=%d params=%.1fM "
                 "window=%ds l2norm=%s device=%s",
                 self.sampling_rate, self.embed_dim, n / 1e6,
                 CLAP_WINDOW // CLAP_SR, self.l2_normalize, self.device)

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """[T] mono @48k → [n_win, CLAP_WINDOW], deterministic, covering all of x.

        Clips shorter than one window are passed through untouched: laion_clap
        then takes its 'repeatpad' branch, which is deterministic.
        """
        T = x.shape[0]
        if T <= CLAP_WINDOW:
            return x.unsqueeze(0)                                  # [1, T]
        starts = list(range(0, T - CLAP_WINDOW + 1, CLAP_WINDOW))
        if starts[-1] + CLAP_WINDOW < T:                           # keep the tail
            starts.append(T - CLAP_WINDOW)                         # end-aligned window
        return torch.stack([x[s: s + CLAP_WINDOW] for s in starts], dim=0)

    @torch.no_grad()
    def _embed_one(self, wav: torch.Tensor) -> np.ndarray:
        """[C, T] waveform @48k → [512] embedding, averaged over 10-s windows."""
        x = wav.mean(dim=0) if wav.dim() == 2 else wav             # [T] mono
        wins = self._windows(x).to(self.device)                    # [n_win, W]
        emb = self.clap.get_audio_embedding_from_data(x=wins, use_tensor=True)  # [n_win, 512]
        emb = emb.float().mean(dim=0)                              # [512]
        if self.l2_normalize:
            emb = emb / emb.norm().clamp_min(1e-12)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: "DataLoader",
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        n_skipped = 0
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="clap-oracle encode"):
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
            log.warning("clap-oracle encode: %d clip(s) skipped (zero embedding).", n_skipped)
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
                f"CLAPOracleEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(f"Task '{task_metadata.name}' provided no audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


def build_clap_model_meta(encoder: CLAPOracleEncoder) -> ModelMeta:
    return ModelMeta(
        loader=lambda model_name, revision, **kw: CLAPOracleEncoder(
            ckpt_path=encoder.ckpt_path, revision=revision, **kw,
        ),
        name="oracle/clap-teacher",
        languages=["eng-Latn"],
        open_weights=True,
        revision="music_audioset_epoch_15_esc_90.14",
        release_date=None,
        max_tokens=None,
        n_parameters=None,
        memory_usage_mb=None,
        embed_dim=encoder.embed_dim,
        license=None,
        reference="https://github.com/LAION-AI/CLAP",
        similarity_fn_name="cosine",
        framework=["PyTorch"],
        use_instructions=False,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
        modalities=["audio"],
        citation=None,
    )
