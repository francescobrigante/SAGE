#!/usr/bin/env python3
# =============================================================================
# MAEB evaluation for ℂ-VAE (Complex/Real Swin VAE) checkpoints.
# Wraps the SwinEncoder in MTEB EncoderProtocol; embedding = VAE mean μ,
# Re+Im concatenated for complex models, mean-pooled over latent tokens.
# =============================================================================
"""
MAEB evaluation for ℂ-VAE (Complex/Real Swin VAE) checkpoints.

Wraps the C-VAE SwinEncoder in the MTEB EncoderProtocol and runs
audio-only MAEB tasks for semantic latent quality evaluation.

The encoder uses the VAE mean μ (deterministic — no sampling noise),
extracted from pre-bottleneck latents via enc_info["pre_bottleneck_latents"].
For complex models, Re and Im parts are concatenated before mean-pooling.
This produces iso-dim embeddings for ×64-compression real vs complex models
(both 16-dim for the swin_cplx_4s_x64 / swin_real_4s_x64 baseline pair).

Usage examples:
  # Full MAEB(audio-only) benchmark on a complex checkpoint:
  python evaluation/maeb_cvae.py \\
      --ckpt checkpoints/swin_cplx_4s_x64/best.ckpt

  # Specific tasks:
  python evaluation/maeb_cvae.py \\
      --ckpt checkpoints/swin_cplx_4s_x64/best.ckpt \\
      --tasks CREMA_D ESC50 GTZAN

  # Custom output directory and smaller batch:
  python evaluation/maeb_cvae.py \\
      --ckpt checkpoints/swin_real_4s_x64/best.ckpt \\
      --output-dir runs/maeb/swin_real \\
      --batch-size 4 --device cuda

  # List supported audio-only tasks:
  python evaluation/maeb_cvae.py --list-tasks

Embedding details:
  - Deterministic VAE mean μ — no reparameterization noise
  - Complex model: Re+Im concatenated → mean-pooled over 128 latent tokens
  - Real model:    direct μ → mean-pooled over 128 latent tokens
  - Embedding dim: 16 for both ×64-compression baseline models
"""
from __future__ import annotations

import sys
sys.stderr.write("MAEB C-VAE: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Route HuggingFace dataset cache to $FAST (fast scratch on Leonardo) when available,
# falling back to $WORK. Must be set before any HF import so datasets resolves it.
_hf_cache = os.path.expandvars("$FAST/hf_cache/datasets")
if _hf_cache == "$FAST/hf_cache/datasets":          # $FAST not set (local dev)
    _hf_cache = os.path.expandvars("$WORK/hf_cache/datasets")
if _hf_cache != "$WORK/hf_cache/datasets":           # at least one var resolved
    os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm

# Make C-VAE src importable when run from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.stderr.write("MAEB C-VAE: loading ar_spectra and mteb…\n")
sys.stderr.flush()

from ar_spectra.models.inference import EuleroEncodeDecode  # noqa: E402

import mteb  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402
from mteb.abstasks.classification import AbsTaskClassification  # noqa: E402
from sklearn.linear_model import LogisticRegression as _LogReg  # noqa: E402

# Raise solver iteration limit: MTEB default (100) often fails to converge.
AbsTaskClassification.evaluator_model = _LogReg(max_iter=1000)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)

_CVAE_SR = 44100
_CVAE_CHANNELS = 2
_DEFAULT_BENCHMARK = "MAEB(audio-only)"
_LEGACY_BENCHMARK_ALIASES = {"MAEB(audio)": _DEFAULT_BENCHMARK}


def _get_expected_frames(module: torch.nn.Module) -> int | None:
    """Recursively search for PatchEmbed img_size to determine fixed STFT time length."""
    if hasattr(module, "img_size") and isinstance(module.img_size, (tuple, list)) and len(module.img_size) == 2:
        return int(module.img_size[1])
    for child in module.children():
        result = _get_expected_frames(child)
        if result is not None:
            return result
    return None


# ===========================================================================
# MTEB encoder wrapper
# ===========================================================================
class CVAEEncoder(AbsEncoder):
    """MTEB-compatible wrapper for ℂ-VAE SwinEncoder checkpoints.

    Supports both complex (swin_cplx_4s_x64) and real (swin_real_4s_x64)
    model families. Long audio is chunked into fixed windows, each chunk
    encoded independently, then embeddings averaged per clip.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.max_audio_length_seconds = max_audio_length_seconds
        self._sub_batch_size_hint: int | None = None

        loader = EuleroEncodeDecode(model_name, device=device)
        self.autoencoder = loader.autoencoder
        self.device = loader.device
        self.sampling_rate: int = loader.sample_rate or _CVAE_SR
        self.audio_channels: int = loader.audio_channels or _CVAE_CHANNELS

        # Derive model properties from the loaded encoder.
        self.is_complex: bool = getattr(self.autoencoder.encoder, "is_complex", False)

        expected_frames = _get_expected_frames(self.autoencoder.encoder)
        if expected_frames is None:
            raise RuntimeError("Cannot determine expected_frames from encoder PatchEmbed.")
        self.expected_frames: int = expected_frames
        self.hop: int = self.autoencoder._stft_config.hop_length
        # chunk_samples: exact waveform length that maps to expected_frames STFT frames
        # (center=True STFT: frames = T // hop + 1, so T = (frames-1) * hop)
        self.chunk_samples: int = (expected_frames - 1) * self.hop

        # parameters_to_predict — needed to slice μ from pre_bottleneck_latents
        self.parameters_to_predict: int = self.autoencoder.bottleneck.parameters_to_predict

        # embed_dim: m real channels per token × 2 for complex (Re+Im concat), × 1 for real
        # encoder.dimension = parameters_to_predict × latent_channels (stored in SwinEncoder)
        encoder_dimension: int = self.autoencoder.encoder.dimension
        m = encoder_dimension // self.parameters_to_predict
        self.embed_dim: int = 2 * m if self.is_complex else m

        log.info(
            "CVAEEncoder: is_complex=%s, expected_frames=%d, chunk_samples=%d, "
            "p2p=%d, embed_dim=%d, device=%s",
            self.is_complex, expected_frames, self.chunk_samples,
            self.parameters_to_predict, self.embed_dim, self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_audio(self, audio_item: dict[str, Any]) -> torch.Tensor:
        """Convert one MAEB audio item → (2, T) float32 tensor at 44.1 kHz, clipped."""
        array = torch.as_tensor(audio_item["array"], dtype=torch.float32)
        sr: int = audio_item["sampling_rate"]

        # (samples, channels) → (channels, samples) for transposed stereo arrays
        if array.dim() == 2 and array.shape[0] > array.shape[1] and array.shape[1] <= 8:
            array = array.transpose(0, 1)
        if array.dim() == 1:
            array = array.unsqueeze(0)
        elif array.dim() != 2:
            raise ValueError(f"Unsupported audio shape {tuple(array.shape)}")

        if sr != self.sampling_rate:
            array = torchaudio.functional.resample(array, sr, self.sampling_rate)

        # Ensure exactly audio_channels channels
        if array.shape[0] < self.audio_channels:
            array = array.repeat(self.audio_channels, 1)
        elif array.shape[0] > self.audio_channels:
            array = array[:self.audio_channels]

        max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
        if array.shape[-1] > max_samples:
            array = array[..., :max_samples]

        return array.contiguous()

    def _waveform_to_chunks(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """Split (2, T) waveform → list of (2, chunk_samples) tensors, last padded."""
        chunks = list(torch.split(waveform, self.chunk_samples, dim=-1))
        last = chunks[-1]
        if last.shape[-1] < self.chunk_samples:
            chunks[-1] = F.pad(last, (0, self.chunk_samples - last.shape[-1]))
        return chunks

    @torch.no_grad()
    def _encode_chunk_batch(self, chunk_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of waveform chunks → mean-pooled μ embeddings.

        Args:
            chunk_batch: (B, 2, chunk_samples) float32

        Returns:
            embeddings: (B, D) float32 — D = 2*m (complex) or m (real)
        """
        spec = self.autoencoder.stft(chunk_batch)        # (B, 2, F, T_frames) complex

        # Pack complex→real for models that expect CAC input format
        if not self.is_complex:
            spec = self.autoencoder._pack_complex(spec)  # (B, 4, F, T_frames) real

        # autoencoder.encode applies power_norm pre_transform internally;
        # return_info=True gives enc_info with pre_bottleneck_latents
        result = self.autoencoder.encode(spec, return_info=True)
        enc_info = result[1]                             # works for 2-tuple and 3-tuple
        pre_bn = enc_info["pre_bottleneck_latents"]      # (B, p2p*m, 128) complex or real

        m = pre_bn.shape[1] // self.parameters_to_predict
        mu = pre_bn[:, :m]                               # (B, m, 128) — VAE mean

        if torch.is_complex(mu):
            mu = torch.cat([mu.real, mu.imag], dim=1)   # (B, 2m, 128)

        return mu.float().mean(dim=-1)                   # (B, D)

    @torch.no_grad()
    def _encode_prepared(
        self,
        prepared: list[torch.Tensor],
        sub_batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a list of variable-length waveforms → (N, D) float32 array.

        Each waveform is chunked into fixed windows, all chunks are batched
        together for a single forward pass, then averaged per original clip.
        OOM-resilient: halves sub_batch_size on CUDA OOM and retries the slice.
        """
        if sub_batch_size is None:
            sub_batch_size = self._sub_batch_size_hint or 8

        # Build flat list of (item_idx, chunk_tensor) pairs
        item_chunks: list[tuple[int, torch.Tensor]] = []
        for item_idx, wav in enumerate(prepared):
            for chunk in self._waveform_to_chunks(wav):
                item_chunks.append((item_idx, chunk))

        all_chunks = torch.stack([c for _, c in item_chunks], dim=0)  # (total, 2, chunk_samples)
        all_item_indices = [idx for idx, _ in item_chunks]

        # Sub-batched forward pass with OOM recovery
        safe_sub_bs = max(1, sub_batch_size)
        chunk_embeddings: list[torch.Tensor] = []
        i = 0
        while i < len(all_chunks):
            end = min(i + safe_sub_bs, len(all_chunks))
            try:
                emb = self._encode_chunk_batch(all_chunks[i:end].to(self.device))
                chunk_embeddings.append(emb.cpu())
                i = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if safe_sub_bs == 1:
                    raise RuntimeError(
                        "CUDA OOM even with sub_batch_size=1. "
                        "Try --device cpu or reducing --max-audio-sec."
                    ) from None
                safe_sub_bs = max(1, safe_sub_bs // 2)
                log.warning("CUDA OOM — retrying slice with sub_batch_size=%d.", safe_sub_bs)

        self._sub_batch_size_hint = safe_sub_bs

        # Scatter-add chunk embeddings and average per original item
        all_emb = torch.cat(chunk_embeddings, dim=0)    # (total_chunks, D)
        n_items = len(prepared)
        D = all_emb.shape[1]
        sums = torch.zeros(n_items, D)
        counts = torch.zeros(n_items)
        for chunk_idx, item_idx in enumerate(all_item_indices):
            sums[item_idx] += all_emb[chunk_idx]
            counts[item_idx] += 1
        return (sums / counts.unsqueeze(1).clamp(min=1)).float().numpy()  # (N, D)

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: "DataLoader",
        show_progress_bar: bool = True,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        sub_batch_size = min(batch_size, self._sub_batch_size_hint or batch_size)

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="C-VAE encode"):
            prepared = [self._prepare_audio(item) for item in batch["audio"]]
            all_embeddings.append(
                self._encode_prepared(prepared, sub_batch_size=sub_batch_size)
            )
            sub_batch_size = min(batch_size, self._sub_batch_size_hint or batch_size)

        return np.concatenate(all_embeddings, axis=0)

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
        if any(modality != "audio" for modality in task_metadata.modalities):
            raise ValueError(
                f"CVAEEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(
                f"Task '{task_metadata.name}' did not provide audio inputs for "
                f"prompt_type={prompt_type!r}."
            )
        return self.get_audio_embeddings(inputs, **kwargs)


# ===========================================================================
# ModelMeta builder
# ===========================================================================
def build_model_meta(ckpt_path: str, encoder: CVAEEncoder) -> ModelMeta:
    name = Path(ckpt_path).stem
    model_type = "cvae-cplx" if encoder.is_complex else "cvae-real"
    return ModelMeta(
        loader=lambda model_name, revision, **kw: CVAEEncoder(
            model_name=model_name, revision=revision, **kw
        ),
        name=f"{model_type}/{name}",
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


# ===========================================================================
# Task / benchmark helpers
# ===========================================================================
def _resolve_benchmark_name(benchmark_name: str) -> str:
    resolved = _LEGACY_BENCHMARK_ALIASES.get(benchmark_name, benchmark_name)
    if resolved != benchmark_name:
        log.warning("Benchmark alias '%s' is deprecated; using '%s'.", benchmark_name, resolved)
    return resolved


def _is_audio_only_task(task: Any) -> bool:
    return bool(task.metadata.modalities) and all(
        modality == "audio" for modality in task.metadata.modalities
    )


def _ensure_audio_only_tasks(tasks: list[Any], *, source: str) -> list[Any]:
    unsupported = [t for t in tasks if not _is_audio_only_task(t)]
    if unsupported:
        preview = ", ".join(
            f"{t.metadata.name} ({'/'.join(t.metadata.modalities)})"
            for t in unsupported[:8]
        )
        if len(unsupported) > 8:
            preview += f", ... (+{len(unsupported) - 8} more)"
        raise ValueError(
            f"CVAEEncoder only supports audio-only tasks. "
            f"Selection from {source} contains unsupported tasks: {preview}"
        )
    return tasks


def _load_audio_only_benchmark_tasks(benchmark_name: str) -> list[Any]:
    return _ensure_audio_only_tasks(
        list(mteb.get_benchmark(benchmark_name)),
        source=f"benchmark '{benchmark_name}'",
    )


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MAEB evaluation with a ℂ-VAE / Real-Swin checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Path to the C-VAE .ckpt file. Required unless --list-tasks.",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Explicit MTEB task names to run. Mutually exclusive with --benchmark.",
    )
    p.add_argument(
        "--benchmark",
        default=None,
        help=(
            'MTEB benchmark name (audio-only only). '
            'Default: "MAEB(audio-only)". '
            'Legacy alias "MAEB(audio)" is also accepted. '
            'Mutually exclusive with --tasks.'
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for MTEB results JSON. Default: ./maeb_results/<ckpt_stem>.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Default: cuda if available, else cpu.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="DataLoader batch size for encoding. Default: 8.",
    )
    p.add_argument(
        "--max-audio-sec",
        type=float,
        default=30.0,
        help="Max audio clip length in seconds. Default: 30.",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all audio-only MTEB tasks and exit.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-task results. Default: resume from disk.",
    )
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("MAEB C-VAE evaluation starting.")

    args = parse_args()

    if args.list_tasks:
        audio_tasks = list(mteb.get_tasks(modalities=["audio"]))
        supported = [t for t in audio_tasks if _is_audio_only_task(t)]
        excluded = len(audio_tasks) - len(supported)
        print(f"\nAudio-only MTEB tasks ({len(supported)}; excluded {excluded} cross-modal):\n")
        for t in sorted(supported, key=lambda x: x.metadata.name):
            print(f"  {t.metadata.name:<55}  type={t.metadata.type}")
        return

    if not args.ckpt:
        raise ValueError("--ckpt is required (unless --list-tasks is used).")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("maeb_results") / ckpt_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    if args.tasks and args.benchmark:
        raise ValueError("--tasks and --benchmark are mutually exclusive.")
    if args.benchmark:
        benchmark_name = _resolve_benchmark_name(args.benchmark)
        tasks = _load_audio_only_benchmark_tasks(benchmark_name)
        log.info("Benchmark '%s': %d tasks.", benchmark_name, len(tasks))
    elif args.tasks:
        tasks = _ensure_audio_only_tasks(
            list(mteb.get_tasks(tasks=args.tasks)),
            source="--tasks",
        )
        log.info("Explicit tasks: %s", [t.metadata.name for t in tasks])
    else:
        benchmark_name = _resolve_benchmark_name(_DEFAULT_BENCHMARK)
        tasks = _load_audio_only_benchmark_tasks(benchmark_name)
        log.info("Default %s suite: %d tasks.", benchmark_name, len(tasks))

    if not tasks:
        raise ValueError("No audio tasks found. Check --tasks / --benchmark.")

    # Build model
    log.info("Loading C-VAE checkpoint: %s", ckpt_path)
    model = CVAEEncoder(
        model_name=str(ckpt_path),
        device=args.device,
        max_audio_length_seconds=args.max_audio_sec,
    )
    model.mteb_model_meta = build_model_meta(str(ckpt_path), model)

    # Run evaluation — one task at a time for clean progress and ETA
    n_tasks = len(tasks)
    log.info(
        "Starting MTEB evaluation → %s (%d tasks). "
        "Results saved per task — interrupt and re-run to resume (--overwrite to restart).",
        output_dir, n_tasks,
    )
    results: list[Any] = []
    encode_kwargs = {"batch_size": args.batch_size}
    start_wall = time.perf_counter()

    for idx, task in enumerate(tasks):
        task_start = time.perf_counter()
        name = task.metadata.name
        log.info("\n%s Task %d/%d: %s %s", "=" * 60, idx + 1, n_tasks, name, "=" * 60)
        evaluation = mteb.MTEB(tasks=[task])
        task_results = evaluation.run(
            model,
            output_folder=str(output_dir),
            encode_kwargs=encode_kwargs,
            overwrite_results=args.overwrite,
        )
        results.extend(task_results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        task_elapsed = time.perf_counter() - task_start
        elapsed_total = time.perf_counter() - start_wall
        done = idx + 1
        if done < n_tasks:
            remaining_min = (elapsed_total / done) * (n_tasks - done) / 60
            log.info(
                "Task %d/%d done in %.1f min. Elapsed: %.1f min. ~%.0f min remaining.",
                done, n_tasks, task_elapsed / 60, elapsed_total / 60, remaining_min,
            )

    log.info(
        "All %d tasks completed in %.1f min.",
        n_tasks, (time.perf_counter() - start_wall) / 60,
    )

    # Print and save summary
    log.info("\n=== MAEB Results Summary ===")
    summary: dict[str, float | None] = {}
    for res in results:
        name = res.task_name
        main_score: float | None = None
        for split_scores in res.scores.values():
            if isinstance(split_scores, list):
                for s in split_scores:
                    if "main_score" in s:
                        main_score = s["main_score"]
                        break
            elif isinstance(split_scores, dict) and "main_score" in split_scores:
                main_score = split_scores["main_score"]
            if main_score is not None:
                break
        summary[name] = main_score
        if main_score is not None:
            log.info("  %-55s: %.4f", name, main_score)
        else:
            log.info("  %s: N/A", name)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "is_complex": model.is_complex,
                "embed_dim": model.embed_dim,
                "results": summary,
            },
            f,
            indent=2,
        )
    log.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
