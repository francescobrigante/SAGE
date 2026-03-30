#!/usr/bin/env python3
"""
MAEB evaluation for Stable Audio Autoencoder ACE (SAO-ACE).

Wraps the SAO-ACE encoder in the MTEB EncoderProtocol and runs
audio-only MTEB/MAEB tasks supported by an audio-only encoder.

Usage examples:
  # Single best checkpoint, full supported MAEB audio-only benchmark:
  conda run -n maeb python maeb_sao_ace.py \\
      --ckpt /home/michelemancusi/stable-audio-tools/alrurt2n_4330k.ckpt

  # Specific audio-only tasks:
  conda run -n maeb python maeb_sao_ace.py \\
      --ckpt /home/michelemancusi/stable-audio-tools/alrurt2n_4330k.ckpt \\
      --tasks CREMA_D ESC50

  # Explicit benchmark name (legacy alias "MAEB(audio)" is also accepted):
  conda run -n maeb python maeb_sao_ace.py \\
      --ckpt /home/michelemancusi/stable-audio-tools/alrurt2n_4330k.ckpt \\
      --benchmark "MAEB(audio-only)"

  # Custom output dir and batch size:
  conda run -n maeb python maeb_sao_ace.py \\
      --ckpt /home/michelemancusi/stable-audio-tools/alrurt2n_4330k.ckpt \\
      --output-dir /mnt/michele-disk/data/maeb_results/alrurt2n_4330k \\
      --batch-size 16 --device cuda

Model details:
  - Config: stable_audio_2_0_vae_ACE.json
  - Sample rate: 44100 Hz (stereo)
  - Latent dim: 64 (VAE mean, pooled over time)
  - Downsampling ratio: 2048x
  - Embedding: deterministic VAE mean, pooled across valid latent frames
"""
from __future__ import annotations

import sys
# First thing: show we started (stderr is unbuffered; conda run may not show stdout early)
sys.stderr.write("MAEB SAO-ACE: script started, loading dependencies…\n")
sys.stderr.flush()

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Make stable_audio_tools importable if not installed in this env.
# Stub out heavy / unavailable optional deps that are not needed for encoding.
# ---------------------------------------------------------------------------
_SAT_REPO = Path("/home/michelemancusi/stable-audio-tools")
if _SAT_REPO.exists() and str(_SAT_REPO) not in sys.path:
    sys.path.insert(0, str(_SAT_REPO))

import types  # noqa: E402

def _make_stub(name: str) -> types.ModuleType:
    """Return a dummy module so that `import X` doesn't raise ModuleNotFoundError."""
    mod = types.ModuleType(name)
    mod.__spec__ = None  # type: ignore[attr-defined]
    return mod

sys.stderr.write("MAEB SAO-ACE: loading stable_audio_tools and mteb (30–60s)…\n")
sys.stderr.flush()

for _stub in [
    "k_diffusion",
    "laion_clap",
    "prefigure",
    "wandb",
    "gradio",
    "v_diffusion_pytorch",
    "local_attention",
    "vector_quantize_pytorch",
    "webdataset",
    "pytorch_lightning",
]:
    if _stub not in sys.modules:
        sys.modules[_stub] = _make_stub(_stub)

from stable_audio_tools.models.factory import create_model_from_config_path  # noqa: E402

import mteb  # noqa: E402
from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta  # noqa: E402
from mteb.abstasks.classification import AbsTaskClassification  # noqa: E402
from sklearn.linear_model import LogisticRegression as _LogReg  # noqa: E402

# Raise the solver iteration limit to avoid ConvergenceWarning on classification tasks.
# MTEB default is max_iter=100; 1000 converges reliably without being noticeably slower.
AbsTaskClassification.evaluator_model = _LogReg(max_iter=1000)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_CONFIG = str(
    _SAT_REPO
    / "stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae_ACE.json"
)
_SAO_ACE_SR = 44100
_SAO_ACE_CHANNELS = 2
_SAO_ACE_LATENT_DIM = 64
_SAO_ACE_DOWNSAMPLE = 2048  # samples per latent frame
_DEFAULT_BENCHMARK = "MAEB(audio-only)"
_DEFAULT_EXTRA_AUDIO_TASKS = [
    "NSynth",
    "GTZANGenreClustering",
    "MusicGenreClustering",
]
_LEGACY_BENCHMARK_ALIASES = {
    "MAEB(audio)": _DEFAULT_BENCHMARK,
}


# ---------------------------------------------------------------------------
# SAO-ACE MTEB encoder wrapper
# ---------------------------------------------------------------------------
class SAOACEEncoder(AbsEncoder):
    """MTEB-compatible wrapper for the Stable Audio Autoencoder ACE encoder."""

    def __init__(
        self,
        model_name: str,          # path to .ckpt (used as unique identifier too)
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_config_path: str = _DEFAULT_MODEL_CONFIG,
        max_audio_length_seconds: float = 30.0,
        pooling: str = "mean",    # "mean" | "max"
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds
        self.pooling = pooling
        self.sampling_rate = _SAO_ACE_SR
        self._sub_batch_size_hint: int | None = None

        log.info(f"Loading SAO-ACE model config from: {model_config_path}")
        self.model = create_model_from_config_path(model_config_path)

        log.info(f"Loading SAO-ACE weights from: {model_name}")
        ckpt = torch.load(model_name, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(
                f"Checkpoint is missing {len(missing)} model keys; "
                f"first keys: {missing[:5]}"
            )
        if unexpected:
            raise ValueError(
                f"Checkpoint contains {len(unexpected)} unexpected keys; "
                f"first keys: {unexpected[:5]}"
            )

        self.model.eval()
        self.model.to(self.device)
        log.info(f"SAO-ACE encoder ready on {self.device}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_audio(self, audio_item: dict[str, Any]) -> tuple[torch.Tensor, int]:
        """
        Convert a single MTEB audio item to a float32 tensor in (channels, samples)
        format at SAO-ACE sample rate and return the number of valid latent frames.
        """
        array = torch.as_tensor(audio_item["array"], dtype=torch.float32)
        sr: int = audio_item["sampling_rate"]

        # Heuristic for datasets returning stereo audio as (samples, channels).
        if array.dim() == 2 and array.shape[0] > array.shape[1] and array.shape[1] <= 8:
            array = array.transpose(0, 1)

        if array.dim() == 1:
            array = array.unsqueeze(0)
        elif array.dim() != 2:
            raise ValueError(
                f"Unsupported audio tensor rank {array.dim()} for item with "
                f"sampling rate {sr}."
            )

        if sr != self.sampling_rate:
            array = torchaudio.functional.resample(array, sr, self.sampling_rate)

        max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
        if array.shape[-1] > max_samples:
            array = array[..., :max_samples]

        valid_frames = max(
            1,
            (int(array.shape[-1]) + _SAO_ACE_DOWNSAMPLE - 1) // _SAO_ACE_DOWNSAMPLE,
        )

        return array.contiguous(), valid_frames

    @torch.no_grad()
    def _encode_prepared(
        self,
        prepared: list,
        sub_batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a list of (tensor, valid_frames) pairs with OOM-resilient sub-batching.

        Padding is computed once for the entire batch (cheap operation, no forward
        pass) so every sample always sees the same global target length, regardless
        of how the batch is later split.  Sub-batching therefore only affects the
        model forward pass and leaves embeddings invariant to OOM fallbacks.

        If a CUDA OOM occurs during a forward pass the sub-batch size is halved
        and the failed slice is retried, down to a minimum of 1 sample.
        """
        if sub_batch_size is None:
            sub_batch_size = self._sub_batch_size_hint or len(prepared)
        safe_sub_batch_size = max(1, sub_batch_size)

        # --- global padding (cheap: no model forward, just resampling + pad) ---
        tensors = [audio for audio, _ in prepared]
        valid_frames_all = torch.tensor(
            [frames for _, frames in prepared],
            dtype=torch.long,
        )
        padded_all = self.model.preprocess_audio_list_for_encoder(
            tensors,
            self.sampling_rate,
        )  # (N, C, T_global) on CPU — every sample sees the same T_global

        # --- sub-batched forward pass ---
        parts: list[np.ndarray] = []
        i = 0
        while i < len(prepared):
            end = min(i + safe_sub_batch_size, len(prepared))
            try:
                padded_sub = padded_all[i:end].to(self.device)
                vf_sub = valid_frames_all[i:end].to(self.device)

                # skip_bottleneck=True → (B, 128, T//2048); chunk → mean (B, 64, T//2048)
                pre_bn = self.model.encode(padded_sub, skip_bottleneck=True)
                mean, _ = pre_bn.chunk(2, dim=1)
                vf_sub = vf_sub.clamp(max=mean.shape[-1])
                frame_mask = (
                    torch.arange(mean.shape[-1], device=self.device).unsqueeze(0)
                    < vf_sub.unsqueeze(1)
                )
                if self.pooling == "mean":
                    masked_sum = (mean * frame_mask.unsqueeze(1)).sum(dim=-1)
                    emb = masked_sum / vf_sub.clamp_min(1).unsqueeze(1)
                elif self.pooling == "max":
                    neg_inf = torch.finfo(mean.dtype).min
                    masked_mean = mean.masked_fill(~frame_mask.unsqueeze(1), neg_inf)
                    emb = masked_mean.max(dim=-1).values
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")
                parts.append(emb.cpu().float().numpy())
                i = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if safe_sub_batch_size == 1:
                    raise RuntimeError(
                        "CUDA OOM even with sub_batch_size=1. "
                        "Try --device cpu or reducing --max-audio-sec."
                    ) from None
                safe_sub_batch_size = max(1, safe_sub_batch_size // 2)
                log.warning(
                    "CUDA OOM — retrying current slice with sub_batch_size=%d.",
                    safe_sub_batch_size,
                )

        self._sub_batch_size_hint = safe_sub_batch_size
        return np.concatenate(parts, axis=0)

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: DataLoader,
        show_progress_bar: bool = True,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings = []
        sub_batch_size = min(batch_size, self._sub_batch_size_hint or batch_size)

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="SAO-ACE encode"):
            audio_list = batch["audio"]
            prepared = [self._prepare_audio(audio_item) for audio_item in audio_list]
            all_embeddings.append(
                self._encode_prepared(prepared, sub_batch_size=sub_batch_size)
            )
            sub_batch_size = min(batch_size, self._sub_batch_size_hint or batch_size)

        return np.concatenate(all_embeddings, axis=0)

    # ------------------------------------------------------------------
    # MTEB EncoderProtocol
    # ------------------------------------------------------------------
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if any(modality != "audio" for modality in task_metadata.modalities):
            raise ValueError(
                f"SAOACEEncoder is audio-only, but task '{task_metadata.name}' "
                f"requires modalities {task_metadata.modalities}."
            )
        if "audio" not in inputs.dataset.features:
            raise ValueError(
                f"Task '{task_metadata.name}' did not provide audio inputs for "
                f"prompt_type={prompt_type!r}."
            )
        return self.get_audio_embeddings(inputs, **kwargs)


# ---------------------------------------------------------------------------
# ModelMeta builder
# ---------------------------------------------------------------------------
def build_model_meta(ckpt_path: str) -> ModelMeta:
    name = Path(ckpt_path).stem   # e.g. "alrurt2n_4330k"
    return ModelMeta(
        loader=lambda model_name, revision, **kw: SAOACEEncoder(
            model_name=model_name, revision=revision, **kw
        ),
        name=f"sao-ace/{name}",
        languages=["eng-Latn"],
        open_weights=False,
        revision="local",
        release_date=None,
        max_tokens=None,
        n_parameters=None,
        memory_usage_mb=None,
        embed_dim=_SAO_ACE_LATENT_DIM,
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


# ---------------------------------------------------------------------------
# Task / benchmark helpers
# ---------------------------------------------------------------------------
def _resolve_benchmark_name(benchmark_name: str) -> str:
    resolved_name = _LEGACY_BENCHMARK_ALIASES.get(benchmark_name, benchmark_name)
    if resolved_name != benchmark_name:
        log.warning(
            "Benchmark alias '%s' is deprecated; using '%s' instead.",
            benchmark_name,
            resolved_name,
        )
    return resolved_name


def _is_audio_only_task(task: Any) -> bool:
    return bool(task.metadata.modalities) and all(
        modality == "audio" for modality in task.metadata.modalities
    )


def _ensure_audio_only_tasks(tasks: list[Any], *, source: str) -> list[Any]:
    unsupported = [task for task in tasks if not _is_audio_only_task(task)]
    if unsupported:
        unsupported_preview = ", ".join(
            f"{task.metadata.name} ({'/'.join(task.metadata.modalities)})"
            for task in unsupported[:8]
        )
        if len(unsupported) > 8:
            unsupported_preview += f", ... (+{len(unsupported) - 8} more)"
        raise ValueError(
            "SAOACEEncoder only supports audio-only tasks. "
            f"The selection from {source} contains unsupported cross-modal tasks: "
            f"{unsupported_preview}"
        )
    return tasks


def _dedupe_tasks(tasks: list[Any]) -> list[Any]:
    deduped = []
    seen_names: set[str] = set()
    for task in tasks:
        name = task.metadata.name
        if name in seen_names:
            continue
        seen_names.add(name)
        deduped.append(task)
    return deduped


def _load_audio_only_benchmark_tasks(benchmark_name: str) -> list[Any]:
    tasks = _ensure_audio_only_tasks(
        list(mteb.get_benchmark(benchmark_name)),
        source=f"benchmark '{benchmark_name}'",
    )

    # Upstream MAEB(audio-only) currently misses a few audio-only tasks that are
    # supported by this encoder, so include them in the default SAO-ACE suite.
    if benchmark_name == _DEFAULT_BENCHMARK:
        extra_tasks = _ensure_audio_only_tasks(
            list(mteb.get_tasks(tasks=_DEFAULT_EXTRA_AUDIO_TASKS)),
            source="default extra audio-only tasks",
        )
        existing_names = {task.metadata.name for task in tasks}
        added_names = [
            task.metadata.name
            for task in extra_tasks
            if task.metadata.name not in existing_names
        ]
        if added_names:
            log.info(
                "Extending %s with extra audio-only tasks supported by SAO-ACE: %s",
                benchmark_name,
                added_names,
            )
        tasks = _dedupe_tasks(tasks + extra_tasks)

    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MAEB evaluation with the SAO-ACE encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Path to the unwrapped SAO-ACE .ckpt file (state_dict only). "
             "Required unless --list-tasks is used.",
    )
    p.add_argument(
        "--model-config",
        default=_DEFAULT_MODEL_CONFIG,
        help=f"Path to the model JSON config. Default: {_DEFAULT_MODEL_CONFIG}",
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
        help='MTEB benchmark name/alias. For ACE use audio-only benchmarks only, '
             'e.g. "MAEB(audio-only)". Legacy alias "MAEB(audio)" is also '
             'accepted. Mutually exclusive with --tasks.',
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save MTEB results JSON. Defaults to "
             "./maeb_results/<ckpt_stem>.",
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
        help="Batch size for encoding. Default: 8.",
    )
    p.add_argument(
        "--max-audio-sec",
        type=float,
        default=30.0,
        help="Max audio clip length in seconds. Default: 30.",
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
        help="Temporal pooling strategy. Default: mean.",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print supported audio-only MTEB tasks for SAO-ACE and exit.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-task results (default: skip and resume from disk).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("MAEB SAO-ACE evaluation starting.")

    args = parse_args()

    if args.list_tasks:
        audio_tasks = list(mteb.get_tasks(modalities=["audio"]))
        supported_tasks = [task for task in audio_tasks if _is_audio_only_task(task)]
        excluded_tasks = len(audio_tasks) - len(supported_tasks)
        print(
            f"\nAudio-only MTEB tasks supported by SAO-ACE "
            f"({len(supported_tasks)}; excluded {excluded_tasks} cross-modal tasks):\n"
        )
        for t in sorted(supported_tasks, key=lambda x: x.metadata.name):
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

    # ------------------------------------------------------------------
    # Build task list
    # ------------------------------------------------------------------
    if args.tasks and args.benchmark:
        raise ValueError("--tasks and --benchmark are mutually exclusive.")

    if args.benchmark:
        benchmark_name = _resolve_benchmark_name(args.benchmark)
        tasks = _load_audio_only_benchmark_tasks(benchmark_name)
        log.info(f"Benchmark '{benchmark_name}': {len(tasks)} tasks.")
    elif args.tasks:
        tasks = _ensure_audio_only_tasks(
            list(mteb.get_tasks(tasks=args.tasks)),
            source="--tasks",
        )
        log.info(f"Explicit tasks: {[t.metadata.name for t in tasks]}")
    else:
        benchmark_name = _resolve_benchmark_name(_DEFAULT_BENCHMARK)
        tasks = _load_audio_only_benchmark_tasks(benchmark_name)
        log.info(f"Default {benchmark_name} suite: {len(tasks)} tasks.")

    if not tasks:
        raise ValueError("No audio tasks found. Check --tasks / --benchmark.")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    log.info(f"Building SAO-ACE encoder from: {ckpt_path}")
    model = SAOACEEncoder(
        model_name=str(ckpt_path),
        model_config_path=args.model_config,
        device=args.device,
        max_audio_length_seconds=args.max_audio_sec,
        pooling=args.pooling,
    )
    # Attach minimal ModelMeta (needed by some tasks)
    model.mteb_model_meta = build_model_meta(str(ckpt_path))

    # ------------------------------------------------------------------
    # Run evaluation (one task at a time for clear progress and ETA)
    # ------------------------------------------------------------------
    n_tasks = len(tasks)
    log.info(
        f"Starting MTEB evaluation → results in: {output_dir} "
        f"({n_tasks} tasks). Results are saved per task; if you interrupt, "
        "re-run the same command to resume (use --overwrite to re-run from scratch)."
    )
    results: list[Any] = []
    encode_kwargs = {"batch_size": args.batch_size}
    start_wall = time.perf_counter()
    for idx, task in enumerate(tasks):
        task_start = time.perf_counter()
        name = task.metadata.name
        log.info(f"\n{'='*60} Task {idx + 1}/{n_tasks}: {name} {'='*60}")
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
            avg_per_task = elapsed_total / done
            remaining_sec = avg_per_task * (n_tasks - done)
            remaining_min = remaining_sec / 60
            log.info(
                f"Task {done}/{n_tasks} done in {task_elapsed / 60:.1f} min. "
                f"Elapsed: {elapsed_total / 60:.1f} min. "
                f"Estimated remaining: ~{remaining_min:.0f} min."
            )
    log.info(f"\nAll {n_tasks} tasks completed in {(time.perf_counter() - start_wall) / 60:.1f} min.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n=== MAEB Results Summary ===")
    summary = {}
    for res in results:
        name = res.task_name
        scores = res.scores
        # Primary metric for each task type
        main_score = None
        for split_scores in scores.values():
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
        log.info(f"  {name:<55}: {main_score:.4f}" if main_score is not None else f"  {name}: N/A")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"ckpt": str(ckpt_path), "results": summary}, f, indent=2)
    log.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
