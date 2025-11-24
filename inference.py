from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torchaudio
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.training_utils.initialization import (
    collate_stft,
    build_inference_dataloader,
    infer_channels_from_dataset_or_batch,
    instantiate_from_spec,
    load_checkpoint,
    prepare_dataset_spec,
    resolve_auto_channels,
    resolve_chunk_sizes,
)
from ar_spectra.training_utils.reproducibility import configure_reproducibility

console = Console()


def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")


def _to_plain_dict(node: Optional[Any]) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        data = OmegaConf.to_container(node, resolve=True)
        return copy.deepcopy(data) if isinstance(data, dict) else {}
    if isinstance(node, dict):
        return copy.deepcopy(node)
    return {}


def _ensure_dataset_seed(spec: Dict[str, Any], seed: int) -> Dict[str, Any]:
    copy_spec = copy.deepcopy(spec)
    kwargs = copy_spec.setdefault("kwargs", {}) or {}
    kwargs.setdefault("seed", int(seed))
    return copy_spec


def save_audio(waveform: torch.Tensor, target_path: Path, sample_rate: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2:
        pass
    else:
        raise ValueError(f"save_audio expects [C, T] or [T]; got shape {tuple(wav.shape)}")
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(str(target_path), wav, sample_rate)


def _encode_decode_batch(
    autoencoder: AutoEncoder,
    audio_batch: torch.Tensor,
    *,
    chunked: bool,
    chunk_size_samples: int,
    overlap_samples: int,
    pack_complex: bool,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latents, encode_info = autoencoder.encode_audio(
            audio_batch,
            chunked=chunked,
            chunk_size=chunk_size_samples,
            overlap_size=overlap_samples,
            pack_complex=pack_complex,
            debug=debug,
        )
        recon = autoencoder.decode_audio(
            latents,
            encode_info,
            chunked=chunked,
            pack_complex=pack_complex,
            debug=debug,
        )
    return recon, latents


def run_single_file_inference(
    autoencoder: AutoEncoder,
    *,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    cfg: DictConfig,
    hop_length: int,
    win_length: int,
    center_flag: bool,
    samples_per_latent: int,
    frames_per_latent: int,
    chunk_opts: Dict[str, Optional[Any]],
    chunked: bool,
    pack_complex: bool,
    debug: bool,
) -> None:
    if not input_path.is_file():
        err(f"Input audio not found: {input_path}")
        return

    wav, sr = torchaudio.load(str(input_path))

    target_sr = int(cfg.get("sample_rate", sr))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.to(device)
    if bool(cfg.get("mono", False)) and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.unsqueeze(0)

    max_seconds = cfg.get("max_seconds", None)
    max_frames = cfg.get("max_frames", None)
    if max_seconds is not None and max_frames is not None:
        err("Specify only one of max_seconds or max_frames.")
        return

    if max_seconds is not None:
        target_samples = int(round(float(max_seconds) * sr))
    elif max_frames is not None:
        pad = win_length // 2 if center_flag else 0
        target_samples = (int(max_frames) - 1) * hop_length + win_length - 2 * pad
        target_samples = max(target_samples, 0)
    else:
        target_samples = None

    if target_samples is not None and wav.shape[-1] > target_samples:
        if wav.shape[-1] - target_samples < hop_length:
            warn(f"Trimming input waveform from {wav.shape[-1]} to {target_samples} samples.")
        wav = wav[..., :target_samples]

    overlap_latent_cfg = chunk_opts.get("overlap_latent")
    overlap_latent = int(overlap_latent_cfg) if overlap_latent_cfg is not None else 0

    try:
        chunk_size_samples, overlap_samples, chunk_size_latent = resolve_chunk_sizes(
            sr=sr,
            hop_length=hop_length,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            segment_seconds=chunk_opts.get("segment_seconds"),
            segment_frames=chunk_opts.get("segment_frames"),
            chunk_size_latent_cfg=chunk_opts.get("chunk_size_latent"),
            overlap_latent=overlap_latent,
        )
    except ValueError as exc:
        err(str(exc))
        return

    ok(
        f"Chunking -> latent_steps={chunk_size_latent}, chunk_samples={chunk_size_samples}, "
        f"overlap_samples={overlap_samples}"
    )

    recon, latents = _encode_decode_batch(
        autoencoder,
        wav,
        chunked=chunked,
        chunk_size_samples=chunk_size_samples,
        overlap_samples=overlap_samples,
        pack_complex=pack_complex,
        debug=debug,
    )
    ok(f"Encoded audio -> latents shape={tuple(latents.shape)}")
    ok(f"Decoded latents -> waveform shape={tuple(recon.shape)}")

    recon = recon.squeeze(0)
    save_audio(recon, output_path, sr)
    ok(f"Saved reconstructed audio to {output_path}")


def run_dataset_inference(
    autoencoder: AutoEncoder,
    *,
    dataset: Any,
    dataset_cfg: Dict[str, Any],
    project_root: Path,
    device: torch.device,
    hop_length: int,
    samples_per_latent: int,
    frames_per_latent: int,
    pack_complex: bool,
    chunked: bool,
    chunk_opts: Dict[str, Optional[Any]],
    sample_rate: int,
    debug: bool,
    force_mono: bool,
) -> None:
    if dataset is None:
        err("Requested dataset for inference is not available.")
        return

    overlap_latent_cfg = chunk_opts.get("overlap_latent")
    overlap_latent = int(overlap_latent_cfg) if overlap_latent_cfg is not None else 0

    try:
        chunk_size_samples, overlap_samples, chunk_size_latent = resolve_chunk_sizes(
            sr=sample_rate,
            hop_length=hop_length,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            segment_seconds=chunk_opts.get("segment_seconds"),
            segment_frames=chunk_opts.get("segment_frames"),
            chunk_size_latent_cfg=chunk_opts.get("chunk_size_latent"),
            overlap_latent=overlap_latent,
        )
    except ValueError as exc:
        err(str(exc))
        return

    full_waveform_mode = bool(getattr(dataset, "full_waveform", False))

    batch_size = int(dataset_cfg.get("batch_size", 1) or 1)
    if full_waveform_mode and batch_size != 1:
        warn(
            f"full_waveform dataset requires batch_size=1; overriding provided batch_size={batch_size}"
        )
        batch_size = 1
    num_workers = int(dataset_cfg.get("num_workers", 0) or 0)

    shuffle = bool(dataset_cfg.get("shuffle", False))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))

    persistent_workers_cfg = dataset_cfg.get("persistent_workers", None)
    persistent_workers = bool(persistent_workers_cfg) if persistent_workers_cfg is not None else False

    prefetch_cfg = dataset_cfg.get("prefetch_factor", None)
    try:
        prefetch_factor = int(prefetch_cfg) if prefetch_cfg is not None else None
    except (TypeError, ValueError):
        warn(f"Ignoring invalid prefetch_factor={prefetch_cfg!r} for dataset inference.")
        prefetch_factor = None

    pin_memory_device = dataset_cfg.get("pin_memory_device", None)

    log_every = dataset_cfg.get("log_every", 1)
    if log_every is not None:
        log_every = max(1, int(log_every))

    save_waveforms = bool(dataset_cfg.get("save_waveforms", True))
    save_inputs = bool(dataset_cfg.get("save_input_audio", False))

    max_batches_cfg = dataset_cfg.get("max_batches", None)
    max_batches = int(max_batches_cfg) if max_batches_cfg is not None else None

    max_samples_cfg = dataset_cfg.get("max_samples", None)
    max_samples = int(max_samples_cfg) if max_samples_cfg is not None else None

    output_dir = project_root / dataset_cfg.get("output_dir", "runs/inference/dataset")
    file_prefix = str(dataset_cfg.get("file_prefix", "sample"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir: Optional[Path] = None
    if save_inputs:
        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "set_epoch"):
        try:
            dataset.set_epoch(0)
        except Exception:
            pass

    dataloader = build_inference_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory_device=pin_memory_device,
    )

    try:
        dataset_len = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        dataset_len = "unknown"

    ok(
        f"Running dataset inference: batch_size={batch_size}, chunk_size_latent={chunk_size_latent}, "
        f"dataset_size={dataset_len}"
    )

    name_counts: Dict[str, int] = {}
    # Track filename occurrences to avoid clobbering when the same stem repeats.

    def resolve_output_stem(sample_idx: int, src_path: Optional[str]) -> str:
        if src_path:
            stem = Path(src_path).stem
            base = f"{file_prefix}{stem}" if file_prefix else stem
        else:
            base = f"{file_prefix}_{sample_idx:06d}" if file_prefix else f"_{sample_idx:06d}"

        occur = name_counts.get(base, 0)
        name_counts[base] = occur + 1
        if occur > 0:
            return f"{base}_{occur:02d}"
        return base

    total_processed = 0
    total_saved = 0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            spec_batch, wav_batch, metadata_batch = batch
        else:
            spec_batch, wav_batch = batch  # type: ignore[misc]
            metadata_batch = None

        if spec_batch is not None:
            _ = spec_batch

        if max_batches is not None and batch_idx >= max_batches:
            break

        if max_samples is not None:
            remaining = max_samples - total_processed
            if remaining <= 0:
                break
            if remaining < wav_batch.size(0):
                wav_batch = wav_batch[:remaining]
                if metadata_batch is not None:
                    metadata_batch = metadata_batch[:remaining]

        metadata_paths: List[Optional[str]]
        if metadata_batch is None:
            metadata_paths = [None] * wav_batch.size(0)
        else:
            metadata_paths = list(metadata_batch)
            if len(metadata_paths) < wav_batch.size(0):
                metadata_paths = metadata_paths + [None] * (wav_batch.size(0) - len(metadata_paths))
            elif len(metadata_paths) > wav_batch.size(0):
                metadata_paths = metadata_paths[: wav_batch.size(0)]

        output_stems: List[str] = []
        for i in range(wav_batch.size(0)):
            src_path = metadata_paths[i]
            output_stems.append(resolve_output_stem(total_processed + i, src_path))

        wav_batch = wav_batch.to(device)

        if force_mono and wav_batch.size(1) > 1:
            wav_batch = wav_batch.mean(dim=1, keepdim=True)

        recon_batch, latents = _encode_decode_batch(
            autoencoder,
            wav_batch,
            chunked=chunked,
            chunk_size_samples=chunk_size_samples,
            overlap_samples=overlap_samples,
            pack_complex=pack_complex,
            debug=debug,
        )

        if log_every is not None and batch_idx % log_every == 0:
            ok(
                f"Batch {batch_idx}: latents shape={tuple(latents.shape)}, "
                f"recon shape={tuple(recon_batch.shape)}"
            )

        if save_inputs and input_dir is not None:
            for i in range(wav_batch.size(0)):
                sample_idx = total_processed + i
                save_audio(
                    wav_batch[i],
                    input_dir / f"{output_stems[i]}_in.wav",
                    sample_rate,
                )

        if save_waveforms:
            for i in range(recon_batch.size(0)):
                save_audio(
                    recon_batch[i],
                    output_dir / f"{output_stems[i]}.wav",
                    sample_rate,
                )
                total_saved += 1

        total_processed += wav_batch.size(0)

        if max_samples is not None and total_processed >= max_samples:
            break

    if total_processed == 0:
        warn("Dataset inference finished without processing any samples.")
    else:
        ok(
            f"Dataset inference completed: processed {total_processed} sample(s); "
            f"saved {total_saved} waveform(s) to {output_dir}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    def _as_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    seed = int(cfg.get("seed", 42))
    deterministic_flag = bool(cfg.get("deterministic", True))
    configure_reproducibility(seed, deterministic=deterministic_flag, warn=warn)

    if not cfg.get("model_config_path"):
        err("inference.yaml must define model_config_path")
        return
    if not cfg.get("checkpoint"):
        err("inference.yaml must define checkpoint")
        return

    model_cfg_path = project_root / Path(str(cfg.model_config_path))
    checkpoint_path = project_root / Path(str(cfg.checkpoint))

    if not model_cfg_path.is_file():
        err(f"Model config not found: {model_cfg_path}")
        return
    if not checkpoint_path.is_file():
        err(f"Checkpoint not found: {checkpoint_path}")
        return

    try:
        model_cfg = OmegaConf.load(model_cfg_path)
    except Exception as exc:
        err(f"Failed to load model configuration: {exc}")
        return

    model_cfg_container = OmegaConf.to_container(model_cfg, resolve=True)
    if not isinstance(model_cfg_container, dict):
        err("Model configuration must resolve to a dictionary.")
        return
    if "model" not in model_cfg_container:
        err("Model configuration must contain a 'model' section.")
        return

    model_spec = copy.deepcopy(model_cfg_container["model"])

    dataset_spec_base_raw = _to_plain_dict(cfg.get("dataset", None))
    if not dataset_spec_base_raw:
        err("inference.yaml must define a dataset specification under 'dataset'.")
        return

    dataset_cfg_block = cfg.get("dataset_inference", {}) or {}
    dataset_mode = bool(dataset_cfg_block.get("enabled", False))

    force_mono_global = bool(cfg.get("mono", False))
    segment_seconds_global = cfg.get("segment_seconds", None)
    segment_frames_global = cfg.get("segment_frames", None)
    chunk_size_latent_global = cfg.get("chunk_size", None)
    overlap_latent_global = cfg.get("overlap", 32)
    chunked_global = bool(cfg.get("chunked", True))
    debug_flag = bool(cfg.get("debug", False))

    dataset_kwargs_base = dataset_spec_base_raw.get("kwargs", {}) or {}
    default_sample_rate = _as_int(
        cfg.get("sample_rate"),
        _as_int(dataset_kwargs_base.get("sample_rate"), 44100),
    )
    default_hop_length = _as_int(dataset_kwargs_base.get("hop_length"), 512)

    prepared_dataset_spec_base, stft_params_base = prepare_dataset_spec(
        dataset_spec_base_raw,
        project_root=project_root,
        segment_seconds=segment_seconds_global,
        segment_frames=segment_frames_global,
        default_sample_rate=default_sample_rate,
        default_hop_length=default_hop_length,
        force_mono=force_mono_global,
    )

    dataset_spec_override_raw = _to_plain_dict(dataset_cfg_block.get("dataset", None))
    segment_seconds_dataset = dataset_cfg_block.get("segment_seconds", segment_seconds_global)
    segment_frames_dataset = dataset_cfg_block.get("segment_frames", segment_frames_global)
    force_mono_dataset = bool(dataset_cfg_block.get("mono", force_mono_global))

    prepared_dataset_spec_mode, stft_params_mode = prepare_dataset_spec(
        dataset_spec_override_raw or dataset_spec_base_raw,
        project_root=project_root,
        segment_seconds=segment_seconds_dataset,
        segment_frames=segment_frames_dataset,
        default_sample_rate=default_sample_rate,
        default_hop_length=default_hop_length,
        force_mono=force_mono_dataset,
    )

    stft_params = stft_params_mode or stft_params_base or {
        "sample_rate": default_sample_rate,
        "n_fft": 2048,
        "hop_length": default_hop_length,
        "win_length": default_hop_length * 4,
        "center": True,
        "normalized": False,
        "onesided": True,
    }

    n_fft = _as_int(stft_params.get("n_fft"), 2048)
    hop_length = max(1, _as_int(stft_params.get("hop_length"), n_fft // 4))
    win_length = _as_int(stft_params.get("win_length"), n_fft)
    center_flag = bool(stft_params.get("center", True))

    stft_common_kwargs: Dict[str, Any] = {"n_fft": n_fft, "hop_length": hop_length}
    if "win_length" in stft_params:
        stft_common_kwargs["win_length"] = win_length
    stft_common_kwargs.setdefault("win_length", win_length)

    if not center_flag:
        warn("Forcing center=True for inference to keep waveform length consistent.")
    stft_common_kwargs["center"] = True

    if "normalized" in stft_params:
        stft_common_kwargs["normalized"] = bool(stft_params.get("normalized", False))
    stft_common_kwargs.setdefault("normalized", False)

    if "onesided" in stft_params:
        stft_common_kwargs["onesided"] = bool(stft_params.get("onesided", True))
    stft_common_kwargs.setdefault("onesided", True)

    channel_dataset_spec = (
        prepared_dataset_spec_mode if dataset_mode and prepared_dataset_spec_mode else prepared_dataset_spec_base
    )
    if not channel_dataset_spec:
        err("Unable to resolve dataset specification for inferring model channels.")
        return

    channel_dataset_spec = _ensure_dataset_seed(channel_dataset_spec, seed)

    channel_cfg_stub = {"train_dataset": {"kwargs": channel_dataset_spec.get("kwargs", {}) or {}}}

    try:
        channel_dataset = instantiate_from_spec(copy.deepcopy(channel_dataset_spec))
    except Exception as exc:
        err(f"Failed to instantiate dataset for channel inference: {exc}")
        return

    channel_loader = DataLoader(
        channel_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_stft,
    )

    try:
        model_channels, audio_channels = infer_channels_from_dataset_or_batch(
            channel_cfg_stub,
            channel_dataset,
            channel_loader,
        )
    except Exception as exc:
        err(f"Failed to infer channel dimensions from dataset: {exc}")
        return
    finally:
        del channel_loader

    ok(
        f"Inferred channels from dataset: model_channels={model_channels}, audio_channels={audio_channels}"
    )

    del channel_dataset

    if force_mono_global:
        warn("Mono inference enabled; dataset stereo flag forced to False where applicable.")

    resolve_auto_channels(model_spec, model_channels)
    autoenc = AutoEncoder.from_config(model_spec)
    ok(
        f"Built AutoEncoder for inference (model_channels={model_channels}, "
        f"audio_channels={audio_channels})"
    )

    autoenc = load_checkpoint(autoenc, checkpoint_path)
    ok(f"Loaded checkpoint from {checkpoint_path}")
    autoenc.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc.to(device)

    try:
        autoenc.set_stft_config(stft_common_kwargs)
    except Exception as exc:
        err(f"Invalid STFT configuration for inference: {exc}")
        return

    samples_per_latent = autoenc.infer_downsampling_ratio(hop_length=hop_length)
    frames_per_latent = max(1, samples_per_latent // hop_length)

    is_complex_model = bool(getattr(autoenc.encoder, "is_complex", False))
    pack_complex = not is_complex_model
    if pack_complex:
        ok("Using real/imag channel packing for complex spectrograms.")
    else:
        warn("Encoder marked complex-capable; skipping real/imag packing.")

    dataset_sample_rate = _as_int(
        (stft_params_mode or stft_params).get("sample_rate"),
        default_sample_rate,
    )

    executed = False

    if dataset_mode:
        inference_dataset_spec = prepared_dataset_spec_mode or prepared_dataset_spec_base
        if not inference_dataset_spec:
            err("dataset_inference.enabled but no dataset specification was provided.")
        else:
            chunked_dataset = bool(dataset_cfg_block.get("chunked", chunked_global))
            chunk_opts_dataset = {
                "segment_seconds": segment_seconds_dataset,
                "segment_frames": segment_frames_dataset,
                "chunk_size_latent": dataset_cfg_block.get("chunk_size", chunk_size_latent_global),
                "overlap_latent": dataset_cfg_block.get("overlap", overlap_latent_global),
            }

            inference_dataset_spec = _ensure_dataset_seed(inference_dataset_spec, seed)

            try:
                dataset_obj = instantiate_from_spec(copy.deepcopy(inference_dataset_spec))
            except Exception as exc:
                err(f"Failed to instantiate inference dataset: {exc}")
                dataset_obj = None

            if dataset_obj is not None:
                if hasattr(dataset_obj, "enable_return_paths"):
                    try:
                        dataset_obj.enable_return_paths()
                    except Exception:
                        warn("Could not enable return_paths on dataset; falling back to index-based naming.")
                elif hasattr(dataset_obj, "return_paths"):
                    setattr(dataset_obj, "return_paths", True)

                run_dataset_inference(
                    autoenc,
                    dataset=dataset_obj,
                    dataset_cfg=dataset_cfg_block,
                    project_root=project_root,
                    device=device,
                    hop_length=hop_length,
                    samples_per_latent=samples_per_latent,
                    frames_per_latent=frames_per_latent,
                    pack_complex=pack_complex,
                    chunked=chunked_dataset,
                    chunk_opts=chunk_opts_dataset,
                    sample_rate=dataset_sample_rate,
                    debug=debug_flag,
                    force_mono=force_mono_dataset,
                )
                executed = True

    input_wav = cfg.get("input_wav", None)
    output_wav = cfg.get("output_wav", None)
    if input_wav is not None and output_wav is not None:
        input_path = project_root / Path(str(input_wav))
        output_path = project_root / Path(str(output_wav))
        chunk_opts_global = {
            "segment_seconds": segment_seconds_global,
            "segment_frames": segment_frames_global,
            "chunk_size_latent": chunk_size_latent_global,
            "overlap_latent": overlap_latent_global,
        }
        run_single_file_inference(
            autoencoder=autoenc,
            input_path=input_path,
            output_path=output_path,
            device=device,
            cfg=cfg,
            hop_length=hop_length,
            win_length=win_length,
            center_flag=center_flag,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            chunk_opts=chunk_opts_global,
            chunked=chunked_global,
            pack_complex=pack_complex,
            debug=debug_flag,
        )
        executed = True

    if not executed:
        warn("Nothing to do. Provide input_wav/output_wav or enable dataset_inference.enabled.")


if __name__ == "__main__":
    main()
