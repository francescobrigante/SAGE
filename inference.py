from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import copy
import math

import hydra
import torch
import torchaudio
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader

from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.training_utils.initialization import (
    build_datasets_and_loaders,
    collate_stft,
    resolve_auto_channels,
    instantiate_from_spec,
)

console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")


def _extract_autoencoder_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the parameters that belong to the AutoEncoder module.

    Training checkpoints created via ``AutoencoderTrainingWrapper`` store
    weights under ``engine.autoencoder.*`` while direct exports may already
    use ``autoencoder.*`` or plain module prefixes (``encoder.``, ``decoder.``).
    This helper normalizes all those layouts so inference consistently receives
    the bare autoencoder state dict.
    """

    prefixes = ("engine.autoencoder.", "autoencoder.")
    extracted: Dict[str, Any] = {}

    for key, value in state_dict.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                matched = True
                break
        if not matched and (key.startswith("encoder.") or key.startswith("decoder.") or key.startswith("bottleneck.")):
            extracted[key] = value

    return extracted or state_dict


def load_checkpoint(autoencoder: AutoEncoder, ckpt_path: Path) -> AutoEncoder:
    """Load state dict from a Lightning checkpoint or plain state dict."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = _extract_autoencoder_state(ckpt["state_dict"])
        missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
        if missing:
            warn(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            warn(f"Unexpected keys when loading checkpoint: {unexpected}")
    else:
        autoencoder.load_state_dict(ckpt)
    return autoencoder


def save_audio(wav: torch.Tensor, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


def _to_plain_dict(config_section: Any) -> Dict[str, Any]:
    """Return a deep-copied plain dict from a DictConfig or mapping."""

    if config_section is None:
        return {}
    if isinstance(config_section, DictConfig):
        return OmegaConf.to_container(config_section, resolve=True)  # type: ignore[arg-type]
    if isinstance(config_section, dict):
        return copy.deepcopy(config_section)
    return {}


def _prepare_dataset_spec(
    raw_spec: Dict[str, Any],
    *,
    project_root: Path,
    segment_seconds: Optional[Any],
    segment_frames: Optional[Any],
    default_sample_rate: int,
    default_hop_length: int,
    force_mono: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize dataset spec, resolving paths and temporal overrides."""

    if not raw_spec:
        return {}, {}

    prepared = copy.deepcopy(raw_spec)
    kwargs = prepared.setdefault("kwargs", {}) or {}

    audio_dir = kwargs.get("audio_dir", None)
    if audio_dir is not None:
        audio_path = Path(audio_dir)
        if not audio_path.is_absolute():
            kwargs["audio_dir"] = str((project_root / audio_path).resolve())
        else:
            kwargs["audio_dir"] = str(audio_path)

    if force_mono and kwargs.get("stereo", True):
        kwargs["stereo"] = False

    sample_rate = int(kwargs.get("sample_rate", default_sample_rate))
    hop_length = int(kwargs.get("hop_length", default_hop_length if default_hop_length > 0 else 512))

    full_waveform_flag = bool(kwargs.get("full_waveform", False))

    if segment_frames is not None:
        kwargs["target_frames"] = max(2, int(segment_frames))
        full_waveform_flag = False
    elif segment_seconds is not None:
        frames = int(math.ceil(float(segment_seconds) * sample_rate / max(1, hop_length))) + 1
        kwargs["target_frames"] = max(2, frames)
        full_waveform_flag = False
    elif kwargs.get("target_frames") is not None:
        kwargs["target_frames"] = max(2, int(kwargs.get("target_frames")))
        full_waveform_flag = False

    if kwargs.get("target_frames") is None and not full_waveform_flag:
        full_waveform_flag = True

    if full_waveform_flag:
        kwargs.pop("target_frames", None)
        kwargs["full_waveform"] = True
    else:
        kwargs["full_waveform"] = False

    kwargs["sample_rate"] = sample_rate
    kwargs["hop_length"] = hop_length
    if "win_length" in kwargs:
        kwargs["win_length"] = int(kwargs.get("win_length"))
    if "n_fft" in kwargs:
        kwargs["n_fft"] = int(kwargs.get("n_fft"))

    return prepared, kwargs


def _build_inference_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    pin_memory_device: Optional[str] = None,
) -> DataLoader:
    """Construct a ``DataLoader`` suitable for inference batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer for inference")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": collate_stft,
    }

    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor) if prefetch_factor is not None else 2
    else:
        loader_kwargs["persistent_workers"] = False

    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = str(pin_memory_device)

    return DataLoader(dataset, **loader_kwargs)


def _encode_decode_batch(
    autoencoder: AutoEncoder,
    audio_batch: torch.Tensor,
    *,
    chunked: bool,
    chunk_size_samples: int,
    overlap_samples: int,
    pack_complex: bool,
    stft_kwargs: Dict[str, Any],
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode and decode a batch of waveforms, returning reconstructions and latents."""

    with torch.no_grad():
        latents, encode_info = autoencoder.encode_audio(
            audio_batch,
            chunked=chunked,
            chunk_size=chunk_size_samples,
            overlap_size=overlap_samples,
            pack_complex=pack_complex,
            debug=debug,
            **stft_kwargs,
        )
        recon = autoencoder.decode_audio(
            latents,
            encode_info,
            chunked=chunked,
            pack_complex=pack_complex,
            debug=debug,
            **stft_kwargs,
        )
    return recon, latents


def _resolve_chunk_sizes(
    *,
    sr: int,
    hop_length: int,
    samples_per_latent: int,
    frames_per_latent: int,
    segment_seconds: Optional[Any],
    segment_frames: Optional[Any],
    chunk_size_latent_cfg: Optional[Any],
    overlap_latent: int,
) -> Tuple[int, int, int]:
    """Resolve chunk sizing options returning ``(chunk_samples, overlap_samples, chunk_latent)``."""

    options_selected = [segment_seconds is not None, segment_frames is not None, chunk_size_latent_cfg is not None]
    if sum(options_selected) > 1:
        raise ValueError("Specify only one among segment_seconds, segment_frames, chunk_size.")

    if segment_seconds is not None:
        segment_samples = int(round(float(segment_seconds) * sr))
        chunk_size_latent = max(1, int(round(segment_samples / max(1, samples_per_latent))))
    elif segment_frames is not None:
        segment_frames = int(segment_frames)
        chunk_size_latent = max(1, int(round(segment_frames / max(1, frames_per_latent))))
    elif chunk_size_latent_cfg is not None:
        chunk_size_latent = max(1, int(chunk_size_latent_cfg))
    else:
        chunk_size_latent = 128

    chunk_size_samples = chunk_size_latent * samples_per_latent
    overlap_samples = max(0, int(overlap_latent)) * samples_per_latent
    return chunk_size_samples, overlap_samples, chunk_size_latent


def run_single_file_inference(
    autoencoder: AutoEncoder,
    *,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    cfg: DictConfig,
    stft_common_kwargs: Dict[str, Any],
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
    """Run inference on a single audio file and store the reconstruction."""

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

    wav = wav.unsqueeze(0)  # [1, C, T]

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

    try:
        chunk_size_samples, overlap_samples, chunk_size_latent = _resolve_chunk_sizes(
            sr=sr,
            hop_length=hop_length,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            segment_seconds=chunk_opts.get("segment_seconds"),
            segment_frames=chunk_opts.get("segment_frames"),
            chunk_size_latent_cfg=chunk_opts.get("chunk_size_latent"),
            overlap_latent=int(chunk_opts.get("overlap_latent", 0)),
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
        stft_kwargs=stft_common_kwargs,
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
    stft_common_kwargs: Dict[str, Any],
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
    """Iterate over a dataset, reconstruct each batch, and save results to disk."""

    if dataset is None:
        err("Requested dataset for inference is not available.")
        return

    try:
        chunk_size_samples, overlap_samples, chunk_size_latent = _resolve_chunk_sizes(
            sr=sample_rate,
            hop_length=hop_length,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            segment_seconds=chunk_opts.get("segment_seconds"),
            segment_frames=chunk_opts.get("segment_frames"),
            chunk_size_latent_cfg=chunk_opts.get("chunk_size_latent"),
            overlap_latent=int(chunk_opts.get("overlap_latent", 0)),
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

    dataloader = _build_inference_dataloader(
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

    total_processed = 0
    total_saved = 0

    for batch_idx, (spec_batch, wav_batch) in enumerate(dataloader):
        if spec_batch is not None:
            _ = spec_batch  # spec not used during inference, keep semantic parity

        if max_batches is not None and batch_idx >= max_batches:
            break

        if max_samples is not None:
            remaining = max_samples - total_processed
            if remaining <= 0:
                break
            if remaining < wav_batch.size(0):
                wav_batch = wav_batch[:remaining]

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
            stft_kwargs=stft_common_kwargs,
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
                save_audio(wav_batch[i], input_dir / f"{file_prefix}_{sample_idx:06d}_in.wav", sample_rate)

        if save_waveforms:
            for i in range(recon_batch.size(0)):
                sample_idx = total_processed + i
                save_audio(recon_batch[i], output_dir / f"{file_prefix}_{sample_idx:06d}.wav", sample_rate)
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
    """Inference entry point supporting single files and full datasets."""

    project_root = Path(get_original_cwd())
    model_cfg = OmegaConf.load(project_root / cfg.model_config_path)
    data_cfg = OmegaConf.load(project_root / cfg.data_config_path)

    unified: Dict[str, Any] = OmegaConf.to_container(
        OmegaConf.merge(model_cfg, data_cfg), resolve=True
    )

    dataset_cfg = cfg.get("dataset_inference", {}) or {}
    dataset_mode = bool(dataset_cfg.get("enabled", False))

    if bool(cfg.get("mono", False)):
        for ds_key in ("train_dataset", "eval_dataset"):
            spec = unified.get(ds_key, None)
            if isinstance(spec, dict):
                kwargs = spec.get("kwargs", {}) or {}
                if kwargs.get("stereo", True) is True:
                    kwargs["stereo"] = False
                    spec["kwargs"] = kwargs
                    unified[ds_key] = spec
        warn("Patched dataset config to mono (stereo=False) for inference; model channels will be 1.")

    data_init = build_datasets_and_loaders(unified)
    model_spec = unified["model"]
    resolve_auto_channels(model_spec, data_init.model_channels)
    autoenc = AutoEncoder.from_config(model_spec)
    ok(
        f"Built AutoEncoder for inference (model_channels={data_init.model_channels}, "
        f"audio_channels={data_init.audio_channels})"
    )

    ckpt_path = project_root / cfg.checkpoint
    if not ckpt_path.is_file():
        err(f"Checkpoint not found: {ckpt_path}")
        return

    autoenc = load_checkpoint(autoenc, ckpt_path)
    autoenc.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc.to(device)

    segment_seconds_global = cfg.get("segment_seconds", None)
    segment_frames_global = cfg.get("segment_frames", None)
    chunk_size_latent_global = cfg.get("chunk_size", None)
    overlap_latent_global = int(cfg.get("overlap", 32))
    chunked_global = bool(cfg.get("chunked", True))
    debug_flag = bool(cfg.get("debug", False))
    force_mono_global = bool(cfg.get("mono", False))

    dataset_spec_base = _to_plain_dict(cfg.get("dataset", None))
    if not dataset_spec_base:
        fallback_spec: Dict[str, Any] = {}
        for key in ("eval_dataset", "train_dataset"):
            candidate = unified.get(key, None)
            if isinstance(candidate, dict) and candidate:
                fallback_spec = copy.deepcopy(candidate)
                break
        dataset_spec_base = fallback_spec

    dataset_spec_override = _to_plain_dict(dataset_cfg.get("dataset", None))
    if dataset_spec_override:
        dataset_spec_base = dataset_spec_override

    segment_seconds_for_spec = dataset_cfg.get("segment_seconds", segment_seconds_global)
    segment_frames_for_spec = dataset_cfg.get("segment_frames", segment_frames_global)
    force_mono_dataset = bool(dataset_cfg.get("mono", force_mono_global))

    dataset_default_sr = (
        dataset_spec_base.get("kwargs", {}).get("sample_rate", None) if dataset_spec_base else None
    )
    dataset_default_hop = (
        dataset_spec_base.get("kwargs", {}).get("hop_length", None) if dataset_spec_base else None
    )
    default_sample_rate = int(cfg.get("sample_rate", dataset_default_sr or 44100))
    default_hop_length = int(dataset_default_hop) if dataset_default_hop is not None else 512

    prepared_dataset_spec, stft_params = _prepare_dataset_spec(
        dataset_spec_base,
        project_root=project_root,
        segment_seconds=segment_seconds_for_spec,
        segment_frames=segment_frames_for_spec,
        default_sample_rate=default_sample_rate,
        default_hop_length=default_hop_length,
        force_mono=force_mono_dataset,
    )

    if not stft_params:
        stft_params = {
            "sample_rate": default_sample_rate,
            "n_fft": 2048,
            "hop_length": default_hop_length,
            "win_length": default_hop_length * 4,
            "center": True,
            "normalized": False,
            "onesided": True,
            "target_frames": 128,
        }

    n_fft = int(stft_params.get("n_fft", 2048))
    hop_length = int(stft_params.get("hop_length", n_fft // 4))
    win_length = int(stft_params.get("win_length", n_fft))
    center_flag = bool(stft_params.get("center", True))

    stft_common_kwargs: Dict[str, Any] = {"n_fft": n_fft, "hop_length": hop_length}
    if "win_length" in stft_params:
        stft_common_kwargs["win_length"] = win_length
    if "center" in stft_params:
        stft_common_kwargs["center"] = center_flag
    if "normalized" in stft_params:
        stft_common_kwargs["normalized"] = bool(stft_params.get("normalized", False))
    if "onesided" in stft_params:
        stft_common_kwargs["onesided"] = bool(stft_params.get("onesided", True))

    dataset_sample_rate = int(stft_params.get("sample_rate", default_sample_rate))

    samples_per_latent = autoenc.infer_downsampling_ratio(hop_length=hop_length)
    frames_per_latent = max(1, samples_per_latent // hop_length)

    is_complex_model = bool(getattr(autoenc.encoder, "is_complex", False))
    pack_complex = not is_complex_model
    if pack_complex:
        ok("Using real/imag channel packing for complex spectrograms.")
    else:
        warn("Encoder marked complex-capable; skipping real/imag packing.")

    executed = False

    if dataset_mode:
        if not prepared_dataset_spec:
            err("dataset_inference.enabled but no dataset specification was provided.")
        else:
            segment_seconds_ds = dataset_cfg.get("segment_seconds", segment_seconds_for_spec)
            segment_frames_ds = dataset_cfg.get("segment_frames", segment_frames_for_spec)
            chunk_size_latent_ds = dataset_cfg.get("chunk_size", chunk_size_latent_global)
            overlap_latent_ds_val = dataset_cfg.get("overlap", None)
            if overlap_latent_ds_val is not None:
                overlap_latent_ds = int(overlap_latent_ds_val)
            else:
                overlap_latent_ds = overlap_latent_global
            chunked_override = dataset_cfg.get("chunked", None)
            chunked_dataset = chunked_global if chunked_override is None else bool(chunked_override)

            try:
                dataset_obj = instantiate_from_spec(prepared_dataset_spec)
            except Exception as exc:
                err(f"Failed to instantiate inference dataset: {exc}")
                dataset_obj = None

            if dataset_obj is not None:
                run_dataset_inference(
                    autoenc,
                    dataset=dataset_obj,
                    dataset_cfg=dataset_cfg,
                    project_root=project_root,
                    device=device,
                    stft_common_kwargs=stft_common_kwargs,
                    hop_length=hop_length,
                    samples_per_latent=samples_per_latent,
                    frames_per_latent=frames_per_latent,
                    pack_complex=pack_complex,
                    chunked=chunked_dataset,
                    chunk_opts={
                        "segment_seconds": segment_seconds_ds,
                        "segment_frames": segment_frames_ds,
                        "chunk_size_latent": chunk_size_latent_ds,
                        "overlap_latent": overlap_latent_ds,
                    },
                    sample_rate=dataset_sample_rate,
                    debug=debug_flag,
                    force_mono=force_mono_dataset,
                )
                executed = True

    input_wav = cfg.get("input_wav", None)
    output_wav = cfg.get("output_wav", None)
    if input_wav is not None and output_wav is not None:
        input_path = project_root / input_wav
        output_path = project_root / output_wav
        run_single_file_inference(
            autoenc,
            input_path=input_path,
            output_path=output_path,
            device=device,
            cfg=cfg,
            stft_common_kwargs=stft_common_kwargs,
            hop_length=hop_length,
            win_length=win_length,
            center_flag=center_flag,
            samples_per_latent=samples_per_latent,
            frames_per_latent=frames_per_latent,
            chunk_opts={
                "segment_seconds": segment_seconds_global,
                "segment_frames": segment_frames_global,
                "chunk_size_latent": chunk_size_latent_global,
                "overlap_latent": overlap_latent_global,
            },
            chunked=chunked_global,
            pack_complex=pack_complex,
            debug=debug_flag,
        )
        executed = True

    if not executed:
        warn("Nothing to do. Provide input_wav/output_wav or enable dataset_inference.enabled.")


if __name__ == "__main__":
    main()
