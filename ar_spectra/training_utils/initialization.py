from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ar_spectra.models.autoencoder import AutoEncoder

if TYPE_CHECKING:  # only for type hints, avoids runtime circular import
    from ar_spectra.training_utils.autoencoders import AutoencoderTrainingWrapper


def instantiate_from_spec(spec: Dict[str, Any]) -> Any:
    """Instantiate a Python object from a config spec.

    The spec format is shared across training and inference code::

        {
            "class": "pkg.mod.Class",
            "args": [...],        # optional positional args
            "kwargs": { ... }     # optional keyword args
        }
    """

    from importlib import import_module

    if "class" not in spec:
        raise ValueError("spec is missing the 'class' key.")
    module_path, class_name = spec["class"].rsplit(".", 1)
    module = import_module(module_path)
    cls = getattr(module, class_name)
    args = spec.get("args", []) or []
    kwargs = spec.get("kwargs", {}) or {}
    return cls(*args, **kwargs)


@dataclass
class DataInitResult:
    """Holds dataset and dataloader objects used for training and evaluation.

    This is the single source of truth for how datasets and dataloaders are
    instantiated from the unified config dictionary used in training.
    """

    train_dataset: Any
    train_dataloader: DataLoader
    eval_dataset: Optional[Any]
    eval_dataloader: Optional[DataLoader]
    model_channels: int
    audio_channels: int


def collate_stft(batch):
    """Default collate function for STFT datasets returning (S, wav[, meta]).

    Supports optional metadata (e.g., source paths) by forwarding them as a list
    without altering legacy behaviour when metadata is absent.
    """

    if not batch:
        raise ValueError("collate_stft received an empty batch")

    first = batch[0]
    if not isinstance(first, tuple):
        raise TypeError(f"collate_stft expects tuples, got {type(first).__name__}")

    if len(first) == 3:
        Ss, wavs, metas = zip(*batch)
    elif len(first) == 2:
        Ss, wavs = zip(*batch)
        metas = None
    else:
        raise ValueError(f"collate_stft expects 2 or 3 items per sample, got {len(first)}")

    first_spec = Ss[0]
    if first_spec is None:
        assert all(x is None for x in Ss), "Mixed spectrogram/None batches are not supported."
        w0 = wavs[0].shape
        assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
        stacked_specs = None
    else:
        s0 = first_spec.shape
        w0 = wavs[0].shape
        assert all(x.shape == s0 for x in Ss), f"STFT shapes differ: {[x.shape for x in Ss]}"
        assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
        stacked_specs = torch.stack(Ss, 0)

    stacked_wavs = torch.stack(wavs, 0)

    if metas is not None:
        return stacked_specs, stacked_wavs, list(metas)
    return stacked_specs, stacked_wavs


def infer_channels_from_dataset_or_batch(
    cfg: Dict[str, Any], train_dataset: Any, train_loader: DataLoader
) -> Tuple[int, int]:
    """Infer model and audio channels from dataset or a single batch.

    This function centralizes the logic used to resolve "auto" placeholders
    for encoder/decoder configuration. It can be reused by inference code to
    obtain consistent channel information.
    """

    ds_spec_ch = getattr(train_dataset, "spec_channels", None)
    ds_audio_ch = getattr(train_dataset, "audio_channels", None)
    cac = bool(cfg.get("train_dataset", {}).get("kwargs", {}).get("cac", False))

    if ds_spec_ch is not None and ds_audio_ch is not None:
        model_channels = int(ds_spec_ch)
        audio_channels = int(ds_audio_ch)
        return model_channels, audio_channels

    sample = next(iter(train_loader))
    sp_reals, _ = sample
    if sp_reals.dim() == 3:
        Cx, _, _ = sp_reals.shape
    elif sp_reals.dim() == 4:
        _, Cx, _, _ = sp_reals.shape
    else:
        raise RuntimeError(f"Unexpected sp_reals shape: {tuple(sp_reals.shape)}")

    model_channels = int(Cx)
    audio_channels = (model_channels // 2) if cac else model_channels
    return model_channels, audio_channels


def resolve_auto_channels(model_cfg: Dict[str, Any], model_channels: int) -> None:
    """Resolve "auto" placeholders in encoder/decoder kwargs in-place.

    The function mutates ``model_cfg`` so that all channel-related parameters
    needed to build the model are concrete integers.
    """

    enc_kwargs = model_cfg.setdefault("encoder", {}).setdefault("kwargs", {})
    dec_kwargs = model_cfg.setdefault("decoder", {}).setdefault("kwargs", {})

    def _set_auto(d: dict, key: str, value: int) -> None:
        v = d.get(key, None)
        if (v is None) or (isinstance(v, str) and v.lower() == "auto"):
            d[key] = int(value)

    _set_auto(enc_kwargs, "input_size", model_channels)
    _set_auto(dec_kwargs, "channels", model_channels)
    if "out_channels" in dec_kwargs:
        _set_auto(dec_kwargs, "out_channels", model_channels)


def build_datasets_and_loaders(cfg: Dict[str, Any]) -> DataInitResult:
    """Instantiate datasets and dataloaders from unified config.

    This is the canonical entry point for training code that needs
    ``train_dataset``, ``eval_dataset`` and their corresponding dataloaders.
    """

    seed_value = int(cfg.get("seed", 42))
    train_spec = copy.deepcopy(cfg["train_dataset"])
    train_kwargs = train_spec.setdefault("kwargs", {}) or {}
    train_kwargs.setdefault("seed", seed_value)
    train_ds = instantiate_from_spec(train_spec)
    dl_cfg = cfg.get("train_dataloader", {}) or {}
    num_workers = int(dl_cfg.get("num_workers", 8))

    train_batch_size = int(dl_cfg.get("batch_size", 8))
    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        drop_last=True,
        persistent_workers=(dl_cfg.get("persistent_workers", False) if num_workers > 0 else False),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
        collate_fn=collate_stft,
    )

    eval_spec = cfg.get("eval_dataset", None)
    if eval_spec:
        eval_spec = copy.deepcopy(eval_spec)
        eval_kwargs = eval_spec.setdefault("kwargs", {}) or {}
        eval_kwargs.setdefault("seed", seed_value)
        eval_ds = instantiate_from_spec(eval_spec)
    else:
        eval_ds = None
    dl_eval_cfg = cfg.get("eval_dataloader", {}) or {}

    if eval_ds is not None:
        eval_batch_size = int(dl_eval_cfg.get("batch_size", train_batch_size))
        eval_dl = DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=bool(dl_eval_cfg.get("pin_memory", False)),
            shuffle=bool(dl_eval_cfg.get("shuffle", False)),
            drop_last=False,
            persistent_workers=(dl_eval_cfg.get("persistent_workers", False) if num_workers > 0 else False),
            prefetch_factor=int(dl_eval_cfg.get("prefetch_factor", 8)) if num_workers > 0 else None,
            collate_fn=collate_stft,
        )
    else:
        eval_dl = None

    model_channels, audio_channels = infer_channels_from_dataset_or_batch(cfg, train_ds, train_dl)

    return DataInitResult(
        train_dataset=train_ds,
        train_dataloader=train_dl,
        eval_dataset=eval_ds,
        eval_dataloader=eval_dl,
        model_channels=model_channels,
        audio_channels=audio_channels,
    )


def build_autoencoder_from_cfg(cfg: Dict[str, Any]) -> Tuple[AutoEncoder, int, int]:
    """Build ``AutoEncoder`` and infer channels from unified config.

    Returns the instantiated model together with ``(model_channels, audio_channels)``
    so the caller can use the same information for training or inference code.
    """

    data_init = build_datasets_and_loaders(cfg)
    model_cfg = cfg["model"]
    resolve_auto_channels(model_cfg, data_init.model_channels)
    autoenc = AutoEncoder.from_config(model_cfg)
    return autoenc, data_init.model_channels, data_init.audio_channels


def build_training_wrapper_from_cfg(cfg: Dict[str, Any]) -> Tuple["AutoencoderTrainingWrapper", DataInitResult]:
    """Factory that builds the full training wrapper from unified config.

    This concentrates how the autoencoder and its Lightning wrapper are
    instantiated so that the same logic can be reused for inference scripts.
    """

    data_init = build_datasets_and_loaders(cfg)
    model_cfg = cfg["model"]
    resolve_auto_channels(model_cfg, data_init.model_channels)

    autoenc = AutoEncoder.from_config(model_cfg)
    optimizer_spec = cfg.get("optimizer", None)
    scheduler_spec = cfg.get("scheduler", None)

    from ar_spectra.training_utils.autoencoders import AutoencoderTrainingWrapper

    wrapper = AutoencoderTrainingWrapper(
        autoencoder=autoenc,
        sample_rate=int(cfg["train_dataset"]["kwargs"].get("sample_rate", 44100)),
        audio_channels=int(data_init.audio_channels),
        loss_config=cfg.get("loss_config", None),
        eval_loss_config=cfg.get("eval_loss_config", None),
        optimizer_configs=None,
        warmup_steps=int(cfg.get("trainer", {}).get("warmup_steps", 0)),
        warmup_mode=str(cfg.get("trainer", {}).get("warmup_mode", "adv")),
        encoder_freeze_on_warmup=bool(cfg.get("trainer", {}).get("encoder_freeze_on_warmup", False)),
        force_input_mono=bool(cfg.get("model", {}).get("autoencoder", {}).get("force_input_mono", False)),
        latent_mask_ratio=float(cfg.get("model", {}).get("autoencoder", {}).get("latent_mask_ratio", 0.0)),
        teacher_model=None,
        stft_params=cfg.get("train_dataset", {}).get("kwargs", {}),
        optimizer_spec=optimizer_spec,
        scheduler_spec=scheduler_spec,
        pre_transform_spec=cfg.get("pre_transform", None),
    )

    return wrapper, data_init


def prepare_dataset_spec(
    raw_spec: Dict[str, Any],
    *,
    project_root: Path,
    segment_seconds: Optional[Any],
    segment_frames: Optional[Any],
    default_sample_rate: int,
    default_hop_length: int,
    force_mono: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize a dataset spec for STFT-based pipelines.

    Returns the patched spec together with the resolved STFT parameters
    that downstream encode/decode utilities can reuse.
    """

    if not raw_spec:
        return {}, {}

    prepared = copy.deepcopy(raw_spec)
    kwargs = prepared.setdefault("kwargs", {}) or {}

    audio_dir = kwargs.get("audio_dir")
    if audio_dir is not None:
        audio_path = Path(audio_dir)
        kwargs["audio_dir"] = str((project_root / audio_path).resolve()) if not audio_path.is_absolute() else str(audio_path)

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


def build_inference_dataloader(
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
    """Create a DataLoader configured for inference workloads."""

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


def resolve_chunk_sizes(
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
    """Resolve chunk sizing options returning (chunk_samples, overlap_samples, chunk_latent)."""

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


def _extract_autoencoder_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the parameters that belong to the AutoEncoder module."""

    prefixes = ("engine.autoencoder.", "autoencoder.")
    extracted: Dict[str, Any] = {}

    for key, value in state_dict.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                matched = True
                break
        if not matched and (
            key.startswith("encoder.")
            or key.startswith("decoder.")
            or key.startswith("bottleneck.")
        ):
            extracted[key] = value

    return extracted or state_dict


def load_checkpoint(autoencoder: AutoEncoder, ckpt_path: Path) -> AutoEncoder:
    """Load state dict from a Lightning checkpoint or plain state dict."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = _extract_autoencoder_state(ckpt["state_dict"])
        missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.warn(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys when loading checkpoint: {unexpected}")
    else:
        autoencoder.load_state_dict(ckpt)
    return autoencoder
