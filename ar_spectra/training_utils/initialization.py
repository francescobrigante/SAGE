from __future__ import annotations

from dataclasses import dataclass
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
    """Default collate function for STFT datasets returning (S, wav).

    Ensures all tensors in the batch have identical shapes before stacking.
    """

    Ss, wavs = zip(*batch)
    s0 = Ss[0].shape
    w0 = wavs[0].shape
    assert all(x.shape == s0 for x in Ss), f"STFT shapes differ: {[x.shape for x in Ss]}"
    assert all(x.shape == w0 for x in wavs), f"Wav shapes differ: {[x.shape for x in wavs]}"
    return torch.stack(Ss, 0), torch.stack(wavs, 0)


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

    train_ds = instantiate_from_spec(cfg["train_dataset"])
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
    eval_ds = instantiate_from_spec(eval_spec) if eval_spec else None
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
