from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf

from ar_spectra.models.autoencoder import AutoEncoder
from ar_spectra.training_utils.initialization import resolve_auto_channels


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "conf").is_dir() and (candidate / "ar_spectra").is_dir():
            return candidate
    raise RuntimeError("Unable to locate repository root.")


def _infer_channels(train_kwargs: Dict[str, Any]) -> Tuple[int, int]:
    audio_channels = train_kwargs.get("audio_channels")
    if audio_channels is None:
        audio_channels = train_kwargs.get("channels")
    if audio_channels is None:
        stereo = bool(train_kwargs.get("stereo", True))
        audio_channels = 2 if stereo else 1
    audio_channels = int(audio_channels)

    if "spec_channels" in train_kwargs:
        model_channels = int(train_kwargs["spec_channels"])
    else:
        cac = bool(train_kwargs.get("cac", False))
        model_channels = audio_channels * (2 if cac else 1)
    return model_channels, audio_channels


def _build_autoencoder(model_cfg_path: Path, data_cfg_path: Path) -> AutoEncoder:
    model_cfg = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
    if "model" not in model_cfg:
        raise KeyError("Model configuration missing 'model' section.")
    model_spec = model_cfg["model"]

    data_cfg = OmegaConf.to_container(OmegaConf.load(data_cfg_path), resolve=True)
    train_spec = (data_cfg.get("train_dataset") or {})
    train_kwargs = (train_spec.get("kwargs") or {})
    model_channels, _ = _infer_channels(train_kwargs)

    resolve_auto_channels(model_spec, model_channels)
    return AutoEncoder.from_config(model_spec)


def test_epoch_029_checkpoint_loads_without_missing_keys() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    model_cfg_path = repo_root / "conf/model/SEANet_cplx_model.yaml"
    data_cfg_path = repo_root / "conf/data/data.yaml"
    ckpt_path = repo_root / "checkpoints/epoch_029.ckpt"

    assert model_cfg_path.is_file(), f"Model config not found: {model_cfg_path}"
    assert data_cfg_path.is_file(), f"Data config not found: {data_cfg_path}"
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    autoencoder = _build_autoencoder(model_cfg_path, data_cfg_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            k.replace("autoencoder.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("autoencoder.")
        }
    else:
        state_dict = checkpoint

    missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
    missing_keys = list(missing)
    unexpected_keys = list(unexpected)

    assert not missing_keys, f"Missing keys: {missing_keys}"
    assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
