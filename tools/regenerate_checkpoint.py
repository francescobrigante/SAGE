#!/usr/bin/env python3
"""Interactive utility to retrofit checkpoints with complete inference metadata.

Given a Lightning checkpoint and a model YAML, the script rebuilds the
``inference_config`` block expected by :mod:`ar_spectra.models.eulero_inference`.
Missing fields (for example channel counts or STFT parameters) are requested
interactively.  A new checkpoint file is emitted alongside the original with
``_rigenerated`` appended to the filename stem; the source checkpoint is left
untouched.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

from ar_spectra.models.autoencoder import AutoEncoder


def _prompt_int(label: str, description: str, default: Optional[int] = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label} ({description}){suffix}: ").strip()
        if not raw:
            if default is not None:
                return int(default)
            print("Please enter an integer value.")
            continue
        try:
            return int(raw)
        except ValueError:
            print("Invalid integer, try again.")


def _prompt_bool(label: str, description: str, default: Optional[bool] = None) -> bool:
    mapping = {"y": True, "yes": True, "n": False, "no": False}
    while True:
        if default is None:
            raw = input(f"{label} ({description}) [y/n]: ").strip().lower()
        else:
            def_char = "y" if default else "n"
            raw = input(f"{label} ({description}) [y/n] [{def_char}]: ").strip().lower()
            if not raw:
                return bool(default)
        if raw in mapping:
            return mapping[raw]
        print("Please reply with 'y' or 'n'.")


def _replace_auto_entries(obj: Any, path: Iterable[str]) -> Any:
    """Recursively replace the literal string 'auto' by prompting the user."""

    if isinstance(obj, dict):
        return {k: _replace_auto_entries(v, (*path, k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_auto_entries(v, (*path, f"[{idx}]") ) for idx, v in enumerate(obj)]
    if isinstance(obj, str) and obj.strip().lower() == "auto":
        field_name = ".".join(path)
        value = _prompt_int(field_name, "value required to replace 'auto'")
        return value
    return obj


def _collect_stft_config(existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    existing = existing or {}
    n_fft = _prompt_int("n_fft", "FFT size", existing.get("n_fft"))
    hop_length = _prompt_int("hop_length", "Hop size", existing.get("hop_length"))
    win_length = _prompt_int("win_length", "Window size", existing.get("win_length", n_fft))
    center = _prompt_bool("center", "Use centered frames", existing.get("center", True))
    normalized = _prompt_bool("normalized", "Normalize STFT", existing.get("normalized", False))
    onesided = _prompt_bool("onesided", "Use one-sided spectrum", existing.get("onesided", True))
    return {
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "win_length": int(win_length),
        "center": bool(center),
        "normalized": bool(normalized),
        "onesided": bool(onesided),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate checkpoint inference metadata.")
    parser.add_argument("checkpoint", type=Path, help="Path to the original checkpoint (.ckpt)")
    parser.add_argument("model_yaml", type=Path, help="Path to the model YAML file")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.model_yaml.is_file():
        parser.error(f"Model YAML not found: {args.model_yaml}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("Checkpoint payload is not a dictionary; aborting.", file=sys.stderr)
        sys.exit(1)
    current_inf_cfg: Dict[str, Any] = copy.deepcopy(ckpt.get("inference_config", {}))

    with args.model_yaml.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    if not isinstance(raw_cfg, dict) or "model" not in raw_cfg:
        print("Model YAML must contain a top-level 'model' key.", file=sys.stderr)
        sys.exit(1)

    model_cfg = _replace_auto_entries(copy.deepcopy(raw_cfg["model"]), ("model",))

    print("\nInstantiating autoencoder to export canonical configuration...\n")
    autoencoder = AutoEncoder.from_config(model_cfg)
    model_export = autoencoder.export_model_config()

    existing_in_channels = current_inf_cfg.get("in_channels")
    existing_audio_channels = current_inf_cfg.get("audio_channels")
    existing_sample_rate = current_inf_cfg.get("sample_rate")

    in_channels = _prompt_int("in_channels", "Spectrogram channels expected by the encoder", existing_in_channels)
    audio_channels = _prompt_int("audio_channels", "Waveform channels produced by the dataset", existing_audio_channels)
    sample_rate = _prompt_int("sample_rate", "Dataset sample rate", existing_sample_rate)

    stft_config = _collect_stft_config(current_inf_cfg.get("stft_config"))

    version = int(current_inf_cfg.get("version", 1))

    new_inference_cfg: Dict[str, Any] = {
        "version": version,
        "model": model_export,
        "in_channels": int(in_channels),
        "audio_channels": int(audio_channels),
        "sample_rate": int(sample_rate),
        "stft_config": stft_config,
    }

    if "train_stft_params" in current_inf_cfg:
        new_inference_cfg["train_stft_params"] = current_inf_cfg["train_stft_params"]

    ckpt_out = dict(ckpt)
    ckpt_out["inference_config"] = new_inference_cfg

    output_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_rigenerated{args.checkpoint.suffix}")
    torch.save(ckpt_out, output_path)

    print("\n✅ Regenerated checkpoint written to", output_path)


if __name__ == "__main__":
    main()
