# =============================================================================
# Backward compatibility with checkpoints written before the code became the
# `sage` package. A checkpoint stores the model architecture as Hydra configs
# (`inference_config["model"]`) whose `_target_` entries are class paths; the
# paper checkpoint (SAGE_FTe992.ckpt) was saved with the pre-release layout
# (`c_vae.*`, `ar_spectra.*`). Only class paths moved: constructor arguments,
# attribute names and therefore state_dict keys are unchanged.
# =============================================================================
from __future__ import annotations

from typing import Any

# Old class path -> current class path. Every class that can appear as a `_target_`
# in a pre-release checkpoint is listed; tests/test_compat.py checks each one resolves.
LEGACY_TARGETS: dict[str, str] = {
    "c_vae.swin.encoder.SwinEncoder": "sage.model.encoder.SAGEEncoder",
    "c_vae.swin.decoder.SwinDecoder": "sage.model.decoder.SAGEDecoder",
    "ar_spectra.models.bottlenecks.VAEBottleneck": "sage.nn.bottleneck.VAEBottleneck",
    "ar_spectra.models.bottlenecks.SkipBottleneck": "sage.nn.bottleneck.SkipBottleneck",
    "c_vae.bottleneck.ComplexVAEBottleneck": "sage.nn.complex.bottleneck.ComplexVAEBottleneck",
}
_LEGACY_ROOTS = ("c_vae.", "ar_spectra.")
_TARGET_KEYS = ("_target_", "class")          # "class" = legacy {class, kwargs} spec format


def upgrade_class_path(path: str) -> str:
    """Map a pre-release class path to its current location (current paths pass through)."""
    if path in LEGACY_TARGETS:
        return LEGACY_TARGETS[path]
    if path.startswith(_LEGACY_ROOTS):
        raise ValueError(
            f"Checkpoint refers to {path!r}, a class that is not part of the released code "
            f"(known legacy classes: {sorted(LEGACY_TARGETS)})."
        )
    return path


def upgrade_model_config(cfg: Any) -> Any:
    """Return a copy of a checkpoint model config with every legacy class path upgraded."""
    if isinstance(cfg, dict):
        return {k: upgrade_class_path(v) if k in _TARGET_KEYS and isinstance(v, str) else upgrade_model_config(v)
                for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [upgrade_model_config(v) for v in cfg]
    return cfg
