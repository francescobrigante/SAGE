# =============================================================================
# Runtime config consistency checks — raised before training starts.
# =============================================================================

from typing import Optional


def check_cac_consistency(
    is_complex_model: bool,
    train_cac: bool,
    eval_cac: Optional[bool] = None,
    demo_cac: Optional[bool] = None,
) -> None:
    """Raise ValueError when a complex model is paired with cac=True dataset.

    Complex models (is_complex=True) receive native complex64 STFT tensors
    (B, C, F, T) dtype=complex64.  CAC mode (complex-as-channels) splits the
    complex tensor into real+imag channels and returns float32 — incompatible.
    A mismatch means the model processes float32 channels instead of complex64,
    producing silently wrong results with a loss that still decreases.

    Args:
        is_complex_model: True when the encoder has is_complex=True.
        train_cac: Value of data.train_dataset.cac in the Hydra config.
        eval_cac: Value of data.eval_dataset.cac (None = no eval dataset).
        demo_cac: Value of data.demo.istft_params.cac (None = no demo).

    Raises:
        ValueError: On any mismatch.
    """
    if not is_complex_model:
        return

    _HINT = (
        "Add these CLI overrides: "
        "data.train_dataset.cac=false "
        "data.eval_dataset.cac=false "
        "data.demo.istft_params.cac=false"
    )

    if train_cac:
        raise ValueError(
            "CONFIG MISMATCH: encoder.is_complex=True but data.train_dataset.cac=True. "
            "Complex models require cac=false (native complex64 tensors). " + _HINT
        )
    if eval_cac:
        raise ValueError(
            "CONFIG MISMATCH: encoder.is_complex=True but data.eval_dataset.cac=True. "
            "Complex models require cac=false (native complex64 tensors). " + _HINT
        )
    if demo_cac:
        raise ValueError(
            "CONFIG MISMATCH: encoder.is_complex=True but data.demo.istft_params.cac=True. "
            "Complex models require cac=false (native complex64 tensors). " + _HINT
        )
