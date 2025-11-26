import torch
from typing import Optional, Dict, Any, Union, Tuple


def _spectrogram_to_complex(S: torch.Tensor) -> torch.Tensor:
    """Convert supported spectrogram layouts to a complex tensor."""
    if torch.is_complex(S):
        return S
    if S.dim() == 4:
        B, Cx, F, T = S.shape
        if Cx % 2 != 0:
            raise ValueError(f"Expected even channels (2C) for CAC input, got {Cx}")
        C = Cx // 2
        Sview = S.view(B, C, 2, F, T)
        real = Sview[:, :, 0, :, :]
        imag = Sview[:, :, 1, :, :]
        return torch.complex(real, imag)
    if S.dim() == 3:
        Cx, F, T = S.shape
        if Cx % 2 != 0:
            raise ValueError(f"Expected even channels (2C) for CAC input, got {Cx}")
        C = Cx // 2
        Sview = S.view(C, 2, F, T)
        real = Sview[:, 0, :, :]
        imag = Sview[:, 1, :, :]
        return torch.complex(real, imag)
    raise ValueError(f"Unsupported spectrogram shape for CAC conversion: {tuple(S.shape)}")


def _spectrogram_from_complex(S: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Project a complex spectrogram back to the original layout."""
    if torch.is_complex(like):
        return S
    if like.dim() == 4:
        B, C, F, T = S.shape
        real = S.real
        imag = S.imag
        out = torch.stack((real, imag), dim=2).reshape(B, 2 * C, F, T)
        return out.to(like.dtype)
    if like.dim() == 3:
        C, F, T = S.shape
        real = S.real
        imag = S.imag
        out = torch.stack((real, imag), dim=1).reshape(2 * C, F, T)
        return out.to(like.dtype)
    raise ValueError(f"Unsupported spectrogram shape for CAC restore: {tuple(like.shape)}")


class IdentityTransform:
    def __init__(self):
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LogMagnitudeTransform:
    """
    Apply a log(1+alpha*|S|) normalization to the magnitude of a complex
    spectrogram, preserving phase. Works with:
    - complex tensors shaped (B,C,F,T) or (C,F,T)
    - real tensors with complex-as-channels shaped (B,2C,F,T) or (2C,F,T)

    inverse() maps back to the linear magnitude domain using expm1 and the
    stored alpha parameter, preserving phase.
    """

    def __init__(self, eps: float = 1e-8, alpha: float = 1.0):
        self.eps = float(eps)
        self.alpha = float(alpha)

    def transform(self, S_in: torch.Tensor) -> torch.Tensor:
        S = _spectrogram_to_complex(S_in)
        mag = torch.abs(S)
        # unit complex with safe denom
        unit = S / (mag + self.eps)
        mag_n = torch.log1p(self.alpha * mag)
        Sout = unit * mag_n
        return _spectrogram_from_complex(Sout, S_in)

    def inverse(self, S_in: torch.Tensor) -> torch.Tensor:
        S = _spectrogram_to_complex(S_in)
        mag_n = torch.abs(S)
        unit = S / (mag_n + self.eps)
        mag = torch.expm1(mag_n) / self.alpha
        Sout = unit * mag
        return _spectrogram_from_complex(Sout, S_in)


class PowerMagnitudeTransform:
    """Apply the beta·|S|^alpha rescaling used by ``normalize_complex`` while preserving phase.

    Args:
        alpha: Power-law exponent applied to the magnitude.
        beta: Multiplicative scale applied after the exponent.
        eps: Numerical stability term used only when extracting phase.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-8):
        if alpha <= 0.0:
            raise ValueError("alpha must be strictly positive for power-law rescaling")
        if beta <= 0.0:
            raise ValueError("beta must be strictly positive for power-law rescaling")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def transform(self, S_in: torch.Tensor) -> torch.Tensor:
        S = _spectrogram_to_complex(S_in)
        mag = torch.abs(S)
        unit = S / (mag + self.eps)
        # magnitude follows beta_rescale * |x|**alpha_rescale
        mag_scaled = self.beta * mag.pow(self.alpha)
        Sout = unit * mag_scaled
        return _spectrogram_from_complex(Sout, S_in)

    def inverse(self, S_in: torch.Tensor) -> torch.Tensor:
        S = _spectrogram_to_complex(S_in)
        mag = torch.abs(S)
        mag_scaled = mag / self.beta
        # undo the normalize_complex exponentiation
        mag_restored = mag_scaled.clamp_min(0.0).pow(1.0 / self.alpha)
        unit = S / (mag + self.eps)
        Sout = unit * mag_restored
        return _spectrogram_from_complex(Sout, S_in)


def create_pre_transform(spec: Optional[Union[str, Dict[str, Any]]]):
    """
    Factory to create a spectrogram pre/post transform.
    Accepts:
    - None or {type: identity}: returns IdentityTransform
    - "identity"
    - "log_mag" or {"type":"log_mag", "config": {eps, alpha}}
    - "power_mag" or {"type":"power_mag", "config": {alpha, beta, eps}}
    """
    if spec is None:
        return IdentityTransform()
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("identity", "none"):
            return IdentityTransform()
        if key in ("log_mag", "logmag", "log_magnitude"):
            return LogMagnitudeTransform()
        if key in ("power_mag", "powermag", "power", "power_norm"):
            return PowerMagnitudeTransform()
        raise ValueError(f"Unknown pre_transform string spec: {spec}")
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported pre_transform spec type: {type(spec).__name__}")
    t = (spec.get("type") or "identity").lower()
    cfg = spec.get("config", {}) or {}
    if t in ("identity", "none"):
        return IdentityTransform()
    if t in ("log_mag", "logmag", "log_magnitude"):
        return LogMagnitudeTransform(**cfg)
    if t in ("power_mag", "powermag", "power", "power_norm"):
        return PowerMagnitudeTransform(**cfg)
    raise ValueError(f"Unknown pre_transform type: {t}")


def resolve_pre_transform(spec: Optional[Union[str, Dict[str, Any]]]) -> Tuple[Optional[Any], bool, bool, bool]:
    """Return (transform, apply_to_encoder, apply_inverse, apply_to_target).

    Control keys accepted in dict specs:
      - apply_encoder / apply_to_encoder: bool
      - apply_inverse / apply_decoder: bool
      - apply_target / apply_to_target: bool

    They are stripped before instantiating the transform. ``None`` yields
    (None, True, True, False) meaning no transform will be applied.
    """

    apply_encoder = True
    apply_inverse = True
    apply_target = False

    if spec is None:
        return None, apply_encoder, apply_inverse, apply_target

    core_spec: Optional[Union[str, Dict[str, Any]]] = spec

    if isinstance(spec, dict):
        apply_encoder = bool(spec.get("apply_encoder", spec.get("apply_to_encoder", True)))
        apply_inverse = bool(spec.get("apply_inverse", spec.get("apply_decoder", True)))
        apply_target = bool(spec.get("apply_target", spec.get("apply_to_target", False)))
        control_keys = {
            "apply_encoder",
            "apply_to_encoder",
            "apply_inverse",
            "apply_decoder",
            "apply_target",
            "apply_to_target",
        }
        core_spec = {k: v for k, v in spec.items() if k not in control_keys}
        if not core_spec:
            core_spec = {"type": "identity"}

    transform = create_pre_transform(core_spec)
    return transform, apply_encoder, apply_inverse, apply_target
