import torch
from typing import Optional, Dict, Any, Union


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

    @staticmethod
    def _to_complex(S: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def _from_complex(S: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # Return in the same representation as `like`
        if torch.is_complex(like):
            return S
        # complex-as-channels
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

    def transform(self, S_in: torch.Tensor) -> torch.Tensor:
        S = self._to_complex(S_in)
        mag = torch.abs(S)
        # unit complex with safe denom
        unit = S / (mag + self.eps)
        mag_n = torch.log1p(self.alpha * mag)
        Sout = unit * mag_n
        return self._from_complex(Sout, S_in)

    def inverse(self, S_in: torch.Tensor) -> torch.Tensor:
        S = self._to_complex(S_in)
        mag_n = torch.abs(S)
        unit = S / (mag_n + self.eps)
        mag = torch.expm1(mag_n) / self.alpha
        Sout = unit * mag
        return self._from_complex(Sout, S_in)


def create_pre_transform(spec: Optional[Union[str, Dict[str, Any]]]):
    """
    Factory to create a spectrogram pre/post transform.
    Accepts:
    - None or {type: identity}: returns IdentityTransform
    - "identity"
    - "log_mag" or {"type":"log_mag", "config": {eps, alpha}}
    """
    if spec is None:
        return IdentityTransform()
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("identity", "none"):
            return IdentityTransform()
        if key in ("log_mag", "logmag", "log_magnitude"):
            return LogMagnitudeTransform()
        raise ValueError(f"Unknown pre_transform string spec: {spec}")
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported pre_transform spec type: {type(spec).__name__}")
    t = (spec.get("type") or "identity").lower()
    cfg = spec.get("config", {}) or {}
    if t in ("identity", "none"):
        return IdentityTransform()
    if t in ("log_mag", "logmag", "log_magnitude"):
        return LogMagnitudeTransform(**cfg)
    raise ValueError(f"Unknown pre_transform type: {t}")
