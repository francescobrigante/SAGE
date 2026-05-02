import torch

def to_complex_spectrogram(X: torch.Tensor) -> torch.Tensor:
    """
    Convert a spectrogram tensor to complex dtype handling multiple layouts:
      - complex tensor (..., C, F, T) -> returned as-is
      - real tensor with RI on last dim (..., C, F, T, 2) -> view_as_complex
      - complex-as-channels (cac=True) from dataset:
            (B, 2C, F, T) or (2C, F, T) with order [c0_r, c0_i, c1_r, c1_i, ...]
        -> returns (B, C, F, T) or (C, F, T) complex
    """
    if torch.is_complex(X):
        return X
    if not X.is_floating_point():
        raise TypeError("Expected floating or complex tensor for spectrogram input.")

    # Case: real/imag in last dimension
    if X.ndim >= 1 and X.size(-1) == 2:
        return torch.view_as_complex(X.contiguous())

    # Case: complex-as-channels, shape (B?, 2C, F, T)
    if X.ndim == 4:
        B, C2, F, T = X.shape
        if C2 % 2 != 0:
            raise ValueError(f"Channel dimension must be even for complex-as-channels. Got {C2}.")
        C = C2 // 2
        Xv = X.reshape(B, C, 2, F, T)
        real = Xv[:, :, 0, :, :].float()
        imag = Xv[:, :, 1, :, :].float()
        return torch.complex(real, imag)
    elif X.ndim == 3:
        C2, F, T = X.shape
        if C2 % 2 != 0:
            raise ValueError(f"Channel dimension must be even for complex-as-channels. Got {C2}.")
        C = C2 // 2
        Xv = X.reshape(C, 2, F, T)
        real = Xv[:, 0, :, :].float()
        imag = Xv[:, 1, :, :].float()
        return torch.complex(real, imag)
    raise ValueError("Unsupported spectrogram shape. Expected (..., C, F, T), (..., C, F, T, 2) or (B, 2C, F, T)/(2C, F, T).")
