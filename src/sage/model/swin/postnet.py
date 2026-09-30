# ===============
# ResidualPostNet: zero-init residual spectral refiner applied after
# PatchUnembed in the SAGEDecoder. Exact identity at init (last conv is
# zero-initialized, ControlNet-style), so it can be bolted onto a pretrained
# decoder for the decoder-finetune phase without perturbing its function.
# ===============

import torch
import torch.nn as nn


class ResidualPostNet(nn.Module):
    """Zero-init residual refiner for the decoder's output spectrogram.

    Targets the patch-boundary discontinuities introduced by PatchUnembed
    (ConvTranspose2d with kernel = stride = patch_size synthesizes each
    freq-band of `patch_size[0]` bins independently per token). The kernel is
    tall in frequency (to stitch across band edges) and short in time (to
    avoid temporal smearing).

    The last conv has zero weight and bias, so at init `forward(S) == S`
    exactly: bolting this onto a pretrained decoder is a no-op until
    gradients flow (zero-convolution trick, ControlNet / ReZero).

    Real-valued spectrograms only (CAC channel layout).

    Args:
        channels: Spectrogram channels (4 = stereo CAC STFT).
        hidden: Hidden conv width.
        kernel_size: (freq, time) kernel of both convs.
    """

    def __init__(
        self,
        channels: int = 4,            # spectrogram channels (stereo CAC = 4)
        hidden: int = 64,             # hidden conv width
        kernel_size: tuple = (7, 3),  # (freq, time) — tall in freq, short in time
    ) -> None:
        super().__init__()
        k_f, k_t = int(kernel_size[0]), int(kernel_size[1])
        pad = (k_f // 2, k_t // 2)
        # reflect padding: avoids zero-pad bias at the DC / Nyquist spectrum edges
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, (k_f, k_t), padding=pad, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden, channels, (k_f, k_t), padding=pad, padding_mode="reflect"),
        )
        # zero-init the last conv → residual is exactly 0 at init (identity).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """``S`` (B, C, F, T) → refined spectrogram (B, C, F, T)."""
        return S + self.net(S)                          # (B, C, F, T)
