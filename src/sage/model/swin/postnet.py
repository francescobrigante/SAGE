# ===============
# ResidualPostNet: zero-init residual spectral refiner applied after
# PatchUnembed in the SAGEDecoder. Exact identity at init (last conv is
# zero-initialized, ControlNet-style), so it can be bolted onto a pretrained
# decoder for the decoder-finetune phase without perturbing its function.
# Long inputs run in time slices, to bound memory and keep every tensor within
# 32-bit indexing.
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

    Inputs of more than ``max_slice_frames`` frames over the batch (~190 s of
    audio at batch 1), or whose widest convolution input (the reflect-padded
    hidden tensor) would exceed ``max_elements``, run per batch item, in time
    slices of at most ``max_slice_frames`` with a halo of one time radius per
    conv. The slices bound the peak memory (at 300 s a single pass raises the
    decoder's from ~14 to ~20 GiB) at the same speed, and keep every tensor
    below 2**31 elements, above which the CUDA reflection pad fails
    (32-bit index math, ~380 s) and, just below which, cuDNN already runs the
    conv 4-5x slower (measured on A100: 135 ms at 32560 frames, 589 ms at
    32600).
    Every output frame sees the same neighbours as in one pass and the reflect
    padding only touches the true edges, so the result is the same up to float
    rounding (bit-identical on the paper model, CPU and A100, 10 s to 10 min);
    shorter inputs run the single pass as before.

    Args:
        channels: Spectrogram channels (4 = stereo CAC STFT).
        hidden: Hidden conv width.
        kernel_size: (freq, time) kernel of both convs.
    """

    max_elements: int = 2**31 - 1     # largest padded conv input of one pass (32-bit index math, cuDNN speed)
    max_slice_frames: int = 16384     # frames of one pass over the batch, and per slice (same speed, less memory)

    def __init__(
        self,
        channels: int = 4,            # spectrogram channels (stereo CAC = 4)
        hidden: int = 64,             # hidden conv width
        kernel_size: tuple = (7, 3),  # (freq, time) — tall in freq, short in time
    ) -> None:
        super().__init__()
        k_f, k_t = int(kernel_size[0]), int(kernel_size[1])
        pad = (k_f // 2, k_t // 2)
        self._pad = pad                                  # (freq, time) padding of each conv
        self._width = max(channels, hidden)              # widest activation, in channels
        self._halo = 2 * pad[1]                          # time context of the two convs, per side
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
        B, _, F, T = S.shape
        fits = B * self._frame_elements(F) * (T + 2 * self._pad[1]) <= self.max_elements
        if fits and B * T <= self.max_slice_frames:
            return S + self.net(S)                      # (B, C, F, T)
        return self._sliced(S)

    def _frame_elements(self, F: int) -> int:
        """Elements per time frame of the widest reflect-padded conv input, for one batch item."""
        return self._width * (F + 2 * self._pad[0])

    def _sliced(self, S: torch.Tensor) -> torch.Tensor:
        """One pass per batch item and time slice; each slice carries ``_halo`` frames per side."""
        T, h = S.shape[-1], self._halo
        step = min(self.max_elements // self._frame_elements(S.shape[-2]) - 2 * (h + self._pad[1]),
                   self.max_slice_frames)
        if step < 1:
            raise ValueError(f"ResidualPostNet: one time frame of {tuple(S.shape)} exceeds "
                             f"{self.max_elements} elements")
        out = torch.empty_like(S)
        for b in range(S.shape[0]):
            for a in range(0, T, step):
                e = min(a + step, T)
                lo, hi = max(a - h, 0), min(e + h, T)
                part = S[b:b + 1, ..., lo:hi].contiguous()   # (1, C, F, e - a + halos): own storage,
                                                             # the 32-bit check reads a view's offsets
                out[b:b + 1, ..., a:e] = (part + self.net(part))[..., a - lo:e - lo]
        return out                                      # (B, C, F, T)
