# =============================================================================
# Stereo loss explored during development and NOT used by the paper recipes.
# Add it to a run through `trainer.loss_config.extra` (see sage.nn.losses.experimental).
# =============================================================================
from typing import List, Optional

import torch

from sage.nn.losses.signal import get_window


class StereoCoherenceLoss(torch.nn.Module):
    r"""Inter-channel coherence, the differentiable surrogate of ``d_pan``.

    Targets the normalised Mid/Side cross-correlation per TF bin::

        γ = 2·M·S* / (|M|² + |S|²)          γ ∈ ℂ,  |γ| ≤ 1

    A single complex number that carries both perceptually-validated metrics
    verified numerically:

    * ``Re(γ) = p``, i.e. **exactly** the pan the metric measures (err 3.6e-07).
    * ``|γ| = 2r/(1+r²)`` with ``r = |S|/|M|`` — monotone in the width.
    * ``arg(γ)`` is the Mid↔Side relative phase.

    Why this and not a magnitude term. ``mrstft_sd`` is spectral-convergence +
    log-magnitude: both on ``|X|``. Rotating ``arg(S)`` by 90° at fixed magnitude
    moves ``d_width`` by 4.8e-09 (numerically nothing) and ``d_pan`` by 0.715 —
    a magnitude loss is provably blind to what pan measures, so it restores the
    Side's energy with whatever phase is cheapest. Measured consequence: the M/S
    arms fix the level (−0.08 dB) and leave ``d_pan`` pinned at the mono null.

    Why not the existing ``w_phs``. ``normalized_complex_distance_loss`` divides
    per bin by ``½(|x|+|y|)``, so a near-silent bin — where phase is noise — gets
    full weight. Here the denominator is ``|M|²+|S|²``: γ goes smoothly to 0 as
    the Side vanishes instead of blowing up, and the energy weight silences those
    bins a second time. Where the reference is mono (γ≈0) a large γ̂ is penalised,
    so the term also punishes *hallucinated* Side — the ``ms_replace`` failure.

    Gradient sensitivity. Near collapse the width is quadratic in ``r`` while
    ``|γ|`` is linear: their slopes differ by ``1/r``, so at the measured
    baseline (−12.4 dB → r = 0.24) this term carries ~4× the gradient of a
    width-based one, exactly in the regime where the model is stuck.

    ``pred``'s Mid is **detached**, so ∂L/∂M̂ is exactly zero and the gradient on
    L and R is antisymmetric — it cancels bit-for-bit in the M = L+R projection.
    That is what makes the term safe to graft onto a half-trained checkpoint:
    FAD/CLAP/CDPAM see the Mid alone and cannot be moved by it.
    """

    def __init__(
        self,
        fft_sizes: List[int] = (2048, 1024, 512),
        hop_sizes: List[int] = (512, 256, 128),
        win_lengths: List[int] = (2048, 1024, 512),
        window: str = "hann_window",
        detach_mid: bool = True,        # keeps ∂L/∂Mid exactly zero — see class docstring
        side_gate_db: Optional[float] = None,   # skip items whose target Side is negligible
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        self.detach_mid = detach_mid
        self.side_gate_db = side_gate_db
        self.eps = eps
        for i, wl in enumerate(self.win_lengths):
            self.register_buffer(f"win_{i}", get_window(window, wl).float(), persistent=False)

    def _gamma(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """γ = 2·M·S*/(|M|²+|S|²) for complex STFTs, shape-preserving."""
        return 2.0 * m * s.conj() / (m.abs().square() + s.abs().square() + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        m_p = (pred[:, 0] + pred[:, 1]) / 2 ** 0.5                     # (B, T)
        s_p = (pred[:, 0] - pred[:, 1]) / 2 ** 0.5                     # (B, T)
        m_t = (target[:, 0] + target[:, 1]) / 2 ** 0.5                 # (B, T)
        s_t = (target[:, 0] - target[:, 1]) / 2 ** 0.5                 # (B, T)

        if self.side_gate_db is not None:                              # item-level gate
            ratio_db = 10.0 * torch.log10(
                s_t.square().sum(-1) / m_t.square().sum(-1).clamp_min(self.eps) + self.eps)
            keep = ratio_db >= self.side_gate_db                       # (B,)
            if not bool(keep.any()):
                return pred.sum() * 0.0                                # graph-connected zero
            if not bool(keep.all()):
                m_p, s_p, m_t, s_t = m_p[keep], s_p[keep], m_t[keep], s_t[keep]

        total = pred.sum() * 0.0                                       # scalar accumulator
        for i, (n_fft, hop, wl) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            win = getattr(self, f"win_{i}").to(pred.device, pred.dtype)
            stft = lambda x: torch.stft(x, n_fft, hop, wl, window=win,
                                        return_complex=True, center=True)
            Mp, Sp = stft(m_p), stft(s_p)                              # (B, F, N) complex
            Mt, St = stft(m_t), stft(s_t)                              # (B, F, N) complex
            if self.detach_mid:
                Mp = Mp.detach()

            d = self._gamma(Mp, Sp) - self._gamma(Mt, St)              # (B, F, N) complex
            mod = (d.real.square() + d.imag.square() + self.eps).sqrt()  # (B, F, N), |·| smooth at 0
            w = Mt.abs().square() + St.abs().square()                  # (B, F, N) reference energy
            total = total + (w * mod).sum() / (w.sum() + self.eps)     # same weighting as d_pan

        return total / len(self.fft_sizes)                             # scalar
