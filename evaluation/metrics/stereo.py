# =============================================================================
# Stereo-image metrics (paper Table 7): width (M/S balance) and panning (L/R
# balance) errors between a reference and a reconstruction, symmetric by
# construction (penalises both collapse and over-widening), plus SI-SDR on the
# Side and on the Mid. ms_metrics_row is the per-file row every model gets.
# =============================================================================
from typing import Optional, Tuple, Dict, List

import torch

EPS = 1e-10


def estimate_delay(ref: torch.Tensor, rec: torch.Tensor, max_shift: int = 16384) -> int:
    """Estimate the integer delay of `rec` w.r.t. `ref` via FFT cross-correlation.

    Args:
        ref: (2, T) or (T,) reference waveform.
        rec: (2, T') or (T',) reconstruction.
        max_shift: maximum absolute delay searched, in samples.

    Returns:
        delay d (samples): rec[t] best matches ref[t + d]. Positive d means
        rec is late (starts later than ref).
    """
    x = ref.mean(0) if ref.dim() == 2 else ref
    y = rec.mean(0) if rec.dim() == 2 else rec
    n = int(x.numel() + y.numel())
    nfft = 1 << (n - 1).bit_length()
    X = torch.fft.rfft(x, nfft)
    Y = torch.fft.rfft(y, nfft)
    xc = torch.fft.irfft(X.conj() * Y, nfft)                     # xc[d] = sum_t x[t] y[t+d]
    lags = torch.cat([xc[: max_shift + 1], xc[-max_shift:]])     # d in [0..max, -max..-1]
    idx = int(torch.argmax(lags.abs()).item())                   # abs: robust to polarity flips
    return idx if idx <= max_shift else idx - (2 * max_shift + 1)


def align(ref: torch.Tensor, rec: torch.Tensor, max_shift: int = 16384) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delay-align `rec` to `ref` and crop both to the common overlap.

    Args:
        ref: (2, T) reference waveform.
        rec: (2, T') reconstruction (any length).

    Returns:
        (ref_a, rec_a): same-length, delay-compensated (2, T'') tensors.
    """
    d = estimate_delay(ref, rec, max_shift=max_shift)
    if d >= 0:                                   # rec is late: drop its first d samples
        ref_a, rec_a = ref, rec[:, d:]
    else:                                        # rec is early: drop ref's first |d| samples
        ref_a, rec_a = ref[:, -d:], rec
    T = min(ref_a.shape[-1], rec_a.shape[-1])
    return ref_a[:, :T].contiguous(), rec_a[:, :T].contiguous()


def _stft_ms(x: torch.Tensor, n_fft: int, hop: int) -> Tuple[torch.Tensor, ...]:
    """Return per-bin powers (E_L, E_R, E_M, E_S) of a (2, T) waveform."""
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop, window=win, return_complex=True)   # (2, F, N)
    L, R = X[0], X[1]
    M, S = (L + R) / 2 ** 0.5, (L - R) / 2 ** 0.5
    return L.abs().square(), R.abs().square(), M.abs().square(), S.abs().square()


def stereo_imaging_distance(
    ref: torch.Tensor,
    rec: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop: int = 512,
    freq_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Energy-weighted width/pan distance between two aligned stereo signals.

    Per TF bin: width w = E_S/(E_M+E_S) in [0,1]; pan p = (E_L-E_R)/(E_L+E_R)
    in [-1,1]. Distances are |Δ| means weighted by *reference* bin energy
    (optionally shaped by `freq_weights` over frequency bins), so both
    directions of error (collapse and over-widening) are penalized equally.
    All quantities are ratios → invariant to global gain.

    Args:
        ref: (2, T) reference waveform (align first!).
        rec: (2, T) reconstruction, same length.
        freq_weights: optional (n_fft//2+1,) nonnegative weights over freq bins.

    Returns:
        dict with:
          d_width     mean |w_rec - w_ref|              (0 = perfect image)
          width_bias  mean (w_rec - w_ref)              (<0 squash, >0 widening)
          d_pan       mean |p_rec - p_ref|
          pan_bias    mean (p_rec - p_ref)
          score       (d_width + d_pan) / 2
          sm_ref_db, sm_rec_db   10*log10(E_S/E_M) global ratios
    """
    assert ref.shape == rec.shape and ref.dim() == 2 and ref.shape[0] == 2, (ref.shape, rec.shape)
    EL_r, ER_r, EM_r, ES_r = _stft_ms(ref, n_fft, hop)
    EL_x, ER_x, EM_x, ES_x = _stft_ms(rec, n_fft, hop)

    w_ref = ES_r / (EM_r + ES_r + EPS)
    w_rec = ES_x / (EM_x + ES_x + EPS)
    p_ref = (EL_r - ER_r) / (EL_r + ER_r + EPS)
    p_rec = (EL_x - ER_x) / (EL_x + ER_x + EPS)

    weight = EM_r + ES_r                                             # reference energy
    if freq_weights is not None:
        weight = weight * freq_weights.to(weight.dtype).view(-1, 1)
    wsum = weight.sum() + EPS

    dw = (weight * (w_rec - w_ref).abs()).sum() / wsum
    bw = (weight * (w_rec - w_ref)).sum() / wsum
    dp = (weight * (p_rec - p_ref).abs()).sum() / wsum
    bp = (weight * (p_rec - p_ref)).sum() / wsum

    if freq_weights is not None:
        fw = freq_weights.view(-1, 1)
        sm_ref = (ES_r * fw).sum() / ((EM_r * fw).sum() + EPS)
        sm_rec = (ES_x * fw).sum() / ((EM_x * fw).sum() + EPS)
    else:
        sm_ref = ES_r.sum() / (EM_r.sum() + EPS)
        sm_rec = ES_x.sum() / (EM_x.sum() + EPS)

    return {
        "d_width": dw.item(), "width_bias": bw.item(),
        "d_pan": dp.item(), "pan_bias": bp.item(),
        "score": 0.5 * (dw + dp).item(),
        "sm_ref_db": 10 * torch.log10(sm_ref + EPS).item(),
        "sm_rec_db": 10 * torch.log10(sm_rec + EPS).item(),
    }


def band_weights(bands: List[Tuple[float, float]], sample_rate: int = 44100, n_fft: int = 2048) -> List[torch.Tensor]:
    """Binary freq_weights masks, one per (lo_hz, hi_hz) band."""
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    return [((freqs >= lo) & (freqs < hi)).float() for lo, hi in bands]


# ── Per-file row of the reconstruction evaluator ─────────────────────────────

# Per-file schema. The first five columns are byte-identical to the legacy
# compute_ms_metrics schema (old ms_metrics.csv stay readable, and readers that
# select columns by name are unaffected). The rest are already computed by
# stereo_imaging_distance and were previously thrown away.
MS_COLUMNS = [
    "file",
    "width_bias", "d_width",      # Side LEVEL (signed) / stereo image error (unsigned)
    "sisdr_s", "sisdr_m",         # Side CONTENT / Mid guardrail
    "d_pan", "pan_bias",          # L/R balance
    "score",                      # (d_width + d_pan) / 2
    "sm_ref_db", "sm_rec_db",     # global 10*log10(E_S/E_M), reference and reconstruction
    "ref_mono",                   # 1 = degenerate reference Side, see below
]

# A reference whose Side sits this far below its Mid carries no stereo
# information at all: in practice a mono file duplicated onto two channels
# (e.g. 14 of the 964 MusicCaps clips). For them S is exactly 0, so width_bias
# and sisdr_s are degenerate and must not enter a Side average. Flagged per
# file rather than dropped, so the caller decides and the count stays visible.
MONO_REF_DB = -80.0


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-10) -> float:
    """Scale-invariant SDR (dB) of a single-channel estimate."""
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est * ref).sum() / (ref.square().sum() + eps)
    proj = a * ref
    return float((10 * torch.log10(proj.square().sum() / ((est - proj).square().sum() + eps))).item())


def ms_metrics_row(ref: torch.Tensor, pred: torch.Tensor, sr: int) -> Dict[str, float]:
    """Stereo-imaging metrics on a (2, T) ref/pred pair, delay-aligned first.

    Two near-orthogonal quantities carry the stereo verdict, so neither
    substitutes for the other:

      width_bias  Side *level* error (→0; <0 squash, >0 over-widening)
      sisdr_s     Side *content*, SI-SDR on S=(L-R)/sqrt(2) — scale-invariant,
                  so it is blind to level and sees only whether the Side is the
                  right signal.

    sisdr_m is the Mid guardrail: whatever fixes the Side must leave the Mid —
    the only thing FAD/CLAP/CDPAM actually measure — intact.
    """
    ref_a, pred_a = align(ref.float(), pred.float())      # delay-compensate (~no-op for our recon)
    d = stereo_imaging_distance(ref_a, pred_a, sample_rate=sr)

    Sr = (ref_a[0] - ref_a[1]) / 2 ** 0.5                 # (T,) reference side
    Sx = (pred_a[0] - pred_a[1]) / 2 ** 0.5               # (T,) predicted side
    Mr = (ref_a[0] + ref_a[1]) / 2 ** 0.5                 # (T,) reference mid
    Mx = (pred_a[0] + pred_a[1]) / 2 ** 0.5               # (T,) predicted mid

    return {
        "width_bias": d["width_bias"], "d_width": d["d_width"],
        "sisdr_s": si_sdr(Sx, Sr), "sisdr_m": si_sdr(Mx, Mr),
        "d_pan": d["d_pan"], "pan_bias": d["pan_bias"], "score": d["score"],
        "sm_ref_db": d["sm_ref_db"], "sm_rec_db": d["sm_rec_db"],
        "ref_mono": int(d["sm_ref_db"] < MONO_REF_DB),
    }
