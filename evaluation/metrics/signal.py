# =============================================================================
# Per-file reconstruction metrics: SDR / SI-SDR, multi-resolution STFT and mel
# distances, and CDPAM perceptual similarity. All functions take [C, T] tensors.
# =============================================================================
from __future__ import annotations

import torch
import torchaudio

from evaluation.common import downmix, CHANNEL_MID

def compute_sdr_and_sisdr(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> tuple[float, float]:
    """Compute both standard scale-dependent SDR and scale-invariant SDR (SI-SDR)
    efficiently in a single pass using dot products, avoiding redundant vector allocations.

    Both metrics are computed on the flattened de-biased (mean-removed) tensors.
    """
    t = target.flatten().float()
    p = pred.flatten().float()
    t = t - t.mean()
    p = p - p.mean()

    t_sq = torch.sum(t ** 2)
    p_sq = torch.sum(p ** 2)
    t_dot_p = torch.sum(t * p)

    # 1. SDR
    noise_sdr = p_sq + t_sq - 2.0 * t_dot_p
    sdr_val = 10.0 * torch.log10(t_sq / (torch.clamp(noise_sdr, min=0.0) + eps))

    # 2. SI-SDR
    proj_sq = (t_dot_p ** 2) / (t_sq + eps)
    noise_sisdr = p_sq - proj_sq
    sisdr_val = 10.0 * torch.log10(proj_sq / (torch.clamp(noise_sisdr, min=0.0) + eps))

    return sdr_val.item(), sisdr_val.item()




# ── SI-SDR ────────────────────────────────────────────────────

def si_sdr(target: torch.Tensor, pred: torch.Tensor) -> float:
    """Scale-invariant SDR (dB) from [C, T] tensors on any device.

    Operates on the full, already-stitched audio. The 'chunking' detail
    belongs in the evaluation pipeline (evaluation/reconstruction.py), not here.
    """
    t = target.flatten().float()
    p = pred.flatten().float()
    t = t - t.mean()
    p = p - p.mean()
    alpha = (p @ t) / (t @ t + 1e-8)
    proj  = alpha * t
    noise = p - proj
    return 10.0 * torch.log10(proj.norm() ** 2 / (noise.norm() ** 2 + 1e-8)).item()


# ── Multi-resolution STFT loss ────────────────────────────────

_hann_windows: dict[tuple, torch.Tensor] = {}
_mel_filters:  dict[tuple, torch.Tensor] = {}

# torchaudio renamed create_fb_matrix → melscale_fbanks (both share the same
# signature); pick whichever the installed version exposes so the mel filterbank
# works across torchaudio releases.
_mel_fb_fn = getattr(torchaudio.functional, "melscale_fbanks", None) \
    or torchaudio.functional.create_fb_matrix

def stft_loss(target: torch.Tensor, pred: torch.Tensor) -> float:
    """Multi-resolution log-magnitude STFT L1 (3 scales) from [C, T] tensors."""
    global _hann_windows
    t_m = target.float().mean(0)
    p_m = pred.float().mean(0)
    loss, eps = 0.0, 1e-8
    for n_fft in (512, 1024, 2048):
        key = (n_fft, target.device)
        if key not in _hann_windows:
            _hann_windows[key] = torch.hann_window(n_fft, device=target.device)
        win = _hann_windows[key]
        T_s = torch.stft(t_m, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs()
        P_s = torch.stft(p_m, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs()
        loss += (torch.log(T_s + eps) - torch.log(P_s + eps)).abs().mean().item()
    return loss / 3.0


def spectral_losses(
    target: torch.Tensor,
    pred: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
) -> dict[str, float]:
    """Multi-resolution STFT L1 and mel L1 (3 scales) in one pass from [C, T] tensors.

    Computes the STFT once per resolution and derives both metrics, halving the
    STFT overhead vs calling stft_loss() and a separate mel_loss() sequentially.
    Returns {"stft_loss": float, "mel_loss": float}.
    """
    global _hann_windows, _mel_filters
    t_m = target.float().mean(0)
    p_m = pred.float().mean(0)
    stft_acc = mel_acc = 0.0
    eps = 1e-8
    for n_fft in (512, 1024, 2048):
        win_key = (n_fft, target.device)
        if win_key not in _hann_windows:
            _hann_windows[win_key] = torch.hann_window(n_fft, device=target.device)
        win = _hann_windows[win_key]

        T_s = torch.stft(t_m, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs()  # [F, T]
        P_s = torch.stft(p_m, n_fft, n_fft // 4, n_fft, win, return_complex=True).abs()  # [F, T]

        stft_acc += (torch.log(T_s + eps) - torch.log(P_s + eps)).abs().mean().item()

        mel_key = (n_fft, n_mels, sample_rate, target.device)
        if mel_key not in _mel_filters:
            fb = _mel_fb_fn(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=float(sample_rate) / 2,
                n_mels=n_mels,
                sample_rate=sample_rate,
            ).to(target.device)                                     # [n_freqs, n_mels]
            _mel_filters[mel_key] = fb
        fb = _mel_filters[mel_key]

        T_mel = fb.T @ T_s                                          # [n_mels, T_frames]
        P_mel = fb.T @ P_s                                          # [n_mels, T_frames]
        mel_acc += (torch.log(T_mel + eps) - torch.log(P_mel + eps)).abs().mean().item()

    return {"stft_loss": stft_acc / 3.0, "mel_loss": mel_acc / 3.0}


# ── CDPAM ─────────────────────────────────────────────────────

_cdpam_model = None
_cdpam_resamplers: dict[int, torchaudio.transforms.Resample] = {}


def cdpam_score(target: torch.Tensor, pred: torch.Tensor, src_sr: int, device="cpu",
                channel: str = CHANNEL_MID) -> float:
    """CDPAM perceptual similarity from [C, T] tensors.

    CDPAM loads weights with torch.load (weights_only=False required for older
    safetensors-free checkpoints). The model itself runs on `device`.
    """
    global _cdpam_model, _cdpam_resamplers
    if _cdpam_model is None:
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        try:
            import cdpam
            try:
                _cdpam_model = cdpam.CDPAM(dev=str(device))
            except Exception:
                _cdpam_model = cdpam.CDPAM(dev="cpu")
                device = "cpu"
        finally:
            torch.load = _orig

    cdpam_sr = 22050
    if src_sr != cdpam_sr:
        if src_sr not in _cdpam_resamplers:
            _cdpam_resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, cdpam_sr)
        rs = _cdpam_resamplers[src_sr]
        target = rs(target.cpu())
        pred   = rs(pred.cpu())

    # CDPAM expects float32 mono [1, T] scaled to [-32768, 32768]
    t = downmix(target.float().cpu(), channel).unsqueeze(0) * 32768.0
    p = downmix(pred.float().cpu(), channel).unsqueeze(0) * 32768.0
    n = min(t.shape[-1], p.shape[-1])
    with torch.no_grad():
        return float(_cdpam_model.forward(t[..., :n], p[..., :n]).item())
