# =============================================================================
# evaluation/losses.py
# Per-file audio reconstruction metrics: SI-SDR, multi-resolution STFT loss,
# and CDPAM perceptual similarity. All functions operate on [C, T] tensors.
# =============================================================================

from __future__ import annotations

import torch
import torchaudio

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
    belongs in the evaluation pipeline (evaluate_swin.py), not here.
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

_hann_windows: dict[tuple[int, torch.device], torch.Tensor] = {}

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


# ── CDPAM ─────────────────────────────────────────────────────

_cdpam_model = None
_cdpam_resamplers: dict[int, torchaudio.transforms.Resample] = {}


def cdpam_score(target: torch.Tensor, pred: torch.Tensor, src_sr: int, device="cpu") -> float:
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
    t = target.float().cpu().mean(0, keepdim=True) * 32768.0
    p = pred.float().cpu().mean(0, keepdim=True) * 32768.0
    n = min(t.shape[-1], p.shape[-1])
    with torch.no_grad():
        return float(_cdpam_model.forward(t[..., :n], p[..., :n]).item())
