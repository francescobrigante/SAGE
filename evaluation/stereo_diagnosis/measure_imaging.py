# ===============
# Task 1 measurement runner: aligns every reconstruction to its original, then
# computes imaging metrics (global / per-band / sliding-window) and quality
# metrics (SI-SDR, MRSTFT, LSD) on L/R and on M/S separately.
# Writes results/metrics.csv, results/bands.csv, results/temporal.npz.
#   uv run python evaluation/stereo_diagnosis/measure_imaging.py [root_dir]
# ===============
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from stereo_imaging import align, band_weights, estimate_delay, stereo_imaging_distance

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/francesco/Desktop/commercial_songs")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

SR = 44100
MODELS = ["sage", "same", "sao-vae", "codicodec", "music2latent"]
BANDS = [(0, 200), (200, 2000), (2000, 8000), (8000, SR // 2)]
BAND_NAMES = ["lt200", "200_2k", "2k_8k", "gt8k"]
N_FFT, HOP = 2048, 512
WIN_FRAMES, WIN_HOP = 86, 43          # ~1.0 s windows, ~0.5 s hop at hop=512

EPS = 1e-10


def to_ms(x: torch.Tensor):
    return (x[0] + x[1]) / math.sqrt(2), (x[0] - x[1]) / math.sqrt(2)


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est * ref).sum() / (ref.square().sum() + EPS)
    proj = a * ref
    return (10 * torch.log10(proj.square().sum() / ((est - proj).square().sum() + EPS))).item()


def _mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft)
    return torch.stft(x, n_fft, hop, window=win, return_complex=True).abs()


def mrstft_dist(est: torch.Tensor, ref: torch.Tensor) -> float:
    """SC + L1 log-mag averaged over 4 resolutions (training-style distance)."""
    total = 0.0
    for n_fft in (2048, 1024, 512, 256):
        E, R = _mag(est, n_fft, n_fft // 4), _mag(ref, n_fft, n_fft // 4)
        sc = torch.linalg.norm(R - E) / (torch.linalg.norm(R) + EPS)
        lm = (torch.log(E.clamp(1e-5)) - torch.log(R.clamp(1e-5))).abs().mean()
        total += (sc + lm).item()
    return total / 4


def lsd(est: torch.Tensor, ref: torch.Tensor) -> float:
    E, R = _mag(est, N_FFT, HOP).square(), _mag(ref, N_FFT, HOP).square()
    d = 10 * (torch.log10(R.clamp(1e-10)) - torch.log10(E.clamp(1e-10)))
    return d.square().mean(dim=0).sqrt().mean().item()


def frame_width(x: torch.Tensor):
    """Per-frame energy-weighted width + per-frame energy, from a (2,T) signal."""
    win = torch.hann_window(N_FFT)
    X = torch.stft(x, N_FFT, HOP, window=win, return_complex=True)
    EM = ((X[0] + X[1]) / math.sqrt(2)).abs().square()
    ES = ((X[0] - X[1]) / math.sqrt(2)).abs().square()
    w = ES / (EM + ES + EPS)
    e = EM + ES
    return (w * e).sum(0) / (e.sum(0) + EPS), e.sum(0)          # (N,), (N,)


rows, band_rows, temporal = [], [], {}

for track_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    track = track_dir.name
    ref_full, sr = torchaudio.load(track_dir / "original.wav")
    assert sr == SR
    for model in MODELS:
        rec_full, sr = torchaudio.load(track_dir / f"{model}.wav")
        assert sr == SR
        delay = estimate_delay(ref_full, rec_full)
        ref, rec = align(ref_full, rec_full)

        m = stereo_imaging_distance(ref, rec, SR, N_FFT, HOP)
        m.update(track=track, model=model, delay=delay)

        # quality on L/R (mean of channels) and on M/S
        Mr, Sr = to_ms(ref)
        Mx, Sx = to_ms(rec)
        m["sisdr_lr"] = 0.5 * (si_sdr(rec[0], ref[0]) + si_sdr(rec[1], ref[1]))
        m["sisdr_m"] = si_sdr(Mx, Mr)
        m["sisdr_s"] = si_sdr(Sx, Sr)
        m["mrstft_lr"] = 0.5 * (mrstft_dist(rec[0], ref[0]) + mrstft_dist(rec[1], ref[1]))
        m["mrstft_m"] = mrstft_dist(Mx, Mr)
        m["mrstft_s"] = mrstft_dist(Sx, Sr)
        m["lsd_m"] = lsd(Mx, Mr)
        m["lsd_s"] = lsd(Sx, Sr)
        rows.append(m)

        # per-band imaging
        for name, fw in zip(BAND_NAMES, band_weights(BANDS, SR, N_FFT)):
            b = stereo_imaging_distance(ref, rec, SR, N_FFT, HOP, freq_weights=fw)
            band_rows.append({"track": track, "model": model, "band": name,
                              "d_width": b["d_width"], "width_bias": b["width_bias"],
                              "sm_ref_db": b["sm_ref_db"], "sm_rec_db": b["sm_rec_db"]})

        # sliding-window width trajectories
        w_ref, e_ref = frame_width(ref)
        w_rec, _ = frame_width(rec)
        n = (w_ref.numel() - WIN_FRAMES) // WIN_HOP + 1
        wr = torch.stack([(w_ref[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES] * e_ref[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES]).sum()
                          / (e_ref[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES].sum() + EPS) for i in range(n)])
        wx = torch.stack([(w_rec[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES] * e_ref[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES]).sum()
                          / (e_ref[i*WIN_HOP:i*WIN_HOP+WIN_FRAMES].sum() + EPS) for i in range(n)])
        temporal[f"{track}|{model}|ref"] = wr.numpy()
        temporal[f"{track}|{model}|rec"] = wx.numpy()

    print(f"[done] {track}")

with open(OUT / "metrics.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wcsv.writeheader(); wcsv.writerows(rows)
with open(OUT / "bands.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
    wcsv.writeheader(); wcsv.writerows(band_rows)
np.savez(OUT / "temporal.npz", **temporal)

# ---- aggregate summary ----
def agg(model, key):
    v = [r[key] for r in rows if r["model"] == model]
    return float(np.mean(v))

print(f"\n{'model':<14}" + "".join(f"{k:>12}" for k in
      ["d_width", "width_bias", "d_pan", "score", "dSM_db", "sisdr_m", "sisdr_s", "mrstft_m", "mrstft_s", "lsd_m", "lsd_s"]))
for model in MODELS:
    dsm = agg(model, "sm_rec_db") - agg(model, "sm_ref_db")
    print(f"{model:<14}"
          f"{agg(model,'d_width'):>12.4f}{agg(model,'width_bias'):>12.4f}{agg(model,'d_pan'):>12.4f}"
          f"{agg(model,'score'):>12.4f}{dsm:>12.2f}{agg(model,'sisdr_m'):>12.2f}{agg(model,'sisdr_s'):>12.2f}"
          f"{agg(model,'mrstft_m'):>12.3f}{agg(model,'mrstft_s'):>12.3f}{agg(model,'lsd_m'):>12.2f}{agg(model,'lsd_s'):>12.2f}")
print("\nделays:", {m: sorted({r['delay'] for r in rows if r['model'] == m}) for m in MODELS})
