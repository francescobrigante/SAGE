# ===============
# Plots for the stereo-collapse analysis. Reads results/ produced by
# measure_imaging.py plus the wavs, writes PNGs into plots/.
#   uv run python evaluation/stereo_diagnosis/make_plots.py [root_dir]
# ===============
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from stereo_imaging import align

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/francesco/Desktop/commercial_songs")
RES = Path(__file__).parent / "results"
PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

SR, N_FFT, HOP = 44100, 2048, 512
MODELS = ["sage", "same", "sao-vae", "codicodec", "music2latent"]
# fixed categorical slot order (validated palette, light mode)
COLORS = {"sage": "#2a78d6", "same": "#eb6834", "sao-vae": "#1baf7a",
          "codicodec": "#eda100", "music2latent": "#e87ba4"}
INK, MUTED, GRID = "#1a1a19", "#5f5e56", "#e4e3db"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11,
})

rows = list(csv.DictReader(open(RES / "metrics.csv")))
band_rows = list(csv.DictReader(open(RES / "bands.csv")))
temporal = np.load(RES / "temporal.npz")


def agg(model, key, src=rows):
    return float(np.mean([float(r[key]) for r in src if r["model"] == model]))


# ── 1. summary: d_width + width_bias ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
y = np.arange(len(MODELS))[::-1]
for ax, key, title in zip(axes, ["d_width", "width_bias"],
                          ["Errore di larghezza  d_width  (0 = perfetto)",
                           "Bias di larghezza  (<0 collasso, >0 widening)"]):
    vals = [agg(m, key) for m in MODELS]
    ax.barh(y, vals, height=0.55, color=[COLORS[m] for m in MODELS])
    ax.set_yticks(y, MODELS)
    ax.set_title(title, loc="left")
    ax.axvline(0, color=MUTED, lw=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.3f}" if key == "width_bias" else f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.margins(x=0.15)
fig.suptitle("Imaging stereo — media su 12 brani (allineati, pesati per energia)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(PLOTS / "summary_imaging.png", dpi=150)
plt.close(fig)

# ── 2. width_bias per banda ──────────────────────────────────────────────────
bands = ["lt200", "200_2k", "2k_8k", "gt8k"]
band_labels = ["<200 Hz", "200–2k", "2k–8k", ">8 kHz"]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
x = np.arange(len(bands))
bw = 0.15
for i, m in enumerate(MODELS):
    vals = [agg(m, "width_bias", [r for r in band_rows if r["band"] == b]) for b in bands]
    ax.bar(x + (i - 2) * bw, vals, width=bw - 0.02, color=COLORS[m], label=m)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(x, band_labels)
ax.set_ylabel("width bias")
ax.set_title("Bias di larghezza per banda — il collasso di SAGE è broadband", loc="left")
ax.legend(ncols=5, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.02))
fig.tight_layout()
fig.savefig(PLOTS / "bands_width_bias.png", dpi=150)
plt.close(fig)

# ── 3. temporal: w_rec vs w_ref small multiples ──────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), sharex=True, sharey=True)
tracks = sorted({k.split("|")[0] for k in temporal.files})
for ax, m in zip(axes, MODELS):
    xs = np.concatenate([temporal[f"{t}|{m}|ref"] for t in tracks])
    ys = np.concatenate([temporal[f"{t}|{m}|rec"] for t in tracks])
    a, b = np.polyfit(xs, ys, 1)
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1, ls="--")
    ax.scatter(xs, ys, s=7, alpha=0.35, color=COLORS[m], edgecolors="none")
    ax.set_title(f"{m}\nw_rec = {a:.2f}·w_ref{b:+.2f}", fontsize=9.5)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
    ax.set_xlabel("larghezza originale")
axes[0].set_ylabel("larghezza ricostruita")
fig.suptitle("Larghezza per finestre di 1 s — la diagonale è la ricostruzione perfetta", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(PLOTS / "temporal_width.png", dpi=150)
plt.close(fig)

# ── 4. S-channel spectrograms per track ──────────────────────────────────────
def s_spec_db(x: torch.Tensor) -> np.ndarray:
    S = (x[0] - x[1]) / math.sqrt(2)
    win = torch.hann_window(N_FFT)
    P = torch.stft(S, N_FFT, HOP, window=win, return_complex=True).abs().square()
    return (10 * torch.log10(P + 1e-10)).numpy()


for track_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    track = track_dir.name
    ref_full, _ = torchaudio.load(track_dir / "original.wav")
    panels = [("original", s_spec_db(ref_full))]
    for m in MODELS:
        rec_full, _ = torchaudio.load(track_dir / f"{m}.wav")
        ref_a, rec_a = align(ref_full, rec_full)
        panels.append((m, s_spec_db(rec_a)))
    vmax = panels[0][1].max()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, (name, spec) in zip(axes.flat, panels):
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmax - 80, vmax=vmax,
                       extent=(0, spec.shape[1] * HOP / SR, 0, SR / 2000))
        ax.set_title(name, fontsize=10)
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("tempo (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("freq (kHz)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="dB")
    fig.suptitle(f"Canale Side (L−R) — {track}", x=0.01, ha="left")
    fig.savefig(PLOTS / f"s_spec_{track}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {track}")

print(f"\nsaved to {PLOTS}")
