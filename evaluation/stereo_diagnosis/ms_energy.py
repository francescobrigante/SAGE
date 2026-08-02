# ===============
# Delta energetico Mid vs Side sui 12 brani in ~/Desktop/commercial (proxy locale
# del training set, che vive su Leonardo). Riporta: gap M/S in dB (time domain),
# frazione di energia della loss che cade su S nel dominio raw e nel dominio
# power-norm (alpha=0.65) in cui vive la ComplexMSE di SAGE, e S-fraction per banda.
# ===============
import sys, glob, math
import torch, torchaudio

SR = 44100
BANDS = [(0, 200), (200, 2000), (2000, 8000), (8000, SR // 2)]
N_FFT, HOP = 2048, 512

rows = []
for path in sorted(glob.glob("/Users/francesco/Desktop/commercial/*/original.wav")):
    name = path.split("/")[-2]
    wav, sr = torchaudio.load(path)
    assert sr == SR and wav.shape[0] == 2, (sr, wav.shape)
    L, R = wav[0], wav[1]
    M, S = (L + R) / math.sqrt(2), (L - R) / math.sqrt(2)

    e_m, e_s = M.pow(2).sum().item(), S.pow(2).sum().item()
    gap_db = 10 * math.log10(e_m / max(e_s, 1e-12))
    frac_raw = e_s / (e_m + e_s)

    win = torch.hann_window(N_FFT)
    stft = lambda x: torch.stft(x, N_FFT, HOP, window=win, return_complex=True)
    SL, SR_ = stft(L), stft(R)

    # dominio della loss: power-norm per canale (beta*|S|^alpha), poi M/S della rappresentazione
    def compress(X, alpha=0.65, beta=0.35):
        mag = X.abs()
        return X / (mag + 1e-8) * beta * mag.pow(alpha)
    CL, CR = compress(SL), compress(SR_)
    CM, CS = (CL + CR) / math.sqrt(2), (CL - CR) / math.sqrt(2)
    frac_comp = (CS.abs().pow(2).sum() / (CM.abs().pow(2).sum() + CS.abs().pow(2).sum())).item()

    # S-fraction per banda (dominio raw STFT)
    SM, SS = (SL + SR_) / math.sqrt(2), (SL - SR_) / math.sqrt(2)
    freqs = torch.linspace(0, SR / 2, SL.shape[0])
    band_fracs = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        es = SS[m].abs().pow(2).sum().item()
        em = SM[m].abs().pow(2).sum().item()
        band_fracs.append(es / max(es + em, 1e-12))

    rows.append((name, gap_db, frac_raw, frac_comp, band_fracs))

print(f"{'track':<28} {'M-S gap dB':>10} {'S-frac raw':>10} {'S-frac loss':>11}  S-frac per banda [<200 | 200-2k | 2k-8k | >8k]")
for name, gap, fr, fc, bf in rows:
    print(f"{name:<28} {gap:>10.1f} {fr:>10.3f} {fc:>11.3f}  [{bf[0]:.3f} | {bf[1]:.3f} | {bf[2]:.3f} | {bf[3]:.3f}]")

import statistics as st
print("-" * 100)
print(f"{'MEDIA':<28} {st.mean(r[1] for r in rows):>10.1f} {st.mean(r[2] for r in rows):>10.3f} {st.mean(r[3] for r in rows):>11.3f}  "
      f"[{st.mean(r[4][0] for r in rows):.3f} | {st.mean(r[4][1] for r in rows):.3f} | {st.mean(r[4][2] for r in rows):.3f} | {st.mean(r[4][3] for r in rows):.3f}]")
