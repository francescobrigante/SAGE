#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from torchmetrics.audio.sdr import SignalDistortionRatio as SISDRMetric
from ar_spectra.training_utils.losses.auraloss import STFTLoss
import numpy as np
from scipy.signal import correlate
from ar_spectra.training_utils.reproducibility import configure_reproducibility

configure_reproducibility(seed=42, deterministic=False, strict_deterministic=False)

def crop_to_min_length(a, b):
    L = min(a.shape[-1], b.shape[-1])
    return a[..., :L], b[..., :L]

def load_audio_tensor(path):
    """Load an audio file as a torch tensor of shape (1, channels, frames)."""
    audio, sr = sf.read(path, always_2d=False)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim == 1:
        tensor = torch.from_numpy(audio)[None, None, :]
    else:
        tensor = torch.from_numpy(audio.T)[None, :, :]

    return tensor.contiguous(), sr



def estimate_time_shift_torch(x, y, sr, max_shift_seconds=1.0):
    """
    Estimate the relative time shift (lag) between two audio tensors using
    cross-correlation computed via FFT.

    The goal is to find an integer lag (in samples) such that the predicted
    signal `y` is maximally aligned with the reference signal `x`. We do this
    by maximizing the (normalized) similarity between the two waveforms over a
    bounded range of lags.

    Mathematically, for two discrete-time signals x[n] and y[n], the
    cross-correlation of y with respect to x at lag ℓ is

        r_yx[ℓ] = sum_n y[n] * x[n + ℓ].

    For each lag ℓ, r_yx[ℓ] measures how similar y is to a time-shifted version
    of x. The lag ℓ that maximizes r_yx[ℓ] is (approximately) the lag that
    minimizes the mean squared error between y[n] and x[n + ℓ]. In other words,
    it is the time shift that best aligns the two signals.

    This function:
      1. Averages over channels to obtain mono waveforms (shape (T,)).
      2. Truncates both signals to the same length T (minimum of their lengths).
      3. Removes the DC component from both signals (mean subtraction), which
         stabilizes the correlation.
      4. Computes the full cross-correlation r_yx[ℓ] using FFT:
             correlate(y, x) = conv(y, flip(x))
         implemented as:
             irfft( rfft(y) * rfft(flip(x)) ).
      5. Constructs the corresponding array of lags ℓ in samples.
      6. Restricts the search to |ℓ| <= max_shift_seconds * sr if
         max_shift_seconds > 0, so only "reasonable" delays are considered.
      7. Returns the lag (in samples) with the maximum cross-correlation value.

    By convention in this implementation:
      - `x` is the reference (target) signal.
      - `y` is the estimated (predicted) signal.
      - A positive lag L > 0 means that `y` is delayed with respect to `x`
        (i.e., y[t] ≈ x[t - L]).
      - A negative lag L < 0 means that `y` is advanced with respect to `x`
        (i.e., y[t] ≈ x[t - L], with L < 0).

    This lag can then be used to time-align the signals before computing
    waveform-based metrics such as SI-SDR or STFT-based losses.

    Args:
        x (torch.Tensor): Reference audio tensor of shape (1, C, T).
        y (torch.Tensor): Estimated audio tensor of shape (1, C, T).
        sr (int): Sampling rate in Hz.
        max_shift_seconds (float, optional): Maximum absolute lag (in seconds)
            to search over when estimating the delay. If <= 0, the full
            correlation support is used.

    Returns:
        int: Estimated lag in samples. Positive values indicate that `y` is
        delayed relative to `x`, negative values indicate that `y` is advanced.
    """
    device = x.device

    # average over channels: (1, C, T) -> (T,)
    x_m = x.mean(dim=1).squeeze(0)
    y_m = y.mean(dim=1).squeeze(0)

    # Ensure both signals share the same temporal support
    T = min(x_m.shape[-1], y_m.shape[-1])
    x_m = x_m[..., :T]
    y_m = y_m[..., :T]

    # Remove the DC component to stabilize the correlation
    x_m = x_m - x_m.mean()
    y_m = y_m - y_m.mean()

    Nx = Ny = T
    L = Nx + Ny - 1  # lunghezza della correlazione 'full' = 2T - 1

    # FFT length as a power of two >= L
    n_fft = 1
    while n_fft < L:
        n_fft *= 2

    # correlate(y, x) == conv(y, flip(x))
    x_flip = torch.flip(x_m, dims=[-1])

    Y = torch.fft.rfft(y_m, n=n_fft)
    X_flip = torch.fft.rfft(x_flip, n=n_fft)

    corr_full = torch.fft.irfft(Y * X_flip, n=n_fft)  # (n_fft,)
    corr_full = corr_full[:L]  # (L,)

    # lags as in scipy.signal.correlate(y, x, mode='full'):
    # index k corresponds to lag = k - (Nx - 1)
    lags = torch.arange(-Nx + 1, Ny, device=device)  # [-T+1, ..., T-1], shape (L,)

    max_shift = int(max_shift_seconds * sr)
    if max_shift > 0:
        mask = (lags >= -max_shift) & (lags <= max_shift)
    else:
        # No constraint on the search window
        mask = torch.ones_like(lags, dtype=torch.bool, device=device)

    if mask.any():
        corr_masked = corr_full[mask]
        lags_masked = lags[mask]
        best_idx = torch.argmax(corr_masked)
        best_lag = lags_masked[best_idx]
    else:
        best_lag = lags[torch.argmax(corr_full)]

    return int(best_lag.item())


def align_by_shift(x, y, lag):
    """
    Align two audio tensors by applying the estimated lag.

    Args:
        x (torch.Tensor): Reference audio tensor of shape (1, C, T).
        y (torch.Tensor): Estimated audio tensor of shape (1, C, T).
        lag (int): Lag in samples. Positive values denote that y is delayed with respect to x.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pair of temporally aligned tensors.
    """
    if lag > 0:
        # y is delayed: remove the first 'lag' samples from y
        y_aligned = y[..., lag:]
        x_aligned = x[..., :y_aligned.shape[-1]]
    elif lag < 0:
        # y is advanced: remove the first '-lag' samples from x
        lag = -lag
        x_aligned = x[..., lag:]
        y_aligned = y[..., :x_aligned.shape[-1]]
    else:
        x_aligned, y_aligned = crop_to_min_length(x, y)

    return x_aligned, y_aligned


def match_pairs(target_dir, preds_dir, allowed_ext=None):
    target_dir = Path(target_dir)
    preds_dir = Path(preds_dir)

    target_map = {}
    for f in target_dir.iterdir():
        if f.is_file():
            stem = f.stem
            ext = f.suffix.lower().lstrip(".")
            if allowed_ext and ext not in allowed_ext:
                continue
            target_map.setdefault(stem, []).append(f)

    pairs = []
    for g in preds_dir.iterdir():
        if g.is_file():
            stem = g.stem
            ext = g.suffix.lower().lstrip(".")
            if allowed_ext and ext not in allowed_ext:
                continue
            if stem in target_map:
                pairs.append((target_map[stem][0], g))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Compare SI-SDR and STFTLoss between two directories (predictions vs. targets).")
    parser.add_argument("--target-dir", default="/home/cerovaz/repos/data/jamendo_full/test_trimmed")
    parser.add_argument("--preds-dir", default="/home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/all_losses_cplx_24.")
    parser.add_argument("--extensions", type=str, default="")
    parser.add_argument("--csv_out", type=str, default="")
    args = parser.parse_args()

    # Select computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    allowed_ext = None
    if args.extensions.strip():
        allowed_ext = {e.strip().lower().lstrip(".") for e in args.extensions.split(",") if e.strip()}

    pairs = match_pairs(args.target_dir, args.preds_dir, allowed_ext=allowed_ext)
    if not pairs:
        print("Nessuna coppia trovata.")
        return

    sisdr_metric = SISDRMetric().to(device)
    stft_cache = {}
    si_sdr_sum = torch.zeros((), device=device)
    stft_sum = torch.zeros((), device=device)
    count = 0
    per_file = []

    pbar = tqdm(pairs, desc="Calcolo metriche", unit="pair")

    for tpath, ppath in pbar:
        try:
            wav_target, sr_target = load_audio_tensor(str(tpath))
            wav_pred, sr_pred = load_audio_tensor(str(ppath))

            if sr_target != sr_pred:
                raise ValueError(f"Sample rate mismatch ({sr_target} vs {sr_pred}).")
            if wav_target.shape[1] != wav_pred.shape[1]:
                raise ValueError(
                    f"Numero di canali differente ({wav_target.shape[1]} vs {wav_pred.shape[1]})."
                )

            wav_target = wav_target.to(device)
            wav_pred   = wav_pred.to(device)
            
            # 1) Coarsely crop to the shortest common length
            wav_target, wav_pred = crop_to_min_length(wav_target, wav_pred)

            # 2) Estimate temporal delay
            lag = estimate_time_shift_torch(wav_target, wav_pred, sr_target, max_shift_seconds=1.0)

            # 3) Align the tensors according to the estimated lag
            wav_target, wav_pred = align_by_shift(wav_target, wav_pred, lag)


            if sr_target not in stft_cache:
                stft_cache[sr_target] = STFTLoss(
                    fft_size=2048,
                    hop_size=512,
                    win_length=2048,
                    perceptual_weighting=True,
                    w_log_mag=1.0,
                    sample_rate=sr_target,
                    reduction="mean",
                ).to(device)
            stft_loss_fn = stft_cache[sr_target]

            with torch.no_grad():
                stft_value = stft_loss_fn(wav_pred, wav_target)
                if stft_value.ndim != 0:
                    stft_value = stft_value.flatten().mean()
                sisdr_metric.reset()
                si_sdr_value = sisdr_metric(wav_pred, wav_target)
                if si_sdr_value.ndim != 0:
                    si_sdr_value = si_sdr_value.flatten().mean()
        except Exception as e:
            print(f"Errore su coppia {tpath.name}/{ppath.name}: {e}")
            continue


        stft_sum = stft_sum + stft_value
        si_sdr_sum = si_sdr_sum + si_sdr_value
        count += 1
        mean_stft = stft_sum / count
        mean_si_sdr = si_sdr_sum / count
        pbar.set_postfix(
            {
                "mean_stft": f"{mean_stft.item():.4f}",
                "mean_si_sdr": f"{mean_si_sdr.item():.4f}",
            }
        )

        per_file.append({
            "target_file": tpath.name,
            "pred_file": ppath.name,
            "stft_loss": stft_value.item(),
            "si_sdr": si_sdr_value.item(),
        })

    if count == 0:
        print("Nessun punteggio calcolato.")
        return

    final_stft = (stft_sum / count).item()
    final_si_sdr = (si_sdr_sum / count).item()
    print(f"\nSTFTLoss media: {final_stft:.6f} su {count} coppie.")
    print(f"SI-SDR medio: {final_si_sdr:.6f} su {count} coppie.")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["target_file", "pred_file", "stft_loss", "si_sdr"],
            )
            writer.writeheader()
            writer.writerows(per_file)
        print(f"Salvato CSV: {out_path}")

if __name__ == "__main__":
    main()