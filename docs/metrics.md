# Metrics & Evaluation Guide

## Overview
EulerAudioBackbone ships with offline metric scripts for reconstruction quality assessment. Each script compares a directory of predictions against reference audio and emits aggregate statistics plus per-file logs.

Metric scripts live under `test_metrics/`:
- `compute_spectral.py` — STFT loss and SI-SDR (PyTorch, torchmetrics, SciPy).
- `compute_cdpam.py` — CDPAM perceptual distance (requires the `cdpam` package and GPU for best performance).
- `compute_fad.py` — Fréchet Audio Distance (FAD) via `frechet_audio_distance` with the CLAP embedding model.

## Running Metrics Individually
All scripts share two mandatory arguments:
- `--target-dir`: reference audio directory.
- `--preds-dir`: directory containing generated reconstructions.

Optional arguments include extension filters and CSV outputs. Examples:
```bash
# Spectral metrics (same environment as training/inference)
python test_metrics/compute_spectral.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx \
  --extensions wav,mp3 \
  --csv_out runs/inference/all_losses_cplx/metrics/spectral.csv

# CDPAM (separate environment with cdpam installed)
python test_metrics/compute_cdpam.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx \
  --device cuda:0 \
  --chunk_size 262144 \
  --csv_out runs/inference/all_losses_cplx/metrics/cdpam.csv

# FAD (CLAP embedding space)
python test_metrics/compute_fad.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx
```

Refer to the script headers for additional flags (e.g., CDPAM chunking, SI-SDR alignment parameters).

## Automated Pipeline
The helper script `test_metrics/compute_all.sh` performs the full workflow:
1. Runs inference via `inference.py` with Hydra overrides.
2. Evaluates spectral metrics in the main virtual environment.
3. Switches to the metrics virtual environment for CDPAM and FAD.
4. Writes CSV/LOG outputs under `<output-dir>/metrics`.

Typical usage:
```bash
./test_metrics/compute_all.sh \
  --checkpoint /path/to/model.ckpt \
  --output-dir runs/inference/exp01 \
  --extensions mp3 \
  --cdpam-device cuda:0
```

### Environment Requirements
- **Main environment** (`DEFAULT_MAIN_ENV` or `--main-env`): contains PyTorch, torchaudio, torchmetrics, SciPy.
- **Metrics environment** (`DEFAULT_METRICS_ENV` or `--metrics-env`): adds `cdpam` and `frechet_audio_distance` dependencies. Keep CUDA/cuDNN versions consistent with the main environment if you plan to run on GPU.

## Output Artefacts
Each script prints the dataset-level mean to stdout and optionally generates structured artefacts:
- `spectral.csv`: columns `target_file`, `pred_file`, `stft_loss`, `si_sdr`.
- `cdpam.csv`: columns `target_file`, `pred_file`, `score`.
- `fad.txt`: CLAP-based FAD score with a human-readable summary.

Metric outputs can be ingested by downstream experiment trackers or aggregated manually to compare checkpoints.

## Best Practices
1. **Align sample rates** between target and prediction directories; mismatches raise explicit errors.
2. **Use extension filters** to guarantee one-to-one filename matches when predictions are stored in mixed formats.
3. **Chunk long clips** for CDPAM via `--chunk_size` to reduce GPU memory pressure.
4. **Avoid overlapping environments.** Keep CDPAM/FAD dependencies separate to prevent conflicts with the main training environment.
5. **Version-control metric CSVs** alongside inference outputs when reporting results.

## Troubleshooting
- `No couples found`: ensure filenames match (same stem) and extensions align with the filter set.
- `Sample rate mismatch`: reconstructions were generated at a different rate; resample or rerun inference with the correct `sample_rate`.
- `CUDA non available`: the script falls back to CPU. Install GPU-enabled dependencies or specify `--device cpu`.
