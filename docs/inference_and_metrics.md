# Inference & Metrics Guide

## 1. Inference Configuration
Inference is orchestrated by `inference.py`, which consumes `conf/inference.yaml`. The configuration gives Hydra everything it needs to instantiate the autoencoder, load checkpoints, stream data, and write reconstructions.

### 1.1 Top-Level Keys
| Key | Description |
| --- | --- |
| `checkpoint` | Absolute or project-relative path to the `.ckpt` file used for reconstruction. |
| `model_config_path` | YAML file describing the autoencoder architecture (see `docs/model.md`). |
| `seed`, `deterministic`, `strict_deterministic` | Reproducibility controls passed to `configure_reproducibility`. |
| `sample_rate`, `mono`, `chunked`, `overlap`, `chunk_size`, `segment_seconds`, `segment_frames` | Global encode/decode options applied to both single-file and dataset inference unless overridden elsewhere. |
| `input_wav`, `output_wav` | Enable single-file inference when both are provided. |

### 1.2 Dataset Block
The `dataset` node describes how to read input audio for channel inference and single-file inference fallback:
```yaml
dataset:
  class: ar_spectra.dataset.OnTheFlySTFTDataset
  kwargs:
    audio_dir: /path/to/eval/set
    sample_rate: 44100
    n_fft: 2048
    hop_length: 512
    win_length: 2048
    center: true
    stereo: true
    extensions: [".mp3"]
    cac: false
    skip_broken_files: true
    skip_criteria: ["sample_rate"]
```
Important notes:
- The loader reuses the same schema as training datasets (see `docs/training_dataset.md`), but only the inference file references this block.
- Fields such as `target_frames` may be left `null` for streaming full waveforms during evaluation.
- The block is also used to infer channel counts before model construction.

### 1.3 Dataset Inference Overrides
When `dataset_inference.enabled=true`, the script iterates over the dataset and exports reconstructions:
```yaml
dataset_inference:
  enabled: true
  output_dir: runs/inference/all_losses_cplx
  batch_size: 32
  num_workers: 4
  chunked: false
  overlap: 32
  save_waveforms: true
  save_input_audio: false
  max_batches: null
  max_samples: null
```
Key behaviours:
- `output_dir` is mandatory when batched inference runs; the script writes `<stem>.wav` files here.
- `chunked`, `segment_seconds`, `segment_frames`, `chunk_size`, `overlap` override the global chunking policy for dataset mode only.
- `save_waveforms` controls whether reconstructed audio is written; disable it when running metrics only.
- `save_input_audio` optionally stores the original input alongside reconstructions.
- `max_batches`/`max_samples` provide runtime guards for large datasets.

### 1.4 Running Inference
1. Validate `conf/inference.yaml`, ensuring `checkpoint` and `model_config_path` exist.
2. Launch:
   ```bash
   python inference.py
   ```
3. Monitor the Rich/tqdm output: the progress bar tracks processed samples, and post-fixes expose latent and reconstruction shapes periodically.
4. Reconstructed files appear in the configured `output_dir`; single-file mode writes to `output_wav`.

### 1.5 CLI Overrides
Hydra permits dot-notation overrides at runtime:
```bash
python inference.py checkpoint=/path/to.ckpt dataset.kwargs.extensions='[".wav"]'
```
Useful overrides include chunk sizes (`chunk_size`, `segment_seconds`), mono/stereo switches, and dataset filtering (extensions, skip criteria).

## 2. Metric Evaluation
The repository provides three offline evaluation scripts under `test_metrics/`.

### 2.1 Available Metrics
| Script | Metrics | Environment |
| --- | --- | --- |
| `compute_spectral.py` | STFT loss + SI-SDR | Main training/inference environment |
| `compute_cdpam.py` | CDPAM perceptual distance | Separate env with `cdpam` installed |
| `compute_fad.py` | Fréchet Audio Distance (CLAP embeddings) | Same env as CDPAM |

Each script expects one-to-one filename matches between target and prediction directories (shared stem, comparable extensions).

### 2.2 Manual Invocation Examples
```bash
# Spectral metrics
python test_metrics/compute_spectral.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx \
  --extensions wav,mp3 \
  --csv_out runs/inference/all_losses_cplx/metrics/spectral.csv

# CDPAM (GPU recommended)
python test_metrics/compute_cdpam.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx \
  --device cuda:0 \
  --chunk_size 262144 \
  --csv_out runs/inference/all_losses_cplx/metrics/cdpam.csv

# FAD
python test_metrics/compute_fad.py \
  --target-dir /data/jamendo/test_trimmed \
  --preds-dir runs/inference/all_losses_cplx
```

### 2.3 Automated Pipeline (`compute_all.sh`)
The shell script `test_metrics/compute_all.sh` combines inference and metrics:
1. Activates the main environment, runs `inference.py` with Hydra overrides for checkpoint, dataset, and output directory.
2. Computes STFT/SI-SDR metrics in the same environment.
3. Switches to the metrics environment (set via `DEFAULT_METRICS_ENV` or `--metrics-env`) to compute CDPAM and FAD.
4. Writes results to `<output-dir>/metrics`:
   - `spectral.csv`
   - `cdpam.csv`
   - `fad.txt`

Minimal invocation:
```bash
./test_metrics/compute_all.sh \
  --checkpoint /path/to/model.ckpt \
  --output-dir runs/inference/exp01 \
  --extensions mp3 \
  --cdpam-device cuda:0
```

### 2.4 Environment Checklist
- **Main environment**: PyTorch, torchaudio, torchmetrics, SciPy, tqdm.
- **Metrics environment**: All main dependencies plus `cdpam` and `frechet_audio_distance`. Ensure CUDA/cuDNN compatibility if running on GPU.

### 2.5 Best Practices
1. Filter extensions consistently to guarantee matching pairs.
2. For long clips, set `--chunk_size` in CDPAM to avoid GPU memory exhaustion.
3. Align sample rates between targets and predictions; mismatches raise explicit errors.
4. Version-control metric CSVs when publishing results.
5. Run inference and metrics in controlled environments to ensure reproducibility.

### 2.6 Troubleshooting
- **"No pairs found"**: Predicted files do not share stems with reference files or were filtered out by the extension list.
- **Sample rate mismatch**: Rerun inference with the correct `sample_rate` or resample predictions.
- **"CUDA not available"**: Install GPU-enabled packages or select `--device cpu` to force CPU execution.
