# Training Dataset Guide

## Scope
This document explains how to configure datasets used during **training**. Inference-specific dataset details are covered in `docs/inference_and_metrics.md`, because inference consumes the dataset declaration embedded in `conf/inference.yaml`.

## Entry Point
Training runs load their dataset definition from Hydra configs under `conf/data/`. The canonical profile is `conf/data/data.yaml`, which is merged by default when you invoke `python train.py`.

A typical structure:
```yaml
train_dataset:
  class: ar_spectra.dataset.OnTheFlySTFTDataset
  kwargs: {...}

train_dataloader:
  batch_size: 32
  num_workers: 4
  ...
```

## Required Dataset Keys
| Key | Description |
| --- | --- |
| `class` | Python path to the dataset implementation (`ar_spectra.dataset.OnTheFlySTFTDataset`). |
| `kwargs.audio_dir` | Root directory containing input audio (recursively scanned). |
| `kwargs.sample_rate` | Target sampling rate; the loader resamples if necessary. |
| `kwargs.n_fft`, `kwargs.hop_length`, `kwargs.win_length` | STFT parameters that must match model reconstruction settings. |

## Common Optional Keys
| Key | Purpose |
| --- | --- |
| `extensions` | List of accepted file extensions. Defaults to common audio types. |
| `stereo` | When `false`, audio is mixed down to mono. |
| `cac` | Complex-As-Channels mode packs real/imag parts along the channel dimension. |
| `target_frames` / `length` | Segment duration in STFT frames (ignored when `full_waveform=true`). |
| `full_waveform` | Streams entire waveforms; useful for evaluation, but increases memory use. |
| `skip_broken_files`, `skip_criteria` | Drop files that mismatch the expected channel count or sample rate. |
| `seed` | Base RNG seed for reproducible segment sampling. |

## Dataloader Configuration
Training loaders reside beside the dataset block:
- `batch_size`: global micro-batch size.
- `num_workers`: PyTorch workers for data loading.
- `prefetch_factor`, `persistent_workers`, `pin_memory`, `pin_memory_device`: performance tuning knobs.

Validation loaders follow the same schema (`eval_dataset`, `eval_dataloader`).

## Best Practices
1. **Align STFT parameters** across `conf/data/*` and `conf/model/*` to avoid reconstruction artefacts. If you are using a complex-valued model you need to set the dataset for it to return complex-valued spectra.
2. **Log dataset stats** (number of files, skipped items) especially when using `skip_criteria`.
3. **Use deterministic seeds** when comparing runs. Combine `kwargs.seed` with `trainer.seed` to control all RNG sources.
4. **Monitor cropping behaviour**. If many files are rejected as "too short", adjust `target_frames`, `max_pad_ratio`, or curate the dataset offline.
5. **Version-control dataset configs**. Treat YAML edits as part of the experimental record, the same as model or trainer changes.

## Troubleshooting
- `RuntimeError: No usable files found`: verify `audio_dir` and extension filters point to actual audio files.
- `File too short relative to threshold`: relax `max_pad_ratio` or reduce `target_frames`.
- `Sample rate mismatch` warnings: either resample offline or add `"sample_rate"` to `skip_criteria` to reject incompatible files.

For inference-time dataset overrides (e.g., chunk sizing, output directories), refer to `docs/inference_and_metrics.md`.
