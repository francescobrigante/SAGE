# Dataset Configuration Guide

## Purpose
This guide explains how to describe datasets for both training and inference when working with EulerAudioBackbone. Dataset specifications determine how audio is discovered on disk, preprocessed (resampling, cropping, STFT generation), and delivered to the autoencoder.

## Configuration Entry Points
- **Training** uses the `data` group in `conf/data/*.yaml`. The canonical file is `conf/data/data.yaml`.
- **Inference** reads its dataset description from `conf/inference.yaml`.
- The optional `dataset_inference` block in `conf/inference.yaml` can override loader behaviour exclusively for batched inference.

All dataset blocks share the same schema because they are parsed by `prepare_dataset_spec` and ultimately instantiate `ar_spectra.dataset.OnTheFlySTFTDataset`.

## Required Keys
| Key | Description |
| --- | --- |
| `class` | Python path of the dataset implementation. For STFT streaming use `ar_spectra.dataset.OnTheFlySTFTDataset`. |
| `kwargs.audio_dir` | Root directory containing the input audio files. Recursion is enabled. |
| `kwargs.sample_rate` | Target sampling rate after resampling. |
| `kwargs.n_fft` / `hop_length` / `win_length` | STFT parameters used for both dataset preparation and model reconstruction. |

## Common Optional Keys
| Key | Effect |
| --- | --- |
| `kwargs.extensions` | Sequence of file extensions accepted (e.g. `[".wav", ".flac", ".mp3"]`). Defaults to a wide audio set. |
| `kwargs.stereo` | `true` keeps stereo channels; `false` mixes down to mono. |
| `kwargs.cac` | When `true` packs real/imag parts as channels (Complex-As-Channels). |
| `kwargs.target_frames` | Number of STFT frames per sample (used when `full_waveform=false`). |
| `kwargs.full_waveform` | When `true`, streams entire waveforms without cropping. |
| `kwargs.skip_broken_files` / `skip_criteria` | Filtering policies for incompatible sample rates, channel counts, etc. |
| `kwargs.seed` | Base RNG seed controlling crop selection. |

### Loader Overrides (Inference Only)
The `dataset_inference` block accepts additional options:
- `enabled`: toggles batched inference. When `false`, only `input_wav`/`output_wav` execution runs.
- `output_dir`: directory for reconstructed waveforms.
- `batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`: standard PyTorch DataLoader knobs.
- `chunked`, `segment_seconds`, `segment_frames`, `chunk_size`, `overlap`: override chunking behaviour for inference separately from training defaults.
- `max_batches`, `max_samples`: safety limits to bound inference runtime.
- `save_waveforms`, `save_input_audio`: control which artefacts are written to disk.

## Example Snippet
```yaml
dataset:
  class: ar_spectra.dataset.OnTheFlySTFTDataset
  kwargs:
    audio_dir: /data/jamendo/test_trimmed
    sample_rate: 44100
    n_fft: 2048
    hop_length: 512
    win_length: 2048
    center: true
    normalized: false
    extensions: [".mp3"]
    stereo: true
    cac: false
    skip_broken_files: true
    skip_criteria: ["sample_rate"]
    target_frames: null

dataset_inference:
  enabled: true
  output_dir: runs/inference/all_losses_cplx
  batch_size: 32
  num_workers: 4
  shuffle: false
  save_waveforms: true
```

## Best Practices
1. **Match STFT parameters** between training and inference. Downstream reconstruction depends on consistent `n_fft`, `hop_length`, and `win_length`.
2. **Explicitly enumerate extensions** when your dataset mixes lossy and lossless formats to avoid unpredictable decoding quality.
3. **Use `skip_criteria` cautiously**. Removing files for channel or rate mismatches helps maintain homogeneous batches, but log the exclusions so you understand coverage.
4. **Full waveform mode** is ideal for evaluation, but increases GPU memory use. Prefer chunked operation for training and long-form inference.
5. **Set seeds** whenever reproducibility matters. Both `kwargs.seed` and the global `seed` in `conf/inference.yaml` or `trainer.seed` in training configs cooperate to make sampling deterministic.

## Troubleshooting
- `RuntimeError: No usable files found`: confirm `audio_dir` is correct and accessible, and that `extensions` matches the files on disk.
- `File too short relative to threshold`: relax `max_pad_ratio` or provide longer excerpts; the dataset protects against overly short segments.
- `Sample rate mismatch` warnings: add `"sample_rate"` to `skip_criteria` to discard incompatible files, or resample offline for consistency.
