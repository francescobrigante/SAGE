# Model Configuration Guide

## Purpose
This document details how to describe the autoencoder architecture that powers EulerAudioBackbone. Model definitions are written in Hydra YAML and consumed by `AutoEncoder.from_config`, enabling interchangeable encoder/decoder pairs, bottlenecks, and pre/post transforms.

## Configuration Entry Points
- Default model specification: `conf/model/SEANet_cplx_model.yaml`.
- Custom variants can be added under `conf/model/` and selected via `model=<name>` on the training command line or by editing `conf/inference.yaml` for evaluation.

## Core Structure
Every model YAML exposes a `model` node with the following fields:
```yaml
model:
  autoencoder:
    encoder: {...}
    decoder: {...}
    bottleneck: {...}
    pre_transform: {...}   # optional
    post_transform: {...}  # optional
  stft:
    n_fft: 2048
    hop_length: 512
    win_length: 2048
    center: true
```

### Encoder / Decoder Blocks
Each block supports multiple declaration styles:
- **Config dictionaries** with `class` and `kwargs`.
- Fully qualified class paths (string).
- Direct class or instance references (Python-side construction).

When specified in YAML, the recommended pattern is:
```yaml
encoder:
  class: ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d
  kwargs:
    input_size: auto
    ratios: [[2, 2], [2, 2], [2, 2]]
```

Use the sentinel `auto` for parameters whose value depends on dataset inspection (e.g., spectrogram channel count). `resolve_auto_channels` patches these fields automatically once data is available.

### Bottleneck Options
Common choices include:
- `ar_spectra.models.bottlenecks.IdentityBottleneck`: deterministic autoencoder.
- `ar_spectra.models.bottlenecks.VAEBottleneck`: variational latent space.
- `ar_spectra.models.bottlenecks.SkipBottleneck`: bypass for residual learning.

Match bottleneck expectations with encoder output channels (e.g., VAE requires doubling for mean/logvar). The constructor validates dimensions to prevent silent mismatches.

### Pre/Post Transforms
Spectrogram normalization and denormalization belong inside the `autoencoder` block:
- `pre_transform`: applied before encoding (e.g., power scaling, log-magnitude transforms).
- `post_transform`: applied after decoding.

Example:
```yaml
pre_transform:
  class: ar_spectra.models.modules.NormalisePower
  kwargs:
    epsilon: 1e-4
```

### STFT Metadata
The `stft` sub-dictionary mirrors dataset STFT parameters. During inference `autoenc.set_stft_config` is called with this block to ensure waveform reconstruction uses the same analysis window.

## Extending the Model Zoo
1. Copy an existing YAML and adjust encoder/decoder classes.
2. Implement new modules under `ar_spectra.models.*` with explicit `forward` signatures.
3. Register any complex-aware layers in `ar_spectra.models.modules` so they can be referenced by name.

## Best Practices
- **Keep encoder/decoder symmetry.** Matching down/up-sampling factors prevents checkerboard artefacts.
- **Leverage complex-aware layers.** Modules in `ar_spectra.models.modules` respect real/imag coupling; mixing real-only layers can degrade phase fidelity.
- **Document latent dimensionality.** Complex-valued latents often require specifying `pack_complex` logic; ensure inference knows whether to pack real/imag pairs.
- **Version control YAMLs.** Treat configuration changes as experiments; store them under descriptive filenames and log the commit hash alongside checkpoints.

## Troubleshooting
- `Model configuration must contain a 'model' section`: ensure the YAML root has the `model` key.
- `Invalid STFT configuration`: verify `n_fft`, `hop_length`, and `win_length` are coherent (e.g., `win_length <= n_fft`).
- `Failed to resolve auto channels`: check that the dataset spec exposes compatible channel counts or specify concrete numbers instead of `auto`.
