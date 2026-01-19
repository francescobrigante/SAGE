# Model Configuration Guide

## Purpose
This document details how to describe the autoencoder architecture that powers EulerAudioBackbone. Model definitions are written in Hydra YAML using the `_target_` instantiation pattern and consumed by `AutoEncoder.from_config`, enabling interchangeable encoder/decoder pairs, bottlenecks, and pre/post transforms.

## Configuration Entry Points
- Default model specification: `conf/model/simple_transformer_AE.yaml`.
- Custom variants can be added under `conf/model/` and selected via `model=<name>` on the training command line or by editing `conf/inference.yaml` for evaluation.

## Core Structure
Every model YAML exposes a `model` node with the following fields:
```yaml
model:
  autoencoder:
    return_latent: false
    pre_transform: {...}   # optional
  encoder:
    _target_: ar_spectra.models.autoencoders...
    input_size: 2          # MUST match data.model_channels
    ...
  decoder:
    _target_: ar_spectra.models.autoencoders...
    channels: 2            # MUST match data.model_channels
    ...
  bottleneck:
    _target_: ar_spectra.models.bottlenecks.VAEBottleneck
    skip_bottleneck: true
```

### Encoder / Decoder Blocks
Each block uses Hydra's `_target_` instantiation pattern:

```yaml
encoder:
  _target_: ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d
  input_size: 2              # stereo without CAC
  ratios: [[2, 2], [2, 2], [2, 2]]
```

  **Return convention:** encoder modules must return ``(latents, encoder_info)``
  where ``encoder_info`` is a dictionary that at least exposes
  ``feature_shape`` when the decoder needs spatial hints (e.g., transformer
  fold/unfold). The autoencoder container propagates this info and, when a
  bottleneck is active, exposes it as an optional third element in the returned
  tuple.

### Channel Configuration (MANUAL)
**Important:** Channel parameters must be set manually to match the dataset configuration.

| Dataset Settings | audio_channels | model_channels |
|-----------------|----------------|----------------|
| stereo=true, cac=false | 2 | 2 |
| stereo=true, cac=true | 2 | 4 |
| stereo=false, cac=false | 1 | 1 |
| stereo=false, cac=true | 1 | 2 |

Set in `conf/data/data.yaml`:
```yaml
audio_channels: 2
model_channels: 2
```

And ensure encoder/decoder configs match:
```yaml
encoder:
  input_size: 2    # = model_channels
decoder:
  channels: 2      # = model_channels
```

### Trainer-required parameters
The training loop expects the following arguments to be present in the encoder/decoder configs:

- `input_size` on the encoder.
- `channels` on the decoder.
- `is_complex` on both encoder and decoder (optional).

`is_complex` is used during training to decide whether audio logged to Weights & Biases should be packed/unpacked as real/imaginary channel pairs; set it to mirror the dataset output representation.

When implementing new autoencoders, prefer subclassing the base classes in [ar_spectra/models/autoencoders/abstract_ae.py](ar_spectra/models/autoencoders/abstract_ae.py). This keeps constructor signatures aligned with the required `input_size`, `channels`, and `is_complex` fields and avoids mismatches at training time.

### Bottleneck Options
Common choices include:
- `ar_spectra.models.bottlenecks.IdentityBottleneck`: deterministic autoencoder.
- `ar_spectra.models.bottlenecks.VAEBottleneck`: variational latent space.
- `ar_spectra.models.bottlenecks.SkipBottleneck`: bypass bottleneck. It's the same as using IdentityBottleneck.

Match bottleneck expectations with encoder output channels (e.g., VAE requires doubling for mean/logvar). The constructor validates dimensions to prevent silent mismatches. 

> **Important:** picking `VAEBottleneck` inside the model config only enables latent sampling and returns the per-batch KL statistic (`loss_info["kl"]`). The actual KL regulariser is injected on the trainer side through `loss_config.bottleneck.weights.kl`. Without that entry the engine falls back to a tiny default (1e-6), so remember to set an explicit weight in `conf/trainer/*.yaml` whenever you expect a meaningful KL term during training.

### Pre/Post Transforms
Spectrogram normalization and denormalization belong inside the `autoencoder` block:
- `pre_transform`: applied before encoding (e.g., power scaling, log-magnitude transforms).

Note: at the moment we support `power_norm`, `log_mag`, or `none`. `log_mag` in our experiments doesn't seem to converge.  

Example:
```yaml
pre_transform:
  class: ar_spectra.models.modules.NormalisePower
  kwargs:
    epsilon: 1e-4
```

## Extending the Model Zoo
1. Copy an existing YAML and adjust encoder/decoder classes.
2. Implement new modules under `ar_spectra.models.*` with explicit `forward` signatures.
3. Register any complex-aware layers in `ar_spectra.modules` so they can be referenced by name.

## Best Practices
- **Complex Compatibility** If using an original model in `ar_spectra.model` the parameter `is_complex` must match the dataset output dtype. E.g. if the dataset outputs a complex-valued spectrogram the flag `is_complex` must be activated, if the dataset outputs a real-valued spectogram (`cac` flag inside data) the flag of the model `is_complex` must be de-activated. 
- **Keep encoder/decoder symmetry.** Matching down/up-sampling factors prevents checkerboard artefacts.
- **Leverage complex-aware layers.** Modules in `ar_spectra.models.modules` respect real/imag coupling; mixing real-only layers can degrade phase fidelity.
- **Document latent dimensionality.** Complex-valued latents often require specifying `pack_complex` logic; ensure inference knows whether to pack real/imag pairs.
- **Version control YAMLs.** Treat configuration changes as experiments; store them under descriptive filenames and log the commit hash alongside checkpoints.

## Troubleshooting
- `Model configuration must contain a 'model' section`: ensure the YAML root has the `model` key.
- `Failed to resolve auto channels`: check that the dataset spec exposes compatible channel counts or specify concrete numbers instead of `auto`.

