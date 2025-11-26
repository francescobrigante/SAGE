# 🎧 EulerAudioBackbone: Generative Backbone for Complex-Valued Audio Spectrograms

## Overview
**ComplexVAE** is a research-oriented backbone for **complex-valued Variational Autoencoders (VAE)** operating directly on **complex STFT spectrograms**.  
It is designed as part of a broader ICML project exploring **complex-valued neural architectures** for:
- Neural Audio Coding  
- Generative Modeling (VAE → Diffusion)  
- Source Separation  

Unlike prior models (e.g., [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0), [Music2Latent2](https://anonymous2732.github.io/music2latent2/), [CoDiCodec](https://sonycslparis.github.io/codicodec)), which employ **real-valued** architectures on **complex inputs**, this repository investigates **end-to-end complex-valued** formulations as true backbones for generative models.

---

## 🔬 Research Goals
- Build and release **open-source complex-valued backbones** for audio representation learning.  
- Conduct systematic **ablation studies** on:
  - Real vs. Complex architectures (ConvNeXt, Conformer, etc.)
  - Reconstruction quality on complex spectrograms
  - Impact of complex-valued normalization, activations, and attention.
- Enable **generalizable backbones** across multiple audio domains (music, speech, ambient, etc.).

---

## 🧠 Architecture
- Encoder/Decoder designed for **complex STFT tensors** `(B, C, F, T)`  
- Supports both **complex-valued convolutions** and **attention-based variants**
- Modular structure for flexible integration into:
  - Latent diffusion models
  - Neural audio codecs
  - Source separation systems  

---

## ⚙️ Planned Experiments
| Task | Backbone | Domain | Evaluation |
|------|-----------|---------|-------------|
| Neural Audio Coding | EuleroDec / complex | Jamendo / DNS / LibriTTS | SI-SDR, FAD, ViSQOL |
| VAE for Diffusion | ComplexVAE | Multi-domain | Reconstruction, KL, FAD |
| Source Separation | Complex Swin  / BSRoformer-C | Music & Speech | SI-SDR, SDRi |

---

## 📚 Related Works
- **Stable Audio Open** — Latent diffusion on waveforms (44.1 kHz stereo).  
- **Music2Latent2** — Continuous autoencoder with summary embeddings and autoregressive decoding.  
- **CoDiCodec** — Unified continuous/discrete consistency model with FSQ and summary embeddings.  
This project extends these paradigms into the **complex domain**, enabling explicit phase modeling and analyticity constraints.

---

## 🧩 Repository Structure and Info

## AutoEncoder Construction Guide

This project lets you build the AutoEncoder in multiple interchangeable ways. All approaches produce an AutoEncoder instance; pick the one that best fits your workflow or tooling.

### 1) From a dict “spec” (same shape as Hydra YAML blocks)

```python
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(
    encoder={
        "class": "ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d",
        "kwargs": {"input_size": 128}
    },
    decoder={
        "class": "ar_spectra.models.autoencoders.SeaNET_AE.SEANetDecoder2d",
        "kwargs": {"input_size": 64}
    },
    bottleneck={
        "class": "ar_spectra.models.bottlenecks.VAEBottleneck",
        "kwargs": {"skip_bottleneck": True}
    }
)
```

### 2) Passing classes directly (no constructor args)

```python
from ar_spectra.models.autoencoders.SeaNET_AE import SEANetEncoder2d, SEANetDecoder2d
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(encoder=SEANetEncoder2d, decoder=SEANetDecoder2d)
```

### 3) Passing class paths as strings (no constructor args)

```python
from ar_spectra.models.autoencoder import AutoEncoder

ae = AutoEncoder(
    encoder="ar_spectra.models.autoencoders.SeaNET_AE.SEANetEncoder2d",
    decoder="ar_spectra.models.autoencoders.SeaNET_AE.SEANetDecoder2d",
)
```

### 4) Passing prebuilt instances

```python
from ar_spectra.models.autoencoders.SeaNET_AE import SEANetEncoder2d, SEANetDecoder2d
from ar_spectra.models.autoencoder import AutoEncoder

enc = SEANetEncoder2d(input_size=128)
dec = SEANetDecoder2d(input_size=64)
ae = AutoEncoder(encoder=enc, decoder=dec)
```

### 5) From a Hydra YAML config

```python
from omegaconf import OmegaConf
from ar_spectra.models.autoencoder import AutoEncoder

cfg = OmegaConf.load("conf/model/SEANet_cplx_model.yaml")
ae = AutoEncoder.from_config(cfg["model"])
```

> **Heads-up:** the spectrogram pre/post normalization now lives inside the model block. When you need power or log magnitude normalization, add the `pre_transform` entry under `model.autoencoder` (see the default `conf/model/SEANet_cplx_model.yaml`). Trainer-level overrides are no longer applied automatically.

---

## How the pieces work together

- AutoEncoder.encode(x): runs encoder(x) → optional bottleneck.encode(latents) → latents.
- AutoEncoder.decode(z): runs optional bottleneck.decode(z) → decoder(z) → reconstruction.
- AutoEncoder.istft(spec, ...): converts complex/RI spectrogram back to waveform using provided STFT params.

### Dimension checks
To prevent silent shape mismatches, the constructor validates channels:
- Without a VAE bottleneck: encoder output channels must equal decoder input channels.
- With VAEBottleneck: encoder channels must be 2 × decoder input (mean + logvar).
- With SkipBottleneck: encoder channels must equal decoder input.

---

## 🚀 Inference Workflow

The inference stack is controlled entirely by `conf/inference.yaml` and the helper script `inference.py`. Training configs (`conf/data/*.yaml`) are no longer required at runtime: the dataset defined inside `inference.yaml` is used both to infer channel dimensions and to feed evaluation batches.

### Key Blocks in `conf/inference.yaml`
- `checkpoint`: absolute or project-relative path to the `.ckpt` file to restore.
- `model_config_path`: YAML describing the architecture (must contain a `model` section).
- `sample_rate`, `mono`, `chunked`, `overlap`, `chunk_size`, `segment_*`: global encode/decode options.
- `dataset`: spec used to infer model/audio channels and (when `dataset_inference.enabled=false`) to process data.
- `dataset_inference`: optional override that enables batched inference over a dataset, with its own loader settings and chunking overrides.
- `input_wav` / `output_wav`: enables single-file reconstruction with optional trimming via `max_seconds` or `max_frames`.

The dataset block follows the same spec format used in training; the resolver `prepare_dataset_spec` makes paths absolute, normalizes STFT parameters, and optionally forces mono audio when requested.

### Running Inference
1. **Edit `conf/inference.yaml`:**
    - Point `checkpoint` to the trained weights.
    - Ensure `model_config_path` references the architecture that produced the checkpoint.
    - Configure the `dataset` block with the audio source and STFT settings used at train time (set `cac`, `stereo`, etc.).
    - Optionally enable `dataset_inference.enabled` to iterate over an entire dataset; otherwise only the single-file path is run.
2. **Launch the script:**
    ```bash
    uv run inference.py        # or: python inference.py
    ```
    Hydra picks up `conf/inference.yaml` as the default configuration. CLI overrides are possible (e.g. `python inference.py chunked=true chunk_size=64`).
3. **Inspect outputs:**
    - Single-file reconstructions land at `output_wav`.
    - Dataset runs write to `dataset_inference.output_dir` (optionally saving inputs if `save_input_audio=true`).

At startup the script instantiates the configured dataset, deduces spectrogram/audio channels, patches the model definition, loads the checkpoint, and runs the selected inference modes. Errors such as missing files or invalid chunk parameters are reported via the Rich-colored logger.

---

## 🔧 Training Configuration and Execution

Training is controlled by a hierarchical configuration composed from three groups: `data`, `model`, and `trainer`. Each group resides in the `conf/` directory.

```
conf/
    config.yaml          # composition defaults
    data/data.yaml       # datasets, dataloaders, demo parameters
    model/model.yaml     # encoder, decoder, bottleneck specification
    trainer/trainer.yaml # optimization, losses, logging, seed, device
```

### 1. Launching a Standard Training Run

Run with default settings:

```bash
python train.py
```

### 2. Overriding Configuration Parameters

Parameters can be overridden on the command line using dot notation. Examples:

Set number of epochs and batch size:
```bash
python train.py trainer.trainer.epochs=50 data.train_dataloader.batch_size=16
```

Adjust learning rate and weight decay:
```bash
python train.py trainer.optimizer.config.lr=3e-4 trainer.optimizer.config.weight_decay=5e-4
```

Change encoder ratios and disable W&B logging:
```bash
python train.py model.model.encoder.kwargs.ratios='[[2,2],[2,2],[2,2]]' trainer.wandb.use_wandb=false
```

Enable distributed data parallel (if multiple GPUs available) and set checkpoint directory:
```bash
python train.py trainer.strategy=ddp trainer.num_gpus=2 trainer.trainer.ckpt_dir=checkpoints/run_ddp
```

### 3. Dataset & Loader Controls

Key adjustable fields (group `data`):
- `train_dataset.kwargs.*`: STFT parameters (`n_fft`, `hop_length`, `win_length`, `target_frames`, `sample_rate`).
- `train_dataloader.batch_size`, `num_workers`, `prefetch_factor`: throughput tuning.
- `train_dataloader.pin_memory`: boolean or the string `auto` (auto-estimation logic in `train.py`).
- `eval_dataset` / `eval_dataloader`: analogous fields for validation.
- `demo.*`: governs validation reconstruction logging frequency and ISTFT parameters.

### 4. Model Adaptation

In `model/model.yaml`, use literal numeric values or the sentinel `auto` for `encoder.kwargs.input_size` and `decoder.kwargs.channels`. These are resolved at runtime from the first batch (spectrogram channel count). This permits reusing the same file across datasets with differing channel layouts (e.g. complex vs. real, CAC formats).

### 5. Optimization and Scheduling

Fields (group `trainer`):
- `optimizer.type`: optimizer identifier (e.g. `AdamW`).
- `optimizer.config.lr`, `weight_decay`, `betas`, etc. according to optimizer signature.
- `scheduler.type`: scheduler identifier, e.g. `InverseLR`.
- `scheduler.config.*`: scheduler hyperparameters (`inv_gamma`, `power`, `warmup`).

### 6. Loss Specification

Loss blocks are nested under `loss_config`. Example spectral MSE:
```yaml
loss_config:
    spectral:
        stft_mse:
            config:
                reduction: mean
        weights: { stft_mse: 1.0 }
```
Extend by adding new keys (e.g. `mrstft_sc`, `time`, `hubert`) following existing module expectations. Weights should be strictly positive for activation; zero omits a component.

### 7. Logging and Checkpointing

Controls:
- `wandb.use_wandb`: enable/disable experiment logging.
- `trainer.trainer.ckpt_dir`: output directory for checkpoints.
- `trainer.trainer.save_every_n_epochs`: checkpoint frequency.
- `trainer.trainer.log_interval`: logging step granularity.

### 8. Device, Precision, and Strategy

Relevant options:
- `trainer.num_gpus`: number of GPU devices to request.
- `trainer.strategy`: strategy identifier (`auto`, `ddp`, `deepspeed`, etc.).
- `device` (in `trainer.yaml`): target device string (`cuda` or `cpu`).

### 9. Hydra Composition Tips

Configs are grouped under `conf/` (e.g. `model/`, `data/`, `trainer/`). Select different variants by overriding the group on the CLI:
```bash
python train.py model=SEANet_cplx_model data=data trainer=trainer
```
Hydra merges the requested groups with `conf/config.yaml` defaults. Replace the group names with any other file in the corresponding directory and combine with per-parameter overrides from §2 to explore new experiments without editing files.

### 10. Reproducibility Guidelines

1. Fix `seed` for all comparative runs (`trainer.seed`).
2. Maintain consistent STFT parameters between training and evaluation to avoid reconstruction bias.
3. Record any external preprocessing transformations separate from configuration changes.
4. Use distinct checkpoint directories per hyperparameter sweep branch.
5. Validate channel auto-resolution once per dataset variant to prevent latent dimensional drift.

### 11. Minimal Repro Example

```bash
python train.py \
    trainer.trainer.epochs=5 \
    data.train_dataloader.batch_size=12 \
    trainer.optimizer.config.lr=3e-4 \
    trainer.wandb.use_wandb=false
```

### 12. Extending Configuration Space

To add a new encoder or dataset variant:
```bash
cp conf/model/model.yaml conf/model/encoder_alt.yaml
cp conf/data/data.yaml conf/data/music_highres.yaml
```
Invoke with:
```bash
python train.py model=encoder_alt data=music_highres
```
All unspecified groups fall back to their default entries defined in `config.yaml`.

---
