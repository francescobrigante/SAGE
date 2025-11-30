# 🎧 EulerAudioBackbone

EulerAudioBackbone is a complex-valued autoencoder designed for generative modeling directly on STFT spectrograms. It supports research on neural audio coding, diffusion-ready VAEs, and source separation while preserving analyticity throughout the pipeline.

---

## Environment Setup

Choose a single workflow and stick to it—either Astral `uv` or a classic virtual environment managed by `pip`.

### Option A — Astral `uv` (recommended)
- Install the `uv` CLI following the official instructions.
- From the repository root run:
  ```bash
  uv sync
  ```
  This creates `.venv/`, installs dependencies from `pyproject.toml`, and honors `uv.lock`.
- Execute scripts without manual activation:
  ```bash
  uv run train.py
  uv run inference.py
  ```
- To add packages, edit `pyproject.toml` and re-run `uv sync` (ignore `requirements.txt`) or follow uv ufficial documentation with `uv.add ...` .

### Option B — Classic virtualenv + pip
- Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # use .venv\Scripts\Activate.ps1 on Windows
  ```
- Install dependencies from `requirements.txt`:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Run scripts via the active environment:
  ```bash
  python train.py
  python inference.py
  ```

---

## Training Workflow

1. **Pick configurations** under `conf/` for the dataset, trainer, and model—you must reference the correct YAML files before launching training. Best practice is to keep the shipped YAML files as templates and apply overrides through Hydra CLI flags or copies stored under `conf/config.yaml`.
2. **Start training** with the environment option you selected:
   - `uv run train.py`
   - `python train.py`
3. **Override hyperparameters** directly from the command line when needed, for example:
   ```bash
   uv run python train.py trainer.trainer.epochs=50 data.train_dataloader.batch_size=16
   ```

---

## Inference Snapshot

You no longer need to wire Hydra configs to reconstruct the model at inference
time.  Checkpoints produced by the updated training loop embed all the metadata
needed by `ar_spectra.models.eulero_inference.EuleroEncodeDecode`.

```python
from ar_spectra.models.eulero_inference import EuleroEncodeDecode

codec = EuleroEncodeDecode("checkpoints/my_model.ckpt")
latents, info = codec.encode_audio(waveform_batch)
recons = codec.decode_audio(latents, info)
```

Under the hood the helper rebuilds the `AutoEncoder` from the serialized specs,
restores the canonical STFT configuration, and exposes encode/decode methods
that operate directly on waveforms.  This same utility is what `inference.py`
invokes, so CLI entry points still work via:

```bash
uv run inference.py
# or
python inference.py
```

> **Note**: regenerate checkpoints after pulling this change so they include
> the `inference_config` block required by the helper.

### Building Models from Configs

When constructing autoencoders manually—from YAML or bespoke scripts—you must
resolve the expected channel counts before calling `AutoEncoder.from_config`.
The constructors assume that:

- `model_channels` (spectrogram channels) and `audio_channels` (wave channels)
  are known integers, not the `auto` sentinel found in templates like
  `conf/model/SEANet_cplx_model.yaml`.
- Pre-transform and STFT parameters are supplied consistently with the dataset
  used during training.

The training initialisation helpers already handle this via
`resolve_auto_channels`.  If you roll your own loader, mimic the following
pattern:

```python
from copy import deepcopy
from ar_spectra.training_utils.initialization import resolve_auto_channels
from ar_spectra.models.autoencoder import AutoEncoder

model_cfg = deepcopy(cfg["model"])
resolve_auto_channels(model_cfg, model_channels=<detected_value>)
autoencoder = AutoEncoder.from_config(model_cfg)
```

Skipping this step will leave placeholders in place and most encoder/decoder
classes will raise shape mismatches during instantiation.

---

## Documentation

Detailed guidance on configuring data loaders, models, trainers, and inference pipelines lives in the `docs/` folder:
- `docs/model.md`
- `docs/dataset.md`
- `docs/inference_and_metrics.md`
- `docs/trainer.md`
- `docs/training_dataset.md`

Refer to these documents to understand configuration fields, recommended overrides, and evaluation best practices.

---
