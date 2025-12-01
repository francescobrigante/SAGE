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

When adding new model architectures, you must ensure that your YAML configuration
exposes the parameters required by the trainer to inject channel information.
The training pipeline expects to find:

- `input_size` in the encoder arguments.
- `channels` in the decoder arguments.

These parameters are mandatory because the trainer uses them to adapt the model
to the dataset's specific channel layout (e.g. complex vs. real-as-channels,
mono vs. stereo).

Therefore, the model configuration must specify `channels` (decoder) and
`input_size` (encoder) explicitly, or rely on the `auto` placeholder.

- **Using `auto` (Recommended for Training)**: The trainer inspects the dataset
  batch (checking for complex dtype or CAC layout) and automatically injects
  the correct channel counts into the model configuration before instantiation.
  This ensures the model matches the data representation (e.g., complex vs.
  real-as-channels) and channel count (mono vs. stereo). 
- **Using Explicit Values**: If you manually set these integers in the YAML,
  you must ensure they strictly match the dataset output. Mismatches (e.g.,
  configuring 2 channels for a mono dataset) will cause runtime errors.
- **Example of `auto` working**: if the dataset has CAC activated and stereo there is going to be an `input_size` of 4. 

**Note on Checkpoints**: Once trained, the resolved values are baked into the
checkpoint's `inference_config`. The inference loader reads these saved values
automatically, so you do not need to worry about `auto` resolution when loading
a trained model.

If there are problems with checkpoint use in inference we suggest the following
procedure as a fallback.

### Regenerating legacy checkpoints

Older checkpoints that predate the embedded metadata can be upgraded with the
interactive helper in `tools/regenerate_checkpoint.py`. The script rebuilds
the autoencoder from a model YAML, prompts for any missing parameters, and
writes a new checkpoint suffixed with `_rigenerated`:

```bash
uv run python tools/regenerate_checkpoint.py \
  checkpoints/legacy.ckpt \
  conf/model/SEANet_cplx_model.yaml
```

Always use the regenerated artefact for inference and metric runs to guarantee
the presence of the `inference_config` block.

---

## Documentation

Detailed guidance on configuring data loaders, models, trainers, and inference pipelines lives in the `docs/` folder:
- `docs/model.md`
- `docs/trainer.md`
- `docs/training_dataset.md`
- `docs/metrics.md`

Refer to these documents to understand configuration fields, recommended overrides, and evaluation best practices.

---
