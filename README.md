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

1. **Pick configurations** under `conf/` for the dataset, trainer, and model—you must reference the correct YAML files before launching training. Best practice is to keep the shipped YAML files as templates and apply overrides through Hydra CLI flags or copies stored under `conf/local/`.
2. **Start training** with the environment option you selected:
   - `uv run python train.py`
   - `python train.py`
3. **Override hyperparameters** directly from the command line when needed, for example:
   ```bash
   uv run python train.py trainer.trainer.epochs=50 data.train_dataloader.batch_size=16
   ```

---

## Inference Snapshot

Configure `conf/inference.yaml` (checkpoint, model, data) and run the matching command for your environment choice:
```bash
uv run python inference.py
# or
python inference.py
```

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
