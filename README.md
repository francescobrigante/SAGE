# 🎧 EulerAudioBackbone: Generative Backbone for Complex-Valued Audio Spectrograms

## Overview

EulerAudioBackbone (a.k.a. ComplexVAE backbone) targets **complex-valued** generative modeling directly over STFT spectrograms. It serves as the foundational autoencoder for:

- neural audio coding,
- diffusion-ready VAEs,
- source-separation research.

Unlike prior real-valued backbones, it maintains analyticity by propagating complex-valued representations end-to-end.

---

## Research Goals

- Release a modular, open-source complex-valued backbone for audio representation learning.
- Benchmark real vs. complex architectural choices (normalization, attention, activation families) on reconstruction quality.
- Provide reusable infrastructure that scales across domains (music, speech, ambient sound).

---

## Environment Setup

You can use the project in **two mutually exclusive ways**:

1. **Using Astral’s `uv` (recommended)** – project-style management via `pyproject.toml` and `uv.lock`.
2. **Using a classic Python virtual environment + `pip`** – dependency management via `requirements.txt`.

Choose **one** of the two; do **not mix** them.

---

### Option 1 – Using Astral `uv` (recommended)

In this workflow:

- Dependencies are declared in `pyproject.toml`.
- Exact versions are locked in `uv.lock`.
- The file `requirements.txt` is **not used** and can be ignored.

#### 1. Install `uv`

Follow the official instructions from Astral’s `uv` repository to install the CLI tool.

#### 2. Create and sync the environment

From the repository root:

~~~bash
# Create and populate a .venv using pyproject.toml + uv.lock
uv sync
~~~

This will:

- create `.venv/` (if not already present),
- install all dependencies declared in `pyproject.toml`,
- respect pinned versions from `uv.lock` (if present).

#### 3. Running commands with `uv`

You have two equivalent patterns:

1. **Without manually activating the environment** (recommended):

   - Training:
     ~~~bash
     uv run python train.py
     ~~~
   - Inference:
     ~~~bash
     uv run python inference.py
     ~~~
   - Any other script:
     ~~~bash
     uv run python path/to/script.py
     ~~~

2. **With manual activation of the `.venv`**:

   - Activate the environment:
     - Linux/macOS:
       ~~~bash
       source .venv/bin/activate
       ~~~
     - Windows (PowerShell):
       ~~~bash
       .venv\Scripts\Activate.ps1
       ~~~
   - Then run:
     ~~~bash
     python train.py
     python inference.py
     ~~~

In this `uv`-based setup:

- To add dependencies, edit `pyproject.toml` and then run `uv sync` again.
- You do **not** install from `requirements.txt`.

---

### Option 2 – Classic venv + pip + requirements.txt

In this workflow:

- Dependencies are managed through `requirements.txt`.
- `pyproject.toml` and `uv.lock` are **not required** for basic usage.

#### 1. Create a virtual environment

From the repository root:

~~~bash
python -m venv .venv
~~~

#### 2. Activate the virtual environment

- Linux/macOS:
  ~~~bash
  source .venv/bin/activate
  ~~~
- Windows (PowerShell):
  ~~~bash
  .venv\Scripts\Activate.ps1
  ~~~

#### 3. Install dependencies

~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

#### 4. Run scripts

With the environment active:

~~~bash
python train.py
python inference.py
~~~

If you are using only this classic setup, you can ignore `pyproject.toml` and `uv.lock`.

---

## Inference Quickstart

After choosing and setting up **one** of the environment options above:

1. **Configure** `conf/inference.yaml` with:
   - target checkpoint path,
   - model configuration,
   - dataset specification.

2. **Run batched inference** (auto-encodes an entire dataset):

   - Using `uv`:
     ~~~bash
     uv run python inference.py
     ~~~
   - Using classic venv:
     ~~~bash
     python inference.py
     ~~~

   Or orchestrate reconstruction + metric computation via the helper script:

   - Using `uv`:
     ~~~bash
     uv run ./test_metrics/compute_all.sh --checkpoint <ckpt> --output-dir <out> [--extensions wav,mp3]
     ~~~
   - Using classic venv:
     ~~~bash
     ./test_metrics/compute_all.sh --checkpoint <ckpt> --output-dir <out> [--extensions wav,mp3]
     ~~~

3. **Inspect outputs**

Reconstructed waveforms are written to the configured `output_dir`. If enabled, metric CSVs/logs will also be generated alongside the audio.

Additional details (configuration keys, metric options, expected directory structure) are documented in:

- `docs/inference_and_metrics.md`
- `docs/training_dataset.md`
- `docs/model.md`

---

## Training Quickstart

Once one of the two environment setups is ready:

1. **Select or customize a configuration**

   Edit the Hydra configuration files under `conf/` or override them from the CLI (e.g., batch size, number of epochs, optimizer).

2. **Launch training**

   - Using `uv`:
     ~~~bash
     uv run train.py
     ~~~
     Example with overrides:
     ~~~bash
     uv run train.py \
       trainer.trainer.epochs=50 \
       data.train_dataloader.batch_size=16
     ~~~

   - Using classic venv:
     ~~~bash
     python train.py
     ~~~
     Example with overrides:
     ~~~bash
     python train.py \
       trainer.trainer.epochs=50 \
       data.train_dataloader.batch_size=16
     ~~~

3. **Monitor training**

Logging, checkpointing, and metric tracking behavior are configurable via Hydra and described in:

- `docs/training_dataset.md`
- `docs/model.md`
- `docs/trainer.md`

---

## Extended Documentation

- Inference pipeline and metrics:
  - `docs/inference_and_metrics.md`
- Training dataset configuration:
  - `docs/training_dataset.md`
- Model specification and autoencoder assembly:
  - `docs/model.md`
- Training loop, optimization, and logging:
  - `docs/trainer.md`

These documents describe:

- dataset wiring and expected directory structures,
- model parametrization (encoder/decoder architecture, complex-valued layers, bottlenecks),
- optimization policies (schedulers, gradient clipping, mixed precision),
- reproducibility settings (seeds, deterministic flags, logging backends).

---
