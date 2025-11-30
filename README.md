# 🎧 EulerAudioBackbone: Generative Backbone for Complex-Valued Audio Spectrograms

## Overview
EulerAudioBackbone (a.k.a. ComplexVAE backbone) targets **complex-valued** generative modeling directly over STFT spectrograms. It serves as the foundational autoencoder for neural audio coding, diffusion-ready VAEs, and source-separation research. Unlike prior real-valued backbones, it maintains analyticity by propagating complex-valued representations end-to-end.

## Research Goals
- Release a modular, open-source complex-valued backbone for audio representation learning.
- Benchmark real vs. complex architectural choices (normalization, attention, activation families) on reconstruction quality.
- Provide reusable infrastructure that scales across domains (music, speech, ambient sound).

## Inference Quickstart
1. **Configure** `conf/inference.yaml` with the target checkpoint, model config, and dataset specification.
2. **Run batched inference** (auto-encodes an entire dataset) with:
   ```bash
   python inference.py
   ```
   or orchestrate reconstruction plus metric computation via:
   ```bash
   ./test_metrics/compute_all.sh --checkpoint <ckpt> --output-dir <out> [--extensions wav,mp3]
   ```
3. **Inspect outputs:** reconstructed waveforms will be written to the configured `output_dir` and optional metric CSVs/logs.

Essential configuration keys and advanced usage details are documented in `docs/dataset.md`, `docs/model.md`, and `docs/metrics.md`.

## Training Quickstart
1. **Install dependencies** and activate the project virtual environment.
2. **Select a configuration** by editing the Hydra files under `conf/` (or override settings on the CLI).
3. **Launch training:**
   ```bash
   python train.py
   ```
   Example override:
   ```bash
   python train.py trainer.trainer.epochs=50 data.train_dataloader.batch_size=16
   ```

Guidance on dataset wiring, model parametrization, optimization policies, and reproducibility lives in `docs/dataset.md`, `docs/model.md`, and `docs/trainer.md`.

## Extended Documentation
- Dataset configuration and preprocessing: `docs/dataset.md`
- Model specification and autoencoder assembly: `docs/model.md`
- Training loop, optimization, and logging: `docs/trainer.md`
- Offline evaluation metrics (STFT, CDPAM, FAD): `docs/metrics.md`

## Selected Related Work
- Stable Audio Open — latent diffusion on stereo waveforms.
- Music2Latent2 — autoregressive decoding over learned latents.
- CoDiCodec — consistency modeling with discrete/continuous latents.

EulerAudioBackbone extends these systems into the complex domain to support phase-aware generation and downstream diffusion models.
