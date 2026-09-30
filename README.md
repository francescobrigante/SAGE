<h1 align="center">SAGE: Semantic Audio Generative Encoder</h1>

<p align="center">
Francesco Brigante<sup>1</sup> · Luca Cerovaz<sup>1,3</sup> · Davide Marincione<sup>1</sup> ·
Giorgio Strano<sup>1</sup> · Luca Zhou<sup>1</sup> · Emanuele Rodolà<sup>1,3</sup> ·
Michele Mancusi<sup>1,2</sup><br>
<sup>1</sup>Sapienza University of Rome · <sup>2</sup>Moises Systems · <sup>3</sup>Paradigma
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.32755"><img src="https://img.shields.io/badge/arXiv-2609.32755-b31b1b.svg" alt="arXiv"></a>
  <a href="https://sage-music.pages.dev/"><img src="https://img.shields.io/badge/Project-page-6d4aff.svg" alt="Project page"></a>
  <a href="https://huggingface.co/francescobrigante/SAGE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-weights-ffcc4d.svg" alt="Weights on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="docs/assets/teaser.png" width="600" alt="FAD-MERT on the MoisesDB mixtures against real-time factor: SAGE has the lowest FAD-MERT at the real-time factor of Stable Audio Open">
  <br><em>Distributional fidelity against inference cost on the MoisesDB mixtures; marker area is the parameter count.</em>
</p>

SAGE is a compact variational autoencoder for stereo music at 44.1 kHz. It encodes
audio into a 16-channel latent representation and decodes it back to a waveform.
The released model has 104.6M parameters, with 64× compression and approximately
86 latent frames per second (one frame per 512 input samples).

Its Swin Transformer V2 encoder and decoder operate on the STFT. During training,
semantic distillation from LAION-CLAP, a pretrained audio-text model, shapes the
latent representation.

- **Fast:** it runs at the inference cost of Stable Audio Open.
- **High fidelity:** its listening-test score matches SAME-L, an autoencoder 8× larger and 4×
  slower, and it surpasses both on objective perceptual and distributional reconstruction metrics.
- **Semantic latent:** it sets the state of the art on all nineteen probing tasks of latent
  semantics, in domain and out of domain.
- **Open data:** it is trained solely on publicly available music.

This repository includes pretrained-model inference, both training phases, and the
paper's evaluation code.

- [Usage](#usage): reconstruct audio, or encode and decode latents.
- [Training guide](docs/training.md): data setup, training recipes, checkpoint resuming, and SLURM.
- [Paper runs](docs/paper_runs.md): commands and protocols for reproducing the reported results.
- [Changes from the development code](docs/changes.md): migration notes for pre-release users.

## Results

Listening test (MUSHRA, 21 raters after filtering; paper Table 3) and average probing score
per block of tasks (paper Table 4):

| Model | Params | RTF ↓ | MUSHRA ↑ | FMA (6) ↑ | MoisesDB (7) ↑ | Upstream MAEB (6) ↑ |
|---|---|---|---|---|---|---|
| **SAGE** | 105M | 0.0045 | 81.6 ± 2.7 | **0.563** | **0.544** | **0.622** |
| SAME-L | 852M | 0.0192 | **81.8 ± 2.6** | 0.470 | 0.493 | 0.473 |
| Stable Audio Open | 156M | 0.0045 | 64.6 ± 3.7 | 0.490 | 0.469 | 0.481 |
| CoDiCodec | 150M | 0.0237 | 66.4 ± 3.5 | 0.474 | 0.456 | 0.471 |

SAGE and SAME-L are statistically indistinguishable in the listening test. Reconstruction
metrics on the five evaluation sets, the per-task probing scores and the other baselines
(SAME-S, Music2Latent) are in the paper; [docs/paper_runs.md](docs/paper_runs.md) gives the
command that reproduces each of them.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/). PyTorch 2.7.1 is installed with CUDA
12.6 wheels on Linux and the default wheels elsewhere.

```bash
git clone https://github.com/francescobrigante/SAGE.git
cd SAGE
uv sync
```

Choose the extras for your workflow:

| Workflow | Install command |
|---|---|
| Inference | `uv sync` |
| Training | `uv sync --extra train` |
| Evaluation (FAD, CLAP, CDPAM, MAEB) | `uv sync --extra train --extra eval` |
| All extras, including paper baselines | `uv sync --all-extras` |

Run the commands below with `uv run`, or activate the environment first with
`source .venv/bin/activate`.

`pip install .` (or `pip install ".[train,eval]"`) also works, without the lock file.
Optional [fused CUDA kernels](docs/training.md#optional-fused-cuda-kernels) can speed
up training and inference on NVIDIA GPUs.

## Checkpoint

| File | Size | Weights |
|---|---|---|
| `SAGE_FTe992.ckpt` | 402 MiB | EMA weights from decoder fine-tuning, epoch 992 |

Download the checkpoint from
[Hugging Face](https://huggingface.co/francescobrigante/SAGE) and place it in
`models/` (the default location; see [Paths](#paths)):

```bash
hf download francescobrigante/SAGE SAGE_FTe992.ckpt --local-dir models
```

This inference checkpoint contains the model configuration and EMA weights,
exported with
`scripts/export_checkpoint.py`.

## Usage

```python
import torch
from sage import SAGE

codec = SAGE.from_checkpoint("models/SAGE_FTe992.ckpt")  # CUDA if available
wav = torch.randn(1, 2, 10 * codec.sample_rate)          # (B, 2, N) at 44.1 kHz

rec = codec.reconstruct(wav)  # encode + decode, same length as the input
```

To work with latents directly:

```python
padded, n = codec.pad(wav)                    # right-pad to the model's frame grid
z = codec.encode(padded, deterministic=True)  # posterior mean, (B, 16, frames)
y = codec.decode(z, target_length=padded.shape[-1])[..., :n]  # restore input length
```

By default, `reconstruct` and `encode` sample the latent, as in the paper's
reconstruction evaluation. Pass `deterministic=True` to use the posterior mean,
as in the manual example above.

The command-line interface resamples audio to 44.1 kHz and duplicates mono to stereo:

```bash
python scripts/encode_decode.py reconstruct song.wav song_rec.wav --ckpt models/SAGE_FTe992.ckpt
python scripts/encode_decode.py encode      song.wav song.pt      --ckpt models/SAGE_FTe992.ckpt
python scripts/encode_decode.py decode      song.pt  song_rec.wav --ckpt models/SAGE_FTe992.ckpt
```

## Paths

Every machine-specific location (datasets, weights, outputs) is set in one file,
[`configs/paths/default.yaml`](configs/paths/default.yaml). Each entry reads an environment
variable and falls back to a default:

| Variable | Used for |
|---|---|
| `SAGE_MODELS` (default `models/`) | the SAGE checkpoint, the LAION-CLAP teacher and the PANN weights of FAD-PANN |
| `SAGE_CHECKPOINT`, `CLAP_TEACHER_CKPT`, `PANN_CKPT`, `CLAP_FAD_CKPT`, `SAO_VAE_DIR` | single weight files, instead of their place under `SAGE_MODELS` |
| `FMA_AUDIO`, `FMA_METADATA` | FMA audio (`fma_large/`) and `fma_metadata/tracks.csv` |
| `FMA_FULL_AUDIO`, `JAMENDO_AUDIO`, `JAMENDO_SPLIT_TSV`, `M4SINGER_AUDIO` | the pretraining corpora |
| `MOISESDB_MIX`, `MOISESDB_STEMS`, `MUSICCAPS`, `SONG_DESCRIBER` | the 10 s clip sets of the reconstruction evaluation |
| `MOISESDB_ROOT`, `MOISESDB_CHUNKS_ROOT` | MoisesDB metadata and 30 s chunks of the probing tasks |
| `SAGE_EVAL_OUTPUT` (`results/`), `SAGE_EVAL_CACHE` (`eval_cache/`) | evaluation outputs and cache |

Instead of environment variables, copy the file to `configs/paths/local.yaml`
(ignored by git), edit it and add `paths=local` to training and evaluation commands.
Any single entry can also be overridden on the command line, e.g.
`paths.fma_audio=/data/fma_large`.

The weights under `SAGE_MODELS`: `SAGE_FTe992.ckpt`,
`LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt` (training teacher and probing oracle,
from [LAION-CLAP](https://github.com/LAION-AI/CLAP)) and `PANN/Cnn14_16k_mAP=0.438.pth`
(from [PANNs](https://zenodo.org/record/3987831)).

## Training

See the [training guide](docs/training.md) for setup, commands, checkpoint handling,
and cluster execution.

## Evaluation

```bash
python -m evaluation.build_references dataset=musiccaps            # once per 10 s clip set
python -m evaluation.reconstruction dataset=fma model=sage          # reconstruction metrics (Table 2)
python -m evaluation.reconstruction dataset=musiccaps model=same-s  # a baseline
python -m evaluation.maeb encoder=sage                              # 19 probing tasks (Tables 4, 9)
python -m evaluation.tables --recon "FMA test=results/recon_fma" --maeb results/maeb/SAGE_FTe992
```

Evaluation sets: `fma`, `moisesdb_mix`, `moisesdb_stems`, `musiccaps`, `song_describer`
(`configs/dataset/`). `build_references` writes the reference embeddings and statistics
into the set's own folder (`embeddings/`, `stats_ours/`), where `reconstruction` reads
them: it overwrites references already there, so use a copy of the folder (symlinks to the
audio suffice) to keep existing ones. [docs/paper_runs.md](docs/paper_runs.md) lists every result of the
paper with the command that reproduces it, how the random parts of the evaluation are
seeded, and what this repository does not reproduce. `scripts/slurm/eval.sbatch` runs any
of these commands on SLURM, sharding the reconstruction metrics over GPUs.

## Tests

```bash
uv sync --all-extras
uv run pytest                        # unit, smoke (tiny training, fine-tuning, evaluation) and CLI tests
uv run pytest -m weights             # + tests that download real pretrained models
```

## Repository layout

```
src/sage/
  inference.py        SAGE.from_checkpoint, encode / decode / reconstruct
  model/              the SAGE architecture: Swin V2 encoder and decoder, variable-length attention
  nn/                 building blocks: bottleneck, losses, discriminators, transformer layers
  training/           Lightning module, loss manager, data pipeline, callbacks
train.py              training entry point (Hydra)
configs/              Hydra configs: training recipes, evaluation, paths
evaluation/           reconstruction metrics, reference statistics, MAEB probing, tables, baselines
scripts/              encode_decode.py, export_checkpoint.py, SLURM templates
docs/                 training guide, paper reproduction commands, migration notes
tests/                test suite
```

## Acknowledgements and third-party code

We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned
by the EuroHPC Joint Undertaking, hosted by CINECA (Italy).

The codebase started from [EuleroDec](https://github.com/CerovazS/EuleroDec) by Luca Cerovaz
([@CerovazS](https://github.com/CerovazS)).

SAGE builds on [PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/),
[Hydra](https://hydra.cc/) and the [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer)
architecture; the mel loss and a discriminator layer come from
[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec). Parts of the code are
adapted from other projects, under their licenses, as noted in the file headers:

- fused Swin window kernels by NVIDIA (Apache 2.0 / MIT): `sage/model/swin/cuda_kernels/`
- [auraloss](https://github.com/csteinmetz1/auraloss) (Apache 2.0): spectral losses in `sage/nn/losses/signal.py`
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (MIT): discriminators, transformer layers
- [ESPnet](https://github.com/espnet/espnet) (Apache 2.0): `sage/nn/complex/layers.py`

Evaluation uses [fadtk](https://github.com/microsoft/fadtk), [LAION-CLAP](https://github.com/LAION-AI/CLAP),
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), [CDPAM](https://github.com/pranaymanocha/PerceptualAudio)
and [MTEB/MAEB](https://github.com/embeddings-benchmark/mteb), and the datasets FMA, MTG-Jamendo, M4Singer,
MoisesDB, MusicCaps and Song Describer.

## License

The code is released under the [MIT License](LICENSE). Files adapted from other projects keep
their original license, as noted in their headers (see above). The model weights are released
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), following the
non-commercial terms of part of the training data.

## Citation

```bibtex
@misc{brigante2026sage,
  title         = {{SAGE}: Semantic Audio Generative Encoder},
  author        = {Brigante, Francesco and Cerovaz, Luca and Marincione, Davide and Strano, Giorgio and
                   Zhou, Luca and Rodol{\`a}, Emanuele and Mancusi, Michele},
  year          = {2026},
  eprint        = {2609.32755},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2609.32755}
}
```
