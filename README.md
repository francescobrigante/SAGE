# SAGE: Semantic Audio Generative Encoder

SAGE is a compact variational autoencoder for stereo music at 44.1 kHz. A Swin
Transformer V2 encoder and decoder operate on the STFT, and the latent is shaped by
distilling the embeddings of a pretrained audio-text model (LAION-CLAP). The released
model has 104.6M parameters and compresses a stereo waveform ×64 into a 16-channel
latent at 86 frames per second (one latent frame per 512 samples).

This repository contains the model, the two training phases, the evaluation code of
the paper and the released checkpoint's loader.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/). PyTorch 2.7.1 is installed with CUDA
12.6 wheels on Linux and the default wheels elsewhere.

```bash
git clone <repository url> sage && cd sage
uv sync                                   # inference only
uv sync --extra train                     # + training
uv sync --extra train --extra eval        # + evaluation (FAD, CLAP, CDPAM, MAEB)
uv sync --all-extras                      # + baselines of the paper (SAME, SAO VAE, CoDiCodec, Music2Latent)
```

`pip install .` (or `pip install ".[train,eval]"`) works too, without the lock file.

Optional, Linux with an NVIDIA GPU: fused CUDA kernels for the Swin window shift/partition
(NVIDIA, see Acknowledgements). They give the same outputs and speed up training and
inference, and are used automatically on GPU once importable. Building them needs the CUDA
toolkit of the PyTorch wheels (`nvcc`, `CUDA_HOME`) and gcc ≥ 9:

```bash
cd src/sage/model/swin/cuda_kernels
python setup.py build_ext --inplace            # with the environment active; writes swin_window_process*.so
export PYTHONPATH=$PWD:$PYTHONPATH              # make it importable (`uv sync` leaves it in place)
```

## Checkpoint

| File | Size | SHA-256 |
|---|---|---|
| `SAGE_FTe992.ckpt` (EMA weights, epoch 992) | 402 MiB | `dd87d01eaee88ca92f96c80ffe0e504a1d7cbc02e4271d8c48033df591d6ac99` |

Download link: *to be added.* Put the file in `models/` (the default location, see
[Paths](#paths)). It holds the model configuration and the EMA weights only; it was
exported from the training checkpoint with `scripts/export_checkpoint.py`.

## Usage

```python
import torch
from sage import SAGE

codec = SAGE.from_checkpoint("models/SAGE_FTe992.ckpt")          # CUDA if available
wav = torch.randn(1, 2, 10 * codec.sample_rate)                  # (B, 2, N) at 44.1 kHz

rec = codec.reconstruct(wav)                                     # encode + decode, same length as the input

padded, n = codec.pad(wav)                                       # right-pad to the model's frame grid
z = codec.encode(padded, deterministic=True)                     # posterior mean, (B, 16, frames)
y = codec.decode(z, target_length=padded.shape[-1])[..., :n]     # back to audio, input length
```

`reconstruct` pads, encodes and decodes exactly as the paper's evaluation does; the last
three lines do the same by hand. `encode` samples the latent unless `deterministic=True`
(the paper's reconstruction metrics use a sampled latent). From the command line, for any audio file (resampled to
44.1 kHz, mono duplicated to stereo):

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

Instead of environment variables, copy the file to `configs/paths/local.yaml` (ignored by
git), edit it and add `paths=local` to the commands below. Any single entry can also be
overridden on the command line, e.g. `paths.fma_audio=/data/fma_large`.

The weights under `SAGE_MODELS`: `SAGE_FTe992.ckpt`,
`LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt` (training teacher and probing oracle,
from [LAION-CLAP](https://github.com/LAION-AI/CLAP)) and `PANN/Cnn14_16k_mAP=0.438.pth`
(from [PANNs](https://zenodo.org/record/3987831)).

## Training

Two phases (paper Section 2.3, Table 6), each a Hydra recipe in `configs/experiment/`:

```bash
# phase 1: pretraining on FMA-full + MTG-Jamendo + M4Singer (500 epochs, 16 GPUs, global batch 128)
python train.py +experiment=pretrain trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4
# phase 2: decoder fine-tuning from the phase-1 checkpoint (encoder frozen)
python train.py +experiment=decoder_ft +init_from=<phase-1 checkpoint> trainer.trainer.num_gpus=4 ++trainer.trainer.num_nodes=4
```

The batch size is global, so the recipes run unchanged on fewer GPUs. Runs are written
to `runs/<name>/<date>/`; logging goes to Weights & Biases (`trainer.wandb.use_wandb=false`
for TensorBoard only). A SLURM requeue, or launching a run name again (the recipe's or
`trainer.wandb.name=`) after a crash, continues its newest checkpoint in its own folder and
its W&B run; `+ckpt_path=<file>` resumes a given checkpoint, `auto_resume=false` or a new
`trainer.wandb.name` starts over. A run that already has checkpoints refuses a new
`+init_from` instead of resuming over it. For SLURM clusters, `scripts/slurm/train.sbatch <recipe>`
runs a recipe with requeueing. On multi-node InfiniBand clusters where NCCL stops at the
first collective with `Could not find NET with id 0` (NCCL 2.26 of the PyTorch wheels, seen
with GPUDirect RDMA), `export NCCL_NET_GDR_LEVEL=LOC` before `sbatch` fixes it. Losses outside the paper are available for experiments in
`sage/nn/losses/experimental/` (see its `__init__`).

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
tests/                test suite
```

## Acknowledgements and third-party code

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

*To be added.*

## Citation

*To be added.*
