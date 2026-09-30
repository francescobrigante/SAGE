# Training SAGE

This guide covers data setup, the two training phases, checkpoint handling, and
SLURM execution. Run all commands from the repository root.

The paper recipes live in [configs/experiment/](../configs/experiment/) and
correspond to Section 2.3 and Table 6. For evaluation after training, see
[Paper runs](paper_runs.md).

## Environment

Use Python 3.11 and install the training dependencies:

```bash
uv sync --extra train
source .venv/bin/activate
```

The commands below use this active environment. Alternatively, prefix Python
commands with `uv run`.

## Data and teacher weights

Both phases use FMA-full, MTG-Jamendo, and M4Singer for training. Validation uses
the FMA-large validation split. Paths are defined in
[configs/paths/default.yaml](../configs/paths/default.yaml).

| Environment variable | Required content |
|---|---|
| `FMA_FULL_AUDIO` | FMA-full audio root for training |
| `FMA_AUDIO` | FMA-large audio root for validation |
| `FMA_METADATA` | FMA `tracks.csv` metadata file, including split information |
| `JAMENDO_AUDIO` | MTG-Jamendo audio root, with the `XX/ID.mp3` layout |
| `JAMENDO_SPLIT_TSV` | MTG-Jamendo split-0 `autotagging-train.tsv` |
| `M4SINGER_AUDIO` | M4Singer audio root containing the `.wav` files |
| `CLAP_TEACHER_CKPT` | LAION-CLAP teacher checkpoint, required for pretraining |
| `SAGE_FILELIST_CACHE` | Optional directory for scanned file lists; defaults to `filelist_cache/` |

For example, replace these paths with your local locations:

```bash
export FMA_FULL_AUDIO=/data/fma_full
export FMA_AUDIO=/data/fma_large
export FMA_METADATA=/data/fma_metadata/tracks.csv
export JAMENDO_AUDIO=/data/mtg_jamendo
export JAMENDO_SPLIT_TSV=/data/mtg_jamendo/splits/split-0/autotagging-train.tsv
export M4SINGER_AUDIO=/data/m4singer
export CLAP_TEACHER_CKPT=/weights/LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt
```

Obtain `music_audioset_epoch_15_esc_90.14.pt` from
[LAION-CLAP](https://github.com/LAION-AI/CLAP). If `CLAP_TEACHER_CKPT` is unset,
the default location is
`models/LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt`, under `SAGE_MODELS` if set.
Decoder fine-tuning does not load the CLAP teacher.

Instead of exporting variables, copy `configs/paths/default.yaml` to
`configs/paths/local.yaml`, edit the paths, and add `paths=local` to each training
command. The local file is ignored by Git. Individual paths can also be overridden,
for example `paths.fma_audio=/data/fma_large`.

## Phase 1: pretraining

The [pretraining recipe](../configs/experiment/pretrain.yaml) trains the encoder
and decoder from scratch for 500 epochs. It combines reconstruction, KL,
semantic distillation, and adversarial losses.

For a single node with four GPUs:

```bash
python train.py +experiment=pretrain trainer.trainer.num_gpus=4
```

The recipe's default run name is `sage_pretrain`. Keep the full training checkpoint
from this phase for decoder fine-tuning.

## Phase 2: decoder fine-tuning

The [fine-tuning recipe](../configs/experiment/decoder_ft.yaml) freezes the encoder,
loads the phase-1 EMA weights, and trains the decoder for 992 epochs. It adds a
zero-initialized decoder post-net, disables KL and semantic losses, and starts a
new discriminator and optimizer state.

Set the path to your phase-1 training checkpoint:

```bash
PHASE1_CKPT=/absolute/path/to/phase1/checkpoints/last.ckpt
python train.py +experiment=decoder_ft +init_from="$PHASE1_CKPT" \
    trainer.trainer.num_gpus=4
```

The default run name is `sage_decoder_ft`. `+init_from` initializes model weights
for a new phase; it does not restore optimizer state or the training step.
Use a full training checkpoint here, rather than the exported inference checkpoint.

## GPU count and batch size

The paper used four nodes with four A100 GPUs each. For multi-node execution, use
the SLURM template below, which sets the GPU and node counts from the allocation.

Both recipes set a **global batch size of 128**, divided across all GPUs:

| GPUs in total | Batch per GPU |
|---|---|
| 16 | 8 |
| 4 | 32 |
| 1 | 128 |

The default is one GPU when `trainer.trainer.num_gpus` is omitted. Fewer GPUs
increase the memory required per device. If the batch does not fit, reduce both
`data.train_dataloader.batch_size` and `data.eval_dataloader.batch_size`; this
changes the paper recipe's batch size. Choose a global batch divisible by the
total number of GPUs to avoid rounding.

## Outputs and logging

Runs are stored under `runs/<name>/<date>/`, relative to the launch directory.

| Location | Content |
|---|---|
| `checkpoints/` | Periodic epoch checkpoints and `last.ckpt` |
| `model_info.json` | Model structure and parameter counts |
| `lightning_logs/` | TensorBoard logs when TensorBoard is selected |
| `.run_ids/<name>` (repository root) | Persisted Weights & Biases run ID |

Pretraining saves periodic checkpoints every 20 epochs; fine-tuning every 25.
Weights & Biases is enabled by default, with project name `sage`. Add
`trainer.wandb.use_wandb=false` to use TensorBoard instead.

Use a distinct `trainer.wandb.name` for each new experiment. This setting also
controls the output folder and automatic resuming when W&B logging is disabled.

## Resume or start a new run

| Intent | Setting |
|---|---|
| Continue the latest checkpoint | Relaunch the same recipe and run name; omit `+init_from` |
| Resume a specific training checkpoint | Add `+ckpt_path=/absolute/path/to/checkpoint.ckpt` |
| Start an independent experiment | Set a new `trainer.wandb.name` |
| Disable automatic resuming for a fresh launch | Add `auto_resume=false` |

Automatic resuming is enabled by default for recipe names and explicitly chosen
names. It selects the newest `last.ckpt` or SLURM requeue checkpoint for that name.
A checkpoint belonging to the same run continues in its existing folder and,
when available, uses the saved W&B run ID.

For example, to continue decoder fine-tuning after an interruption:

```bash
python train.py +experiment=decoder_ft trainer.trainer.num_gpus=4
```

If that name already has a checkpoint, launching with a new `+init_from` stops
instead of silently resuming. Remove `+init_from` to continue, or choose a new
name to start another fine-tuning run. An explicit `+ckpt_path` takes precedence
over `+init_from`. SLURM requeues always resume their own run, even when
`auto_resume=false`.

## SLURM

[scripts/slurm/train.sbatch](../scripts/slurm/train.sbatch) defaults to four
nodes, four GPUs and four tasks per node, eight CPUs per task, and a four-day
time limit. Adapt the resources to your cluster and select its account and
partition. Activate the environment and export the data paths before submission;
the job inherits them.

The template writes stdout to `logs/`, so create that directory before submitting.
Validate the script and the intended allocation on the cluster's login node:

```bash
mkdir -p logs
bash -n scripts/slurm/train.sbatch
sbatch --test-only -A <account> -p <partition> scripts/slurm/train.sbatch pretrain
```

Check the cluster's live resource and time limits as well. Once validation passes,
submit the same command without `--test-only`:

```bash
sbatch -A <account> -p <partition> scripts/slurm/train.sbatch pretrain
```

For the first fine-tuning launch, pass `decoder_ft +init_from="$PHASE1_CKPT"`
instead of `pretrain`, and validate that exact command before submission.
Arguments after the recipe are Hydra overrides. `RESUME_CKPT=/path/to/file.ckpt`
selects an explicit resume checkpoint for the template.

Lightning saves a checkpoint and requeues the job on the time-limit signal.
The relaunched job continues its checkpoint and W&B run.

### InfiniBand troubleshooting

If NCCL fails at the first collective with `Could not find NET with id 0` on a
multi-node InfiniBand cluster, try the workaround used with the PyTorch wheels'
NCCL 2.26 and GPUDirect RDMA. Export it before validation and submission:

```bash
export NCCL_NET_GDR_LEVEL=LOC
```

## Optional fused CUDA kernels

The NVIDIA Swin window shift/partition kernels can speed up training and inference.
They are used automatically on GPU once importable. Building them requires Linux,
an NVIDIA GPU, gcc ≥ 9, and a CUDA toolkit matching the PyTorch wheels (`nvcc` and
`CUDA_HOME`). With the environment active:

```bash
cd src/sage/model/swin/cuda_kernels
python setup.py build_ext --inplace
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd ../../../../..
```

The build creates `swin_window_process*.so` in the kernel directory. Export the
same `PYTHONPATH` in each new shell where you want to use the kernels.

## Experimental losses

Losses outside the paper are in
[src/sage/nn/losses/experimental/](../src/sage/nn/losses/experimental/). See that
package's `__init__.py` for supported classes and configure them through
`trainer.loss_config.extra`.
