#!/bin/bash
# ============================================================
# K4_ps64x1_464_e256: Suite K (Deep Stages) — PS=(64,1), Win=(4,32), 4 stage, embed=256
# Depths: encoder=[4,6,4], decoder=[4,6,4]
#
# GRID (PatchEmbed → PatchMerge ×2):
#   Stage 0:  16×128  (1024/64  ×  128/1)
#   Stage 1:   8×64
#   Stage 2:   4×32   (bottleneck) — 128 token
#
# GLOBAL ATTENTION — window=[4,32]:
#   Stage 0: win [4,32]  — 4×4=16 windows
#   Stage 1: win [4,32]  — 2×2=4 windows
#   Stage 2: win [4,32] == grid [4,32] → global ✓
#
# RATIO & PARAMS:
#   embed=256: ratio 2× patch input (64×1×2=128 complex) — canonical Swin-Tiny ratio
#   Params: ~300M (scaled from ~194M)
#   Latent: 128×8 (seq=128, channels=8 complex) — Ottimo per LDM, KL atteso ~71
#
# LOSS: MRSTFT(1.0)
# DATA: FMA-medium (19,922 train tracks)
# ============================================================
#SBATCH --job-name=K4_ps64x1_464
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[64,1]" \
    "models.model.swin.window_size=[4,32]" \
    "models.model.swin.embed_dim=256" \
    "models.model.encoder.depths=[4,6,4]" \
    "models.model.encoder.num_heads=[8,16,32]" \
    "models.model.decoder.depths=[4,6,4]" \
    "models.model.decoder.num_heads=[32,16,8]" \
    models.model.latent_channels=8 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=97 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.optimizer.lr=1e-3 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=false" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/K4_ps64x1_464 \
    trainer.wandb.name=K4_ps64x1_464
