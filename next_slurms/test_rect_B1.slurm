#!/bin/bash
# ============================================================
# Phase 8 — Rect B1: PS=(16,2), Win=8, 4 stage, latent 64×16
# Grid: 64×64 → 32×32 → 16×16 → 8×8. Griglia quadrata a tutti
# gli stage: win=8 scalare ottimale, global attention a 8×8.
# FC-ratio≈21 ✔. FMA-medium, 4xA100, 100k steps, loss=MRSTFT
# ============================================================
#SBATCH --job-name=p8_B1
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/test_rect_B1
module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[16,2]" \
    models.model.swin.window_size=8 \
    models.model.latent_channels=16 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=97 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    trainer.scheduler.total_steps=120000 \
    trainer.scheduler.warmup_steps=6000 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/test_rect_B1 \
    trainer.wandb.name=test_rect_B1
