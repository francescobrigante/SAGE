#!/bin/bash
# ============================================================
# Swin Real VAE - 4-stage, x64, 80 epochs, 1 GPU (A100)
#
# Hardware: 1 Nodo, 1 GPU
# Dati: 6400 campioni -> ~400 batch per epoca (su singola GPU).
# Steps totali = 80 epoche * 400 batch = 32000 step.
# ============================================================

#SBATCH --job-name=train_real
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # Solo 1 GPU usata
#SBATCH --gres=gpu:1                   # Alloca 1x A100 (64GB)
#SBATCH --cpus-per-task=8              # 8 CPU associate alla GPU per I/O
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-22:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/real_x64

module load profile/deeplrn cineca-ai/4.3.0

# OMP=1: each DataLoader worker runs single-threaded STFT. Without this,
# num_workers=8 workers × OMP=8 = 64 threads on 8 CPUs → context-switch thrashing.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

echo "============================================================"
echo "Starting Real-valued model training on 1 GPU"
echo "============================================================"

# Calcolo Scheduler: 80 epoche * 400 batch (su 1 GPU) = 32000 total_steps
# Warmup: 5% di 32000 = 1600 step

srun python train.py \
    models=swin_real_4s_x64 \
    trainer.trainer.epochs=80 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.scheduler.total_steps=32000 \
    trainer.scheduler.warmup_steps=1600 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.3 \
    trainer.trainer.ckpt_dir=checkpoints/real_x64 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=auto \
    trainer.wandb.use_wandb=true \
    trainer.wandb.name=real64x  # <--- CAMBIA QUI IL NOME VISIBILE SU W&B
