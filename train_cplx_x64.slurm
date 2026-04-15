#!/bin/bash
# ============================================================
# Swin Complex VAE - 4-stage, x64, 50 epochs, 1 GPU (A100)
#
# Hardware: 1 Nodo, 1 GPU
# Dati: 6400 campioni -> ~400 batch per epoca (su singola GPU).
# Steps totali = 50 epoche * 400 batch = 20000 step.
# ============================================================

#SBATCH --job-name=train_cplx
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # Solo 1 GPU usata
#SBATCH --gres=gpu:1                   # Alloca 1x A100 (64GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/cplx_x64

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

echo "============================================================"
echo "Starting COMPLEX-Valued training on 1 GPU"
echo "============================================================"

# Calcolo Scheduler: 50 epoche * 400 batch (su 1 GPU) = 20000 total_steps
# Warmup: 5% di 20000 = 1000 step

srun python train.py \
    models=swin_cplx_4s_x64 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    trainer.trainer.epochs=50 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.scheduler.total_steps=20000 \
    trainer.scheduler.warmup_steps=1000 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.3 \
    trainer.trainer.ckpt_dir=checkpoints/cplx_x64 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=auto \
    trainer.wandb.use_wandb=true \
    trainer.wandb.name=TEST_cplx_x64  # <--- CAMBIA QUI IL NOME VISIBILE SU W&B
