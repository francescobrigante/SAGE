#!/bin/bash
# ============================================================
# Day-N DN.2 — KL Sweep: kl=5e-5
# PLACEHOLDER: sostituire WINNER_CONFIG e WINNER_OVERRIDES
# ============================================================
#SBATCH --job-name=dn2_kl5e5
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/dn2_kl5e5
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

srun python train.py \
    models=WINNER_CONFIG \
    WINNER_OVERRIDES \
    trainer.loss_config.bottleneck.weights.kl=5e-5 \
    trainer.trainer.ckpt_dir=checkpoints/dn2_kl5e5 \
    trainer.wandb.name=dn2_winner_kl5e5
