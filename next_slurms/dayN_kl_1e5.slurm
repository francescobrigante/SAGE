#!/bin/bash
# ============================================================
# Day-N DN.1 — KL Sweep: kl=1e-5 (baseline KL, già misurato se winner=run1)
#
# PLACEHOLDER: sostituire WINNER_CONFIG e WINNER_OVERRIDES
# con la config del vincitore Day-1.
# ============================================================
#SBATCH --job-name=dn1_kl1e5
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

mkdir -p logs checkpoints/dn1_kl1e5
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

# TODO: Copiare la srun del winner Day-1 e cambiare solo kl e ckpt_dir/wandb.name
srun python train.py \
    models=WINNER_CONFIG \
    WINNER_OVERRIDES \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.trainer.ckpt_dir=checkpoints/dn1_kl1e5 \
    trainer.wandb.name=dn1_winner_kl1e5
