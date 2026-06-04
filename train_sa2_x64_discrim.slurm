#!/bin/bash
# ============================================================
# SA2-VAE Oobleck (real) - channels=128, x64, with Encodec discriminator
#
# Hardware: 1 Nodo, 4 GPU DDP
# Dati: FMA-large
# Config: fma_vae_discrim.json (adv=0.1, feat_match=5.0, kl=1e-4)
# ============================================================

#SBATCH --job-name=train_SAO_original_Disc
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                   # 4x A100 64GB
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/sa2_x64_160M_disc

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- Wandb online via squid proxy (login07:3150) ----
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache

# ---- Load .env (DATA_PATH, FMA_METADATA, WANDB_API_KEY) ----
set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE/stable_audio_baseline
source $WORK/.envs/sa2/bin/activate

srun python train_fma.py \
    --model-config configs/model/fma_vae_discrim.json \
    --batch-size 4 \
    --num-workers 8 \
    --save-dir checkpoints/sa2_x64_160M_disc \
    --name SAO160M_Disc \
    --num-gpus 4 \
    --precision bf16-mixed \
    --checkpoint-every 50000 \
    --val-every 2500 \
    --subset large \
    --max-steps 520000 \
    --ckpt-path $WORK/C-VAE/runs/SAO160M_Disc/2026-05-24_21-55-56/checkpoints/sa2-step=400000.ckpt \
    --wandb-id tks7fffq \
    --run-dir $WORK/C-VAE/runs/SAO160M_Disc/2026-05-24_21-55-56
