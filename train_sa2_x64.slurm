#!/bin/bash
# ============================================================
# SA2-VAE Oobleck (real) - channels=72, x64, 100k steps, 4 GPU (A100)
#
# Hardware: 1 Nodo, 4 GPU DDP
# Dati: FMA-medium
# Steps totali = 100k
# Iso-parameter vs Swin cplx x64: 49.85M scalar params.
# ============================================================

#SBATCH --job-name=train_SAO_original_noDisc
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                   # 4x A100 64GB
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/sa2_x64_160M

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
mkdir -p $WORK/C-VAE/.wandb_cache

# ---- Load .env (DATA_PATH, FMA_METADATA, WANDB_API_KEY) ----
set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE/stable_audio_baseline
source $WORK/.envs/sa2/bin/activate

echo "============================================================"
echo "Starting SA2-VAE (Oobleck real) on 4x A100 DDP"
echo "Model  : channels=128, 160M params, x64 compression"
echo "Steps  : 500000  (FMA-large, batch=4/GPU x 4 GPU = 16 global)"
echo "============================================================"

srun python train_fma.py \
    --model-config configs/model/fma_vae_nodiscrim.json \
    --batch-size 4 \
    --num-workers 8 \
    --save-dir checkpoints/sa2_x64_160M \
    --name SAO160M_noDisc \
    --num-gpus 4 \
    --precision bf16-mixed \
    --checkpoint-every 50000 \
    --val-every 2500 \
    --subset large \