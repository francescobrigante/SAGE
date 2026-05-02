#!/bin/bash
# ============================================================
# SA2-VAE Oobleck (real) - channels=72, x64, 80-epoch equiv, 2 GPU (A100)
#
# Hardware: 1 Nodo, 2 GPU DDP
# Dati: 6400 campioni -> 200 step/epoca per GPU (batch=16 x 2 GPU = 32 global)
# Steps totali = 80 epoche * 200 step = 16000 step
# Iso-parameter vs Swin cplx x64: 49.85M scalar params.
# ============================================================

#SBATCH --job-name=train_sa2
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 64GB
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-12:00:00              # SA2 training piu' lento di Swin (1D conv su waveform)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/sa2_x64

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- Wandb online via squid proxy (login07:3150) ----
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export WANDB_TAGS="$SLURM_JOB_NAME"

# ---- Load .env (DATA_PATH, FMA_METADATA, WANDB_API_KEY) ----
set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE/stable_audio_baseline
source $WORK/.envs/sa2/bin/activate

echo "============================================================"
echo "Starting SA2-VAE (Oobleck real) on 2x A100 DDP"
echo "Model  : channels=72, 49.85M params, x64 compression"
echo "Steps  : 16000  (equiv. 80 epochs @ batch=16 x 2 GPU)"
echo "============================================================"

# Scheduler note:
#   SAO InverseLR: inv_gamma=200000, power=0.5, warmup=0.999
#   Warmup si completa ~step 5000; lr decade come (1 + step/200000)^-0.5
#   Simile a Swin (warmup 1600 step + InverseLR), solo piu' lunga la rampa.

srun python train_fma.py \
    --model-config configs/model/fma_vae_nodiscrim.json \
    --batch-size 16 \
    --num-workers 8 \
    --save-dir checkpoints/sa2_x64 \
    --name sa2_oobleck_x64 \
    --num-gpus 2 \
    --precision bf16-mixed \
    --max-steps 16000 \
    --checkpoint-every 2000 \
    --val-every 200 \
    --subset small
