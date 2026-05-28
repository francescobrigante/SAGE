#!/bin/bash
# ============================================================
# eval_swin_varT_large: Full FMA-large evaluation — VarT Swin
#
# Evaluates baseline + STFT_3seconds in parallel (1 per GPU).
# Single forward pass per file (VarT padding, no chunking).
# Metrics: SI-SDR, STFT-loss, CDPAM.
# Use --resume to restart safely after preemption.
# ============================================================
#SBATCH --job-name=eval_swin_varT_large
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# ---- checkpoints (rank 0 → baseline, rank 1 → STFT_3seconds) ----
CKPT0=$WORK/C-VAE/checkpoints/baseline.ckpt
CKPT1=$WORK/C-VAE/checkpoints/STFT_3seconds.ckpt

srun python evaluation/evaluate_swin_varT.py \
    --checkpoint "$CKPT0" "$CKPT1" \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_varT_large \
    --num-workers 8 \
    --resume
