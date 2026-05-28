#!/bin/bash
# ============================================================
# eval_swin_debug.slurm
# Smoke evaluation on 100 files — VarT Swin (single forward pass).
# Verifies SI-SDR, STFT, CLAP, FAD pipeline end-to-end.
# CDPAM skipped for speed; enable by removing --skip-cdpam.
# ============================================================
#SBATCH --job-name=eval_swin_debug
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-00:30:00
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

CACHE_DIR="$SCRATCH/eval_cache"
mkdir -p "$CACHE_DIR"

# rank 0 → GPU 0 → baseline;  rank 1 → GPU 1 → STFT_3seconds
srun python evaluation/evaluate_swin_varT.py \
    --checkpoint checkpoints/baseline.ckpt checkpoints/STFT_3seconds.ckpt \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_debug \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 4 \
    --max-files 100 \
    --skip-cdpam
