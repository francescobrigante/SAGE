#!/bin/bash
# ============================================================
# eval_swin_real.slurm
# Evaluation of the real-valued Swin baseline (real.ckpt).
# Single checkpoint → single GPU. VarT forward pass (no chunking).
# Output: runs/eval_swin_real/real/metrics/
# ============================================================
#SBATCH --job-name=eval_swin_real
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=04:00:00
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

srun python evaluation/evaluate_swin_varT.py \
    --checkpoint $HOME/work/C-VAE/checkpoints/real.ckpt \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_real \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 16 \
    --max-files 2500 \
    --cdpam-only \
    --resume
