#!/bin/bash
# ============================================================
# eval_swin_large.slurm
# Full FMA-large evaluation — VarT Swin (single forward pass).
# Metrics: SI-SDR, STFT, CLAP cosine, FAD (MERT + CLAP-audio).
# CDPAM skipped at scale (too slow); run debug job for it.
# Use --resume to restart safely after preemption.
# ============================================================
#SBATCH --job-name=eval_swin_n11
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=02:00:00
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

# Checkpoints to evaluate (default: N7 + N8). Each rank/GPU handles one checkpoint
# (evaluate_swin_varT.py splits --checkpoint round-robin across ranks). Pass extra
# checkpoint paths as CLI args to override the default list.
CKPTS=("$@")
if [ ${#CKPTS[@]} -eq 0 ]; then
    CKPTS=(
        checkpoints/N11_real_mse_bigvgan_fm05.ckpt
    )
fi
echo "Evaluating checkpoints: ${CKPTS[*]}"

srun python evaluation/evaluate_swin_varT.py \
    --checkpoint "${CKPTS[@]}" \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_large \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 16 \
    --max-files 2500 \
    --resume
