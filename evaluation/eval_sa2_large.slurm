#!/bin/bash
# ============================================================
# eval_sa2_large.slurm
# Evaluate the sa2 500k checkpoint on FMA-large test split
# (11,263 tracks). One node, 1 A100.
#
# Architecture:
#   - single checkpoint (step=500000) on GPU 0
#
# Target embeddings (CLAP/MERT) cached to $SCRATCH/eval_cache/
# and reused across checkpoints and future runs.
#
# CDPAM skipped (too slow at 11k-file scale — run debug job for it).
# Use --resume to restart after preemption without recomputing.
# ============================================================
#SBATCH --job-name=eval_sa2_large
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_sa2_large_%j.out
#SBATCH --error=logs/eval_sa2_large_%j.err

set -euo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/sa2/bin/activate

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1   # plain-text logs — ANSI codes make SLURM output unreadable
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_large"
EVAL_ROOT="$WORK/C-VAE/runs/eval_sa2_large"
CACHE_DIR="$SCRATCH/eval_cache"

mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

SA2_CKPTS=(
    # "checkpoints/sa2Disc-step=100000.ckpt"
    # "checkpoints/sa2Disc-step=250000.ckpt"
    "checkpoints/sa2Disc-step=500000.ckpt"
)

COMMON_FLAGS=(
    --target-dir   "$TARGET_DIR"
    --cache-dir    "$CACHE_DIR"
    --num-workers  16
    --max-files    2500
    --resume
)

echo "============================================================"
echo "EVAL SA2 — FMA-large test split (11,263 tracks) — 1 GPU (step=500000)"
echo "Target embeddings cached to: $CACHE_DIR"
echo "============================================================"

echo "[$(date)] Launching evaluation task..."

# GPU 0 runs the 500k checkpoint
CUDA_VISIBLE_DEVICES=0 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_sao.py \
    --checkpoint "${SA2_CKPTS[0]}" \
    --output-dir "$EVAL_ROOT" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU0] /'

echo "============================================================"
echo "[$(date)] SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
