#!/bin/bash
# ============================================================
# eval_sa2_large.slurm
# Evaluate the 3 sa2 checkpoints on FMA-large test split
# (11,263 tracks). One node, 2 A100s, 48-hour window.
#
# Architecture:
#   - ckpt0 (GPU 0) and ckpt1 (GPU 1) run in parallel
#   - ckpt2 (GPU 0) runs sequentially after both finish
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
#SBATCH --gres=gpu:2
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
    "checkpoints/sa2-step=100000.ckpt"
    "checkpoints/sa2-step=250000.ckpt"
    "checkpoints/sa2-step=500000.ckpt"
)

COMMON_FLAGS=(
    --target-dir   "$TARGET_DIR"
    --cache-dir    "$CACHE_DIR"
    --num-workers  16
    --max-files    2500
    --resume
)

echo "============================================================"
echo "EVAL SA2 — FMA-large test split (11,263 tracks) — 2 GPUs (Parallel)"
echo "Target embeddings cached to: $CACHE_DIR"
echo "============================================================"

echo "[$(date)] Launching parallel GPU evaluation tasks..."

# GPU 0 runs ckpt0 and ckpt2 sequentially — loads heavy models ONLY ONCE
CUDA_VISIBLE_DEVICES=0 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_sao.py \
    --checkpoint "${SA2_CKPTS[0]}" "${SA2_CKPTS[2]}" \
    --output-dir "$EVAL_ROOT" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU0] /' &
PID0=$!

# GPU 1 runs ckpt1 — loads heavy models ONLY ONCE
CUDA_VISIBLE_DEVICES=1 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_sao.py \
    --checkpoint "${SA2_CKPTS[1]}" \
    --output-dir "$EVAL_ROOT" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU1] /' &
PID1=$!

wait $PID0 $PID1

echo "============================================================"
echo "[$(date)] SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
