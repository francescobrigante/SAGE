#!/bin/bash
# ============================================================
# eval_swin_large.slurm
# Evaluate the 2 Swin C-VAE checkpoints on FMA-large test split
# (3,000 files budget, resume support). One node, 1 GPU, 48-hour window.
#
# Target embeddings (CLAP/MERT) cached to $SCRATCH/eval_cache/
# and reused across checkpoints and future runs.
#
# CDPAM skipped (too slow at scale — run debug job for it).
# Use --resume to restart after preemption without recomputing.
# ============================================================
#SBATCH --job-name=eval_swin_large
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/eval_swin_large_%j.out
#SBATCH --error=logs/eval_swin_large_%j.err

set -euo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1   # plain-text logs — ANSI codes make SLURM output unreadable
export PYTHONUNBUFFERED=1

cd $WORK/C-VAE

TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_large"
EVAL_ROOT="$WORK/C-VAE/runs/eval_swin_large"
CACHE_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/eval_cache"

mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

SWIN_CKPTS=(
    "checkpoints/STFT_3seconds.ckpt"
    "checkpoints/baseline.ckpt"
)

COMMON_FLAGS=(
    --target-dir        "$TARGET_DIR"
    --shared-cache-dir  "$CACHE_DIR"
    --num-workers       8
    --batch-size        1
    --max-files         2500
    --resume
)

echo "============================================================"
echo "EVAL SWIN — FMA-large test split (3,000 files budget) — 2 GPUs (Parallel)"
echo "Target embeddings cached to: $CACHE_DIR"
echo "============================================================"

echo "[$(date)] Launching parallel GPU evaluation tasks..."

# GPU 0 runs STFT_3seconds.ckpt — loads heavy models ONLY ONCE
CUDA_VISIBLE_DEVICES=0 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_swin.py \
    --checkpoint "${SWIN_CKPTS[0]}" \
    --output-dir "$EVAL_ROOT" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU0] /' &
PID0=$!

# GPU 1 runs baseline.ckpt — loads heavy models ONLY ONCE
CUDA_VISIBLE_DEVICES=1 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_swin.py \
    --checkpoint "${SWIN_CKPTS[1]}" \
    --output-dir "$EVAL_ROOT" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU1] /' &
PID1=$!

wait $PID0 $PID1

echo "============================================================"
echo "[$(date)] SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
