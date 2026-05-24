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
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/eval_sa2_large_%j.out
#SBATCH --error=logs/eval_sa2_large_%j.err

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
EVAL_ROOT="$WORK/C-VAE/runs/eval_sa2_large"
CACHE_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/eval_cache"

mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

SA2_CKPTS=(
    "checkpoints/sa2-step=100000.ckpt"
    "checkpoints/sa2-step=250000.ckpt"
    "checkpoints/sa2-step=500000.ckpt"
)

COMMON_FLAGS=(
    --target-dir        "$TARGET_DIR"
    --shared-cache-dir  "$CACHE_DIR"
    --num-workers       4
    --batch-size        16
    --max-files         3000
    --resume
)

echo "============================================================"
echo "EVAL SA2 — FMA-large test split (11,263 tracks) — 2 GPUs"
echo "Target embeddings cached to: $CACHE_DIR"
echo "============================================================"

# ── Checkpoints 0 and 1 in parallel (one per GPU) ─────────────
echo "[$(date)] Starting ckpt0 (GPU 0) and ckpt1 (GPU 1) in parallel..."

CUDA_VISIBLE_DEVICES=0 python3 -u evaluation/evaluate.py \
    --checkpoint "${SA2_CKPTS[0]}" \
    --output-dir "$EVAL_ROOT/$(basename ${SA2_CKPTS[0]} .ckpt)" \
    --device cuda \
    "${COMMON_FLAGS[@]}" \
    2>&1 | sed -u 's/^/[ckpt0] /' &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python3 -u evaluation/evaluate.py \
    --checkpoint "${SA2_CKPTS[1]}" \
    --output-dir "$EVAL_ROOT/$(basename ${SA2_CKPTS[1]} .ckpt)" \
    --device cuda \
    "${COMMON_FLAGS[@]}" \
    2>&1 | sed -u 's/^/[ckpt1] /' &
PID1=$!

wait $PID0 $PID1
echo "[$(date)] Parallel checkpoints done."

# ── Checkpoint 2 on GPU 0 ─────────────────────────────────────
echo "[$(date)] Starting ckpt2 (GPU 0)..."

CUDA_VISIBLE_DEVICES=0 python3 -u evaluation/evaluate.py \
    --checkpoint "${SA2_CKPTS[2]}" \
    --output-dir "$EVAL_ROOT/$(basename ${SA2_CKPTS[2]} .ckpt)" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1

echo "============================================================"
echo "[$(date)] SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
