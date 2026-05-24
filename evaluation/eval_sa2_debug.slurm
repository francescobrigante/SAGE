#!/bin/bash
# ============================================================
# eval_sa2_debug.slurm
# Smoke test — all metrics including CDPAM on 10 files/ckpt.
# Run this before eval_sa2_large.slurm to verify the pipeline
# and warm up model weight caches.
#
# Expected runtime: ~15–30 min (first run, with HF downloads)
#                   ~5–10 min (subsequent runs)
# ============================================================
#SBATCH --job-name=eval_sa2_dbg
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_sa2_debug_%j.out
#SBATCH --error=logs/eval_sa2_debug_%j.err

set -euo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1

cd $WORK/C-VAE

TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_large"
EVAL_ROOT="$WORK/C-VAE/runs/eval_sa2_debug"
CACHE_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/eval_cache"
MAX_FILES=100

# Always start fresh — debug runs are short, stale state causes spurious skips
rm -rf "$EVAL_ROOT"
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
    --max-files         "$MAX_FILES"
)

echo "============================================================"
echo "EVAL SA2 DEBUG — ${MAX_FILES} files, all metrics, 2 GPUs"
echo "Cache: $CACHE_DIR"
echo "============================================================"

# ── Checkpoints 0 and 1 in parallel ───────────────────────────
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
echo "[$(date)] DEBUG SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
