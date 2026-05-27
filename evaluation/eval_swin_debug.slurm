#!/bin/bash
# ============================================================
# eval_swin_debug.slurm
# Smoke test — all metrics including CDPAM on 100 files/ckpt
# for the Swin C-VAE baseline and STFT_3seconds checkpoints.
#
# Expected runtime: ~10–20 min
# ============================================================
#SBATCH --job-name=eval_swin_dbg
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_swin_debug_%j.out
#SBATCH --error=logs/eval_swin_debug_%j.err

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
EVAL_ROOT="$WORK/C-VAE/runs/eval_swin_debug"
CACHE_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/eval_cache"
MAX_FILES=100

# Always start fresh — debug runs are short, stale state causes spurious skips
rm -rf "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

SWIN_CKPTS=(
    "checkpoints/STFT_3seconds.ckpt"
    "checkpoints/baseline.ckpt"
)

COMMON_FLAGS=(
    --target-dir        "$TARGET_DIR"
    --num-workers       8
    --max-files         "$MAX_FILES"
)

echo "============================================================"
echo "EVAL SWIN DEBUG — ${MAX_FILES} files, all metrics, 2 GPUs (Parallel)"
echo "Cache: $CACHE_DIR"
echo "============================================================"

echo "[$(date)] Launching parallel GPU evaluation tasks..."

CUDA_VISIBLE_DEVICES=0 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_swin.py \
    --checkpoint "${SWIN_CKPTS[0]}" \
    --output-dir "$EVAL_ROOT/$(basename ${SWIN_CKPTS[0]} .ckpt)" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU0] /' &
PID0=$!

CUDA_VISIBLE_DEVICES=1 $WORK/.envs/c_vae/bin/python3 -u evaluation/evaluate_swin.py \
    --checkpoint "${SWIN_CKPTS[1]}" \
    --output-dir "$EVAL_ROOT/$(basename ${SWIN_CKPTS[1]} .ckpt)" \
    --device cuda \
    "${COMMON_FLAGS[@]}" 2>&1 | sed -u 's/^/[GPU1] /' &
PID1=$!

wait $PID0 $PID1

echo "============================================================"
echo "[$(date)] DEBUG SWEEP COMPLETED"
echo "Results in: $EVAL_ROOT/*/metrics/"
echo "============================================================"
