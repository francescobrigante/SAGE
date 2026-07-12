#!/bin/bash
# ============================================================
# eval_ft_internal.slurm — INTERNAL model-selection eval for the
# FT_LARGE2_mr05_decFT_postnet finetune.
#
# Reconstruction metrics ONLY (spectral + clap + cdpam + fad_mert + fad_gudgud),
# on a FIXED subset of the FMA-large test split (--max-files, first-N sorted →
# same files for every checkpoint → consistent INTERNAL ranking). These numbers
# are NOT directly comparable to the full-test-set evals (different file count):
# the goal is to rank the finetune's own checkpoints (every ~20 epochs) and pick
# the best, which is THEN re-evaluated on the full test set.
#
# Usage:
#   sbatch evaluation/eval_ft_internal.slurm [EPOCH ...]   # default: 19
#   sbatch evaluation/eval_ft_internal.slurm 19 39 59
#   sbatch evaluation/eval_ft_internal.slurm base 19       # +LARGE2 start baseline
#   MAXFILES=2000 sbatch evaluation/eval_ft_internal.slurm 19
#   SRC_RUN=cooldown sbatch evaluation/eval_ft_internal.slurm 49   # cooldown run
#
# For each EPOCH it: globs epoch_<EPOCH>.ckpt across all run dirs (robust to
# requeue), symlinks it to checkpoints/FT_LARGE2_mr05_e<EPOCH>.ckpt, and feeds
# all of them to one evaluate_swin_varT.py pass (shared model setup).
# The token "base" (or "large2") evaluates the LARGE2_mrmel05_e99 STARTING
# checkpoint on the same subset → in-table reference row for the finetune.
# ============================================================
#SBATCH --job-name=eval_ft_internal
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

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=8       # multi-thread BLAS speeds the final FAD covariance
export MKL_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

CACHE_DIR="$SCRATCH/eval_cache"
mkdir -p "$CACHE_DIR"

MAXFILES="${MAXFILES:-2500}"

# SRC_RUN selects which run's checkpoints to evaluate (default = finetune).
#   SRC_RUN=cooldown  → evaluate the WSD cooldown run instead (stem COOLDOWN_LARGE2_mr05_e<E>).
SRC_RUN="${SRC_RUN:-finetune}"
case "$SRC_RUN" in
    cooldown)
        RUN_BASE="$WORK/C-VAE/runs/COOLDOWN_LARGE2_mr05"
        STEM_PREFIX="COOLDOWN_LARGE2_mr05_e" ;;
    finetune|*)
        RUN_BASE="$WORK/C-VAE/runs/FT_LARGE2_mr05_decFT_postnet"
        STEM_PREFIX="FT_LARGE2_mr05_e" ;;
esac
echo "[SRC] SRC_RUN=$SRC_RUN → RUN_BASE=$RUN_BASE (stem prefix: ${STEM_PREFIX})"

EPOCHS=("$@")
[ ${#EPOCHS[@]} -eq 0 ] && EPOCHS=(19)

# ─── Resolve each epoch checkpoint + symlink into checkpoints/ ────────────────
CKPTS=()
for E in "${EPOCHS[@]}"; do
    # Baseline: the LARGE2 starting checkpoint (no postnet) on the same subset.
    if [ "$E" = "base" ] || [ "$E" = "large2" ]; then
        LINK="$WORK/C-VAE/checkpoints/LARGE2_mrmel05_e99.ckpt"
        if [ ! -e "$LINK" ]; then
            echo "[WARN] LARGE2 baseline ckpt not found: $LINK — skipping" >&2
            continue
        fi
        echo "[BASE] $LINK (LARGE2 start → stem LARGE2_mrmel05_e99)"
        CKPTS+=("$LINK")
        continue
    fi
    EP=$(printf "%03d" "$E")
    SRC=$(ls -t "$RUN_BASE"/*/checkpoints/epoch_${EP}.ckpt 2>/dev/null | head -1)
    if [ -z "$SRC" ]; then
        echo "[WARN] epoch_${EP}.ckpt not found under $RUN_BASE/*/checkpoints/ — skipping" >&2
        continue
    fi
    LINK="$WORK/C-VAE/checkpoints/${STEM_PREFIX}${E}.ckpt"
    ln -sf "$SRC" "$LINK"
    echo "[LINK] $LINK -> $SRC"
    CKPTS+=("$LINK")
done

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[FATAL] no checkpoints resolved; nothing to evaluate" >&2
    exit 1
fi

echo "============================================================"
echo "[$(date)] INTERNAL recon eval — max-files=$MAXFILES"
echo "checkpoints: ${CKPTS[*]}"
echo "output:      $WORK/C-VAE/runs/eval_ft_internal"
echo "============================================================"

srun $WORK/.envs/c_vae/bin/python3 evaluation/evaluate_swin_varT.py \
    --checkpoint  "${CKPTS[@]}" \
    --target-dir  $FAST/datasets/fma/fma_large \
    --output-dir  $WORK/C-VAE/runs/eval_ft_internal \
    --cache-dir   "$CACHE_DIR" \
    --num-workers 8 \
    --max-files   "$MAXFILES" \
    --resume

echo "============================================================"
echo "[$(date)] DONE → runs/eval_ft_internal/FT_LARGE2_mr05_e<EPOCH>/metrics/"
echo "============================================================"
