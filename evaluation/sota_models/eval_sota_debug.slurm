#!/bin/bash
# ============================================================
# eval_sota_debug.slurm
# Debug gate for sota_models: validates the shared pipeline (identity self-test,
# expect ~perfect metrics) then a first real codicodec reconstruction, both on a
# few FMA-test files. 1 A100, boost_qos_dbg (max 30 min, high priority).
# Uses the SHARED eval cache so FAD target embeddings match SAO/Swin.
# ============================================================
#SBATCH --job-name=eval_sota_dbg
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_sota_dbg_%j.out
#SBATCH --error=logs/eval_sota_dbg_%j.err

set -euo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH                                    # critical: isolate the sota venv
source $WORK/.envs/sota/bin/activate

export HF_HOME=$FAST/hf_cache                       # weights + datasets cache
export http_proxy='http://login07:3150'             # compute nodes reach internet via squid
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

FMA=/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma
TARGET_DIR="$FMA/fma_large"
FMA_CSV="$FMA/fma_metadata/tracks.csv"
EVAL_ROOT="$WORK/C-VAE/runs/eval_sota_debug"
CACHE_DIR="$SCRATCH/eval_cache"                     # SHARED with SAO/Swin → comparable FAD
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

echo "============================================================"
echo "[$(date)] SOTA debug gate — identity self-test + codicodec (few files)"
echo "============================================================"

echo "[$(date)] --- M0: identity self-test (3 files) — expect SI-SDR huge, FAD~0, CLAP~1 ---"
python -u evaluation/sota_models/evaluate_sota.py \
    --model identity \
    --target-dir "$TARGET_DIR" --fma-csv-path "$FMA_CSV" \
    --output-dir "$EVAL_ROOT" --cache-dir "$CACHE_DIR" \
    --device cuda --num-workers 8 --max-files 3

echo "[$(date)] --- M1: codicodec (5 files) — first real reconstruction ---"
python -u evaluation/sota_models/evaluate_sota.py \
    --model codicodec \
    --target-dir "$TARGET_DIR" --fma-csv-path "$FMA_CSV" \
    --output-dir "$EVAL_ROOT" --cache-dir "$CACHE_DIR" \
    --device cuda --num-workers 8 --max-files 5

echo "============================================================"
echo "[$(date)] DONE — results in $EVAL_ROOT/{identity,codicodec}/metrics/"
echo "============================================================"
