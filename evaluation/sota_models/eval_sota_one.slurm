#!/bin/bash
# ============================================================
# eval_sota_one.slurm — single-model debug/eval on FMA test.
# Parametrized via env: MODEL (adapter key), MAXF (max files), EVAL_ROOT.
#   sbatch --export=ALL,MODEL=music2latent,MAXF=5 eval_sota_one.slurm
# 1 A100, boost_qos_dbg (<=30 min). Shared eval cache => FAD comparable to SAO/Swin.
# ============================================================
#SBATCH --job-name=eval_sota_one
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_sota_one_%j.out
#SBATCH --error=logs/eval_sota_one_%j.err

set -euo pipefail
mkdir -p logs

: "${MODEL:?set MODEL=<adapter key>}"
: "${MAXF:=5}"
: "${EVAL_ROOT:=$WORK/C-VAE/runs/eval_sota_debug}"

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH                                    # isolate the sota venv
source $WORK/.envs/sota/bin/activate

export HF_HOME=$FAST/hf_cache
export HF_HUB_OFFLINE=1                             # weights already cached; no fetch
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE
FMA=/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma
CACHE_DIR="$SCRATCH/eval_cache"                     # SHARED with SAO/Swin
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

echo "[$(date)] eval_sota_one — MODEL=$MODEL MAXF=$MAXF EVAL_ROOT=$EVAL_ROOT"
python -u evaluation/sota_models/evaluate_sota.py \
    --model "$MODEL" \
    --target-dir "$FMA/fma_large" --fma-csv-path "$FMA/fma_metadata/tracks.csv" \
    --output-dir "$EVAL_ROOT" --cache-dir "$CACHE_DIR" \
    --device cuda --num-workers 8 --max-files "$MAXF"
echo "[$(date)] DONE -> $EVAL_ROOT/$MODEL/metrics/"
