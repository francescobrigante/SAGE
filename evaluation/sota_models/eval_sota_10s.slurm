#!/bin/bash
# ============================================================
# eval_sota_10s.slurm — eval ONE SOTA baseline on the colleague's
# chunks_mix_original dataset (1998 × 10s WAVs). Single GPU.
# Pre-computed reference embeddings/stats are read from DATA_DIR.
#
# Usage:
#   sbatch --export=NONE evaluation/sota_models/eval_sota_10s.slurm codicodec
#   sbatch --export=NONE evaluation/sota_models/eval_sota_10s.slurm music2latent
#   sbatch --export=NONE evaluation/sota_models/eval_sota_10s.slurm sao-vae
#   sbatch --export=NONE evaluation/sota_models/eval_sota_10s.slurm same
#
# Quick debug (20 files):
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=NONE \
#          evaluation/sota_models/eval_sota_10s.slurm codicodec --max-files 20
# ============================================================
#SBATCH --job-name=eval_sota_10s
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_sota_10s_%j.out
#SBATCH --error=logs/eval_sota_10s_%j.err

set -uo pipefail
mkdir -p logs

MODEL="${1:?Usage: sbatch eval_sota_10s.slurm <codicodec|music2latent|sao-vae|same> [extra args]}"
shift
EXTRA=("$@")

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH
source $WORK/.envs/sota/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

DATA_DIR="$FAST/datasets/moisesdb/chunks_mix_original/original"
EVAL_ROOT="$WORK/C-VAE/runs/eval_sota_10s"

mkdir -p "$EVAL_ROOT"

echo "============================================================"
echo "EVAL SOTA 10s — model=$MODEL"
echo "data:   $DATA_DIR"
echo "output: $EVAL_ROOT/$MODEL/metrics/"
echo "extra:  ${EXTRA[*]-}"
echo "============================================================"
echo "[$(date)] Starting..."

python -u evaluation/sota_models/evaluate_sota_10s.py \
    --model      "$MODEL" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$EVAL_ROOT" \
    --device     cuda \
    --num-workers 8 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "============================================================"
echo "[$(date)] DONE → $EVAL_ROOT/$MODEL/metrics/"
echo "============================================================"
