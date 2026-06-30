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

MODEL="${1:?Usage: sbatch eval_sota_10s.slurm <codicodec|music2latent|sao-vae|same> [--data-dir DIR] [extra args]}"
shift

# Optional dataset override (default: moisesdb chunks_mix_original). Intercept
# --data-dir / --dset-tag here so they are NOT forwarded to python; everything
# else flows through as EXTRA. Output lands under runs/eval_sota_10s/<tag>/.
DATA_DIR="$FAST/datasets/moisesdb/chunks_mix_original/original"
DSET_TAG=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --dset-tag) DSET_TAG="$2"; shift 2;;
    *)          EXTRA+=("$1"); shift;;
  esac
done
[ -z "$DSET_TAG" ] && DSET_TAG="$(basename "$(dirname "$DATA_DIR")")"

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

EVAL_ROOT="$WORK/C-VAE/runs/eval_sota_10s/$DSET_TAG"

mkdir -p "$EVAL_ROOT"

echo "============================================================"
echo "EVAL SOTA 10s — model=$MODEL  dataset=$DSET_TAG"
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
