#!/bin/bash
# ============================================================
# eval_swin_10s.slurm — eval Swin C-VAE checkpoint on chunks_mix_original (10s)
#
#   sbatch --export=NONE evaluation/eval_swin_10s.slurm checkpoints/LARGE2_mrmel05_e99.ckpt
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=NONE \
#          evaluation/eval_swin_10s.slurm checkpoints/LARGE2_mrmel05_e99.ckpt --max-files 20
# ============================================================
#SBATCH --job-name=eval_swin_10s
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_swin_10s_%j.out
#SBATCH --error=logs/eval_swin_10s_%j.err

set -uo pipefail
mkdir -p logs

CKPT="${1:?Usage: sbatch eval_swin_10s.slurm <checkpoint_path> [--data-dir DIR] [extra args]}"
shift

# Optional dataset override (default: moisesdb chunks_mix_original). Intercept
# --data-dir / --dset-tag here so they are NOT forwarded to python.
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
source $WORK/.envs/c_vae/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

EVAL_ROOT="$WORK/C-VAE/runs/eval_swin_10s/$DSET_TAG"
mkdir -p "$EVAL_ROOT"

echo "============================================================"
echo "EVAL SWIN 10s — checkpoint=$CKPT  dataset=$DSET_TAG"
echo "data:   $DATA_DIR"
echo "output: $EVAL_ROOT"
echo "extra:  ${EXTRA[*]-}"
echo "============================================================"
echo "[$(date)] Starting..."

python -u evaluation/evaluate_swin_10s.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$EVAL_ROOT" \
    --device     cuda \
    --num-workers 8 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "============================================================"
echo "[$(date)] DONE → $EVAL_ROOT/$(basename ${CKPT%.ckpt})/metrics/"
echo "============================================================"
