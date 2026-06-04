#!/bin/bash
# ============================================================
# evaluation/maeb_swin.slurm
# Run MAEB(audio-only) with the Swin C-VAE encoder on 1 A100.
#
# Uses the ISOLATED maeb_dl venv (its own torch 2.4.1).
# complextorch/complexpytorch are pure-Python — no cineca-custom
# torch needed. src/ is injected into sys.path automatically by
# swin_encoder.py. Datasets are read from the warm $FAST cache.
#
#   sbatch evaluation/maeb_swin.slurm <ckpt.ckpt>
# ============================================================
#SBATCH --job-name=maeb_swin
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/maeb_swin_%j.out
#SBATCH --error=logs/maeb_swin_%j.err

set -euo pipefail
mkdir -p logs

cd $WORK/C-VAE

# Isolated venv: do NOT module-load cineca-ai (it injects a PYTHONPATH that
# breaks mteb). unset PYTHONPATH for the same reason — see CINECA.md §5.4.
unset PYTHONPATH
source $WORK/.envs/maeb_dl/bin/activate

export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1            # datasets already cached → no internet needed
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${1:?Usage: sbatch evaluation/maeb_swin.slurm <ckpt.ckpt> [extra maeb_swin.py args]}"
shift   # remaining args ("$@") forwarded to maeb_swin.py (e.g. --tasks --max-files)

echo "============================================================"
echo "MAEB Swin C-VAE — FMA suite — 1 GPU"
echo "ckpt: $CKPT"
echo "extra args: $*"
echo "cache: $HF_DATASETS_CACHE"
echo "============================================================"

python evaluation/maeb_swin.py \
    --ckpt "$CKPT" \
    --output-dir "$WORK/C-VAE/runs/maeb_swin" \
    --device cuda \
    --batch-size 1 \
    "$@"

echo "============================================================"
echo "[$(date)] MAEB Swin C-VAE DONE — results in $WORK/C-VAE/runs/maeb_swin"
echo "============================================================"
