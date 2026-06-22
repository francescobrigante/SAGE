#!/bin/bash
# ============================================================
# evaluation/maeb_sota.slurm
# Run MAEB(audio-only) with a SOTA baseline codec (codicodec | music2latent)
# on 1 A100, using the ISOLATED maeb_dl venv (torch 2.4.1+cu121 — cuDNN OK on
# Leonardo's CUDA-12.2 driver). The two codecs are installed into maeb_dl
# (--no-deps); datasets are read from the warm $FAST cache (offline node).
#
#   sbatch evaluation/maeb_sota.slurm codicodec    [extra args]
#   sbatch evaluation/maeb_sota.slurm music2latent --max-files 2000 --overwrite
# ============================================================
#SBATCH --job-name=maeb_sota
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/maeb_sota_%j.out
#SBATCH --error=logs/maeb_sota_%j.err

set -euo pipefail
mkdir -p logs
cd $WORK/C-VAE

# Isolated venv: do NOT module-load cineca-ai; purge modules and unset PYTHONPATH (CINECA.md §5.4).
module purge || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source $WORK/.envs/maeb_dl/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:?Usage: sbatch evaluation/maeb_sota.slurm <codicodec|music2latent> [extra args]}"
shift
EXTRA=("$@")

echo "============================================================"
echo "MAEB SOTA — FMA suite — model=$MODEL — 1 GPU"
echo "extra args: ${EXTRA[*]-}"
echo "============================================================"

python evaluation/maeb_sota.py \
    --model "$MODEL" \
    --output-dir "$WORK/C-VAE/runs/maeb_$MODEL" \
    --device cuda \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "============================================================"
echo "[$(date)] MAEB SOTA ($MODEL) DONE — results in $WORK/C-VAE/runs/maeb_$MODEL"
echo "============================================================"
