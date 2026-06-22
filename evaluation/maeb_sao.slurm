#!/bin/bash
# ============================================================
# evaluation/maeb_sao.slurm
# Run MAEB(audio-only) with the SAO-ACE encoder on 1 A100.
#
# Uses the ISOLATED maeb_dl venv (its own torch 2.4.1) — SAO is a plain conv
# VAE, no cineca-custom torch needed. Datasets are read from the warm $FAST
# cache (Phase A), so the offline compute node needs no internet.
#
#   sbatch evaluation/maeb_sao.slurm <ckpt.ckpt> [model_config.json]
# ============================================================
#SBATCH --job-name=maeb_sao
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/maeb_sao_%j.out
#SBATCH --error=logs/maeb_sao_%j.err

set -euo pipefail
mkdir -p logs

cd $WORK/C-VAE

# Isolated venv: do NOT module-load cineca-ai (it injects a PYTHONPATH that
# breaks mteb). purge modules and unset PYTHONPATH for the same reason — see CINECA.md §5.4.
module purge || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source $WORK/.envs/maeb_dl/bin/activate

export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1            # datasets already cached → no internet needed
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${1:?Usage: sbatch evaluation/maeb_sao.slurm <ckpt.ckpt> [model_config.json] [extra args]}"
shift
EXTRA=()
# Optional positional model_config (a .json) kept for backward compatibility.
if [ "${1:-}" ] && [[ "$1" == *.json ]]; then
    EXTRA+=(--model-config "$1"); shift
fi
EXTRA+=("$@")   # remaining args forwarded (e.g. --tasks --max-files)

echo "============================================================"
echo "MAEB SAO-ACE — FMA suite — 1 GPU"
echo "ckpt: $CKPT"
echo "extra args: ${EXTRA[*]}"
echo "cache: $HF_DATASETS_CACHE"
echo "============================================================"

python evaluation/maeb_sao.py \
    --ckpt "$CKPT" \
    --output-dir "$WORK/C-VAE/runs/maeb_sao" \
    --device cuda \
    --batch-size 1 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "============================================================"
echo "[$(date)] MAEB SAO-ACE DONE — results in $WORK/C-VAE/runs/maeb_sao"
echo "============================================================"
