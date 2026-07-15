#!/bin/bash
# =====================================================================
# maeb_swiglu_muon_ep499: MAEB 15-task eval for
# MULTICORPUS_swiglu_muon_500ep @ep499 (FMA + MoisesDB, one GPU job).
# Loads ONLY profile/deeplrn (NOT cineca-ai/4.3.0) to avoid pydantic v1
# contamination in the maeb_dl venv (which requires pydantic v2).
#
# Account IscrC_MM (IscrC_AHNetBio budget exhausted).
# Output: runs/multicorpus_eval/maeb/maeb_MULTICORPUS_swiglu_muon_ep499/
# =====================================================================
#SBATCH --job-name=maeb_swiglu_muon_ep499
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn

export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export HF_HOME=$FAST/hf_cache
export PYTHONUNBUFFERED=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export TF_CPP_MIN_LOG_LEVEL=3

set -a; source $WORK/C-VAE/.env; set +a
cd $WORK/C-VAE

CKPT_LINK="checkpoints/MULTICORPUS_swiglu_muon_ep499.ckpt"
EVAL_ROOT="$WORK/C-VAE/runs/multicorpus_eval"
TAG="MULTICORPUS_swiglu_muon_ep499"

echo "[$(date '+%T')] MAEB — $TAG"
unset PYTHONPATH
source $WORK/.envs/maeb_dl/bin/activate

python evaluation/maeb_swin.py \
    --ckpt "$CKPT_LINK" \
    --with-moisesdb --standardize-bottleneck \
    --output-dir "$EVAL_ROOT/maeb/maeb_$TAG" \
    --device cuda

# Upstream MAEB music tasks (6) on original datasets — merges into same summary.
python evaluation/maeb_swin.py \
    --ckpt "$CKPT_LINK" \
    --maeb-original-music-only --standardize-bottleneck \
    --output-dir "$EVAL_ROOT/maeb/maeb_$TAG" \
    --device cuda

echo "[$(date '+%T')] MAEB DONE — $TAG"
