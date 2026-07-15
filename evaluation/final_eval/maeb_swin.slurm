#!/bin/bash
# ============================================================
# final_eval/maeb_swin.slurm — FINAL paper eval.
# ALL 15 MAEB tasks (6 FMA + GTZAN/MusicGenre extras + 7 MoisesDB-30s) for the
# SAGE (FT_LARGE2 e299) checkpoint, 1 GPU. 64-d protocol (--standardize-bottleneck,
# default: fold C×F_lat=64 → 64-d, matching SAO/SOTA). MoisesDB = chunks_30s.
# Writes into runs/final_eval/maeb/maeb_sage/ (isolated).
#
#   sbatch evaluation/final_eval/maeb_swin.slurm
#   # smoke:
#   sbatch --qos=boost_qos_dbg --time=00:25:00 \
#          evaluation/final_eval/maeb_swin.slurm --max-files 20
# ============================================================
#SBATCH --job-name=fe_maeb_swin
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Compute on IscrC_MM (IscrC_AHNetBio budget exhausted), but WORK/FAST must point
# to IscrC_AHNetBio where repo, venv, checkpoints and HF cache live (user-owned,
# world-readable → readable from an IscrC_MM job).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

mkdir -p logs
cd $WORK/C-VAE

# Isolated maeb_dl venv: purge modules + unset PYTHONPATH (CINECA.md §5.4).
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

: "${OUT:=$WORK/C-VAE/runs/final_eval/maeb/maeb_sage}"
echo "[$(date)] FINAL MAEB (all 15 tasks) — SAGE_e299 — out=$OUT"

# Pass 1: FMA (6) + MoisesDB (7).
python evaluation/maeb_swin.py \
    --ckpt checkpoints/SAGE_e299.ckpt \
    --with-moisesdb --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

# Pass 2: upstream MAEB music tasks (6) on original datasets — merges into the
# same summary.json ($OUT), non-destructive (runner.py merge).
python evaluation/maeb_swin.py \
    --ckpt checkpoints/SAGE_e299.ckpt \
    --maeb-original-music-only --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

echo "[$(date)] DONE → $OUT"
