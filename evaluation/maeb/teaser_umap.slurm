#!/bin/bash
# ============================================================
# evaluation/maeb/teaser_umap.slurm — SAGE paper teaser figure.
# Embeds MoisesDB instrument stems (chunks_30s, 7 instrument classes) with the
# PAPER multi-corpus encoder (MULTICORPUS_swiglu, AdamW — NOT the muon variant),
# using the exact MAEB protocol (SwinEncoder + prepare_audio + standardize_bottleneck).
#
# Encoder = pretrain MULTICORPUS_swiglu_500ep epoch 359 (the epoch the paper FT
# freezes and fine-tunes the decoder from → identical encoder). EMA weights are
# auto-selected by EuleroEncodeDecode. Architecture (SwiGLU FFN + XSA attention)
# is rebuilt from the checkpoint's own inference_config — no code change needed.
#
# Output: evaluation/maeb/figures/instruments_emb.npz  (transfer + re-UMAP locally)
#         + a default UMAP PNG/PDF as a sanity check.
# 1 GPU, short job (single encoder forward per stem).
#
#   sbatch evaluation/maeb/teaser_umap.slurm
# ============================================================
#SBATCH --job-name=teaser_umap
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Compute on IscrC_MM (IscrC_AHNetBio budget exhausted), but WORK/FAST must point
# to IscrC_AHNetBio where repo, venv, checkpoints, MoisesDB data and HF cache live
# (user-owned, world-readable → readable from an IscrC_MM job). Pointing FAST here
# also makes config.FAST_DIR resolve chunks_30s automatically.
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

CKPT="runs/MULTICORPUS_swiglu_500ep/2026-07-03_14-43-41/checkpoints/epoch_359.ckpt"
OUT="evaluation/maeb/figures"

echo "[$(date)] teaser UMAP — encoder=$CKPT — out=$OUT"

# No --embed-only: saves instruments_emb.npz FIRST, then also renders a default
# UMAP PNG/PDF sanity plot (the final figure style is redone locally from the npz).
python evaluation/maeb/plot_latent_umap.py \
    --ckpt "$CKPT" \
    --out  "$OUT" \
    --device cuda

echo "[$(date)] DONE → $OUT/instruments_emb.npz"
