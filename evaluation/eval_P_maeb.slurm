#!/bin/bash
# ============================================================
# eval_P_maeb.slurm  —  MAEB (FMA suite, standardized) for the P-sweep
#                       into runs/P_sweep_eval/maeb
# Usage:  sbatch eval_P_maeb.slurm <TAG> <RUN_NAME|CKPT_PATH>
#   (same arg convention as eval_P_recon.slurm)
# Isolated maeb_dl venv: do NOT module-load cineca-ai, unset PYTHONPATH.
# ============================================================
#SBATCH --job-name=eval_P_maeb
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs
cd $WORK/C-VAE

unset PYTHONPATH
source $WORK/.envs/maeb_dl/bin/activate
export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG="$1"; SRC="$2"
OUT="$WORK/C-VAE/runs/P_sweep_eval/maeb"
mkdir -p "$OUT"

# ── resolve checkpoint ───────────────────────────────────────────────────────
if [ -f "$SRC" ]; then
    CK="$(readlink -f "$SRC")"
else
    CK="$(readlink -f "$(ls -t runs/$SRC/*/checkpoints/last.ckpt 2>/dev/null | head -1)")"
fi
if [ -z "$CK" ] || [ ! -f "$CK" ]; then echo "[ERR] checkpoint not found for '$SRC'"; exit 2; fi

# ── properly-named symlink so run name == TAG (not 'last') ───────────────────
LN="checkpoints/${TAG}.ckpt"
ln -sfn "$CK" "$LN"
echo "[eval_P_maeb] TAG=$TAG  ckpt=$CK  -> $OUT/cvae-real__${TAG}__std"

python evaluation/maeb_swin.py \
    --ckpt "$LN" \
    --output-dir "$OUT" \
    --device cuda \
    --batch-size 1 \
    --max-files 2000 \
    --overwrite
