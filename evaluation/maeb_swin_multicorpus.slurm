#!/bin/bash
# ============================================================
# maeb_swin_multicorpus.slurm — MAEB for an ARBITRARY multicorpus Swin checkpoint.
#   Reproduces the 15-task suite used by the MULTICORPUS notebook tables 3 & 4
#   (6 FMA + 7 MoisesDB-30s + GTZANGenre + MusicGenreClustering) for any ckpt.
#   Two passes into ONE output dir (runner.py merges, non-destructive):
#     pass 1: --with-moisesdb           → FMA (6) + MoisesDB (7)
#     pass 2: --maeb-original-music-only → upstream music tasks (GTZAN/MusicGenre)
#   64-d protocol (--standardize-bottleneck), matching the existing entries.
#
#   OUT = runs/multicorpus_eval/maeb/maeb_<ckpt-stem>  (exactly what the notebook
#   reads via _maeb_root). Compute on IscrC_MM (AHNetBio budget exhausted); WORK/
#   FAST forced to IscrC_AHNetBio where repo/venv/checkpoints/HF cache live.
#
#   sbatch --account=IscrC_MM --qos=normal \
#          evaluation/maeb_swin_multicorpus.slurm checkpoints/FT_swiglu_ep359_e1299.ckpt
#   # smoke:
#   sbatch --account=IscrC_MM --qos=boost_qos_dbg --time=00:25:00 \
#          evaluation/maeb_swin_multicorpus.slurm checkpoints/FT_swiglu_ep359_e1299.ckpt --max-files 20
# ============================================================
#SBATCH --job-name=maeb_mc_swin
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

# Compute on IscrC_MM, but WORK/FAST point to IscrC_AHNetBio (user-owned, world-
# readable → readable from an IscrC_MM job). Mirrors final_eval/maeb_swin.slurm.
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

mkdir -p logs
cd $WORK/C-VAE

CKPT="${1:?Usage: sbatch maeb_swin_multicorpus.slurm <checkpoint_path> [extra args]}"
shift
STEM="$(basename "${CKPT%.ckpt}")"
# OUT overridable via env (e.g. final_eval); default = multicorpus_eval namespace.
: "${OUT:=$WORK/C-VAE/runs/multicorpus_eval/maeb/maeb_${STEM}}"

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

echo "============================================================"
echo "[$(date)] MULTICORPUS MAEB (15 tasks) — ckpt=$CKPT  stem=$STEM"
echo "output: $OUT"
echo "extra:  $*"
echo "============================================================"

# Pass 1: FMA (6) + MoisesDB (7).
python evaluation/maeb_swin.py \
    --ckpt "$CKPT" \
    --with-moisesdb --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

# Pass 2: upstream MAEB music tasks — merges into the same summary.json.
python evaluation/maeb_swin.py \
    --ckpt "$CKPT" \
    --maeb-original-music-only --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

echo "============================================================"
echo "[$(date)] DONE → $OUT"
echo "============================================================"
