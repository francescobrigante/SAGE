#!/bin/bash
# ============================================================
# final_eval/maeb_sota.slurm — FINAL paper eval.
# ALL 15 MAEB tasks (6 FMA + GTZAN/MusicGenre extras + 7 MoisesDB-30s) for ONE
# SOTA baseline, 1 GPU. MoisesDB uses chunks_30s (the metadata-bearing set).
# Writes into runs/final_eval/maeb/maeb_<model>/ (isolated).
# NOTE: 'same' is native 256-d (NOT width-matched) — tabulate with a caveat.
#
#   sbatch evaluation/final_eval/maeb_sota.slurm sao-vae
#   sbatch evaluation/final_eval/maeb_sota.slurm same
#   # smoke:
#   sbatch --qos=boost_qos_dbg --time=00:25:00 \
#          evaluation/final_eval/maeb_sota.slurm codicodec --max-files 20
# ============================================================
#SBATCH --job-name=fe_maeb_sota
#SBATCH --account=IscrC_AHNetBio
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

MODEL="${1:?Usage: sbatch maeb_sota.slurm <sao-vae|codicodec|music2latent|same> [extra args]}"
shift
EXTRA=("$@")

: "${OUT:=$WORK/C-VAE/runs/final_eval/maeb/maeb_$MODEL}"
echo "[$(date)] FINAL MAEB (all 15 tasks) — model=$MODEL — out=$OUT"

python evaluation/maeb_sota.py \
    --model "$MODEL" \
    --with-extra --with-moisesdb \
    --output-dir "$OUT" \
    --device cuda \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[$(date)] DONE → $OUT"
