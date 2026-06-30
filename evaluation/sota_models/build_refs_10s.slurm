#!/bin/bash
# ============================================================
# build_refs_10s.slurm — precompute clean-original reference embeddings + FAD
# stats for a flat 10s WAV dir, so the *_10s evaluators can score any model on
# it. Run ONCE per dataset. Single GPU.
#
# Usage:
#   sbatch --export=NONE evaluation/sota_models/build_refs_10s.slurm \
#          $FAST/datasets/musiccaps/balanced_44k_stereo/audio
#   sbatch --export=NONE evaluation/sota_models/build_refs_10s.slurm \
#          $FAST/datasets/song_describer/sdd_44k_stereo_10s/audio
#
# Quick debug (20 files):
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=NONE \
#          evaluation/sota_models/build_refs_10s.slurm <data_dir> --max-files 20
# ============================================================
#SBATCH --job-name=build_refs_10s
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_refs_10s_%j.out
#SBATCH --error=logs/build_refs_10s_%j.err

set -uo pipefail
mkdir -p logs

DATA_DIR="${1:?Usage: sbatch build_refs_10s.slurm <data_dir> [extra args]}"
shift
EXTRA=("$@")

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH
source $WORK/.envs/sota/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

echo "============================================================"
echo "BUILD REFS 10s"
echo "data:   $DATA_DIR"
echo "writes: $DATA_DIR/{embeddings,stats_ours}/"
echo "extra:  ${EXTRA[*]-}"
echo "============================================================"
echo "[$(date)] Starting..."

python -u evaluation/sota_models/build_refs_10s.py \
    --data-dir    "$DATA_DIR" \
    --device      cuda \
    --num-workers 8 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "============================================================"
echo "[$(date)] DONE → $DATA_DIR/{embeddings,stats_ours}/"
echo "============================================================"
