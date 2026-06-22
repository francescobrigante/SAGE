#!/bin/bash
# ============================================================
# eval_sao_large_sa2disc.slurm — compute the two MISSING FADs
# (fad_mert MERT-v1-95M-4 + fad_gudgud clap-laion-audio-gud) for
# sa2Disc-step=500000 on FMA-large test (30s). Single normal pass
# with --skip-cdpam → ONE decode produces BOTH FADs (memory-safe:
# per-stem pred embeddings saved to disk + incremental FAD stats).
#
# Cache: reuses the shared target caches under $SCRATCH/eval_cache
#        (MERT-v1-95M-4/target, clap-laion-audio-gud/target).
# NO --resume on purpose: a stale metrics/_done marker exists and
# would otherwise short-circuit the whole run. evaluate_sao.py has
# no per-file done.txt, so there is no skip trap.
#
#   sbatch evaluation/eval_sao_large_sa2disc.slurm
# ============================================================
#SBATCH --job-name=fad_sa2disc_both
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

CACHE_DIR="$SCRATCH/eval_cache"
mkdir -p "$CACHE_DIR"

echo "============================================================"
echo "[$(date)] sa2Disc-500k 30s — BOTH FADs (mert + gudgud), skip-cdpam"
echo "cache:  $CACHE_DIR"
echo "output: $WORK/C-VAE/runs/eval_sa2_large/sa2Disc-step=500000/metrics/"
echo "============================================================"

python -u evaluation/evaluate_sao.py \
    --checkpoint   checkpoints/sa2Disc-step=500000.ckpt \
    --target-dir   $FAST/datasets/fma/fma_large \
    --output-dir   $WORK/C-VAE/runs/eval_sa2_large \
    --cache-dir    "$CACHE_DIR" \
    --fma-csv-path $FAST/datasets/fma/fma_metadata/tracks.csv \
    --num-workers  8 \
    --max-files    0 \
    --skip-cdpam

echo "============================================================"
echo "[$(date)] DONE → fad_mert.csv + fad_gudgud.csv written"
echo "============================================================"
