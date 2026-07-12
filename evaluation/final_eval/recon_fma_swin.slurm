#!/bin/bash
# ============================================================
# final_eval/recon_fma_swin.slurm — FINAL paper eval.
# SAGE (FT_LARGE2 e299) reconstruction on the FULL FMA-large test split
# (max-files 0), ALL metrics incl. CDPAM, file-sharded across 2 GPUs.
# Writes into runs/final_eval/recon_fma/SAGE_e299/metrics/ (isolated from
# every prior run). FAD target refs reuse the shared $SCRATCH/eval_cache.
#
#   sbatch --time=30:00:00 evaluation/final_eval/recon_fma_swin.slurm
#   # smoke (12 files, throwaway out):
#   sbatch --qos=boost_qos_dbg --time=00:20:00 \
#          --export=ALL,EVAL_ROOT=$WORK/C-VAE/runs/final_eval/_smoke/recon_fma \
#          evaluation/final_eval/recon_fma_swin.slurm --max-files 12
# ============================================================
#SBATCH --job-name=fe_recon_fma_swin
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=30:00:00
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
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_fma}"
CACHE_DIR="$SCRATCH/eval_cache"                     # SHARED target refs → FAD comparable
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

echo "[$(date)] FINAL recon FMA — SAGE_e299 — 2-GPU shard — out=$EVAL_ROOT"

srun $WORK/.envs/c_vae/bin/python3 evaluation/evaluate_swin_varT.py \
    --checkpoint   checkpoints/SAGE_e299.ckpt \
    --dataset      fma \
    --target-dir   $FAST/datasets/fma/fma_large \
    --fma-csv-path $FAST/datasets/fma/fma_metadata/tracks.csv \
    --output-dir   "$EVAL_ROOT" \
    --cache-dir    "$CACHE_DIR" \
    --num-workers  16 \
    --max-files    0 \
    --shard-files \
    --resume \
    "$@"

echo "[$(date)] DONE → $EVAL_ROOT/SAGE_e299/metrics/"
