#!/bin/bash
# ============================================================
# final_eval/recon_fma_swin_tag.slurm — como recon_fma_swin.slurm ma con
# CHECKPOINT passato da CLI invece di cablato su SAGE_e299, e account
# EUHPC_D34_099 (IscrC_AHNetBio, usato dall'originale, e' chiuso: 0 residuo).
#
# SAGE reconstruction sulla FULL FMA-large test split (max-files 0), tutte le
# metriche incl. CDPAM, file-sharded su 2 GPU. Scrive in
# runs/final_eval/recon_fma/<CKPT_TAG>/metrics/. FAD target refs condivisi
# via $SCRATCH/eval_cache (stesso cache di ogni altro modello: FAD confrontabile).
#
# Tempo reale misurato per SAGE_e299 (stesso script, stesso dataset): 3h08m su
# 2 GPU shard, 11252/11263 file. time=05:00:00 lascia margine.
#
# Il fix variable-length (tri2) e' applicato di default da EuleroEncodeDecode
# (config/inference/varlen.yaml): nessun flag necessario, vedi VARLEN_SEAMS.md §8.4.
#
#   sbatch evaluation/final_eval/recon_fma_swin_tag.slurm SAGE_e992_tri2
#   # smoke:
#   sbatch --qos=boost_qos_dbg --time=00:20:00 \
#          evaluation/final_eval/recon_fma_swin_tag.slurm SAGE_e992_tri2 --max-files 12
# ============================================================
#SBATCH --job-name=fe_recon_fma_tag
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

cd "$WORK/C-VAE" || exit 1
mkdir -p logs

TAG="${1:?Usage: sbatch recon_fma_swin_tag.slurm <CKPT_TAG> [extra args]}"
shift

module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1

set -a
source $WORK/C-VAE/.env
set +a

source $WORK/.envs/c_vae/bin/activate

: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_fma}"
CACHE_DIR="$SCRATCH/eval_cache"                     # SHARED target refs → FAD comparable
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

echo "[$(date)] FINAL recon FMA — $TAG — 2-GPU shard — out=$EVAL_ROOT"

srun $WORK/.envs/c_vae/bin/python3 evaluation/evaluate_swin_varT.py \
    --checkpoint   checkpoints/$TAG.ckpt \
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

echo "[$(date)] DONE → $EVAL_ROOT/$TAG/metrics/"
