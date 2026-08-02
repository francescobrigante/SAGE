#!/bin/bash
# ============================================================
# final_eval/recon_moises10s_swin.slurm — FINAL paper eval.
# SAGE (FT_LARGE2 e299) reconstruction on the FULL MoisesDB-10s set
# (chunks_mix_original/original, 1998 WAVs, max-files 0), ALL metrics.
# SINGLE GPU (evaluate_swin_10s.py has no sharding; 1998 files ≈ 1.5-2h).
# Uses the dataset-bundled ref stats → FAD comparable to the SOTA 10s jobs.
# Writes into runs/final_eval/recon_moisesdb_10s/SAGE_e299/metrics/.
#
#   sbatch evaluation/final_eval/recon_moises10s_swin.slurm
#   # smoke:
#   sbatch --qos=boost_qos_dbg --time=00:20:00 \
#          --export=ALL,EVAL_ROOT=$WORK/C-VAE/runs/final_eval/_smoke/recon_moisesdb_10s \
#          evaluation/final_eval/recon_moises10s_swin.slurm --max-files 12
# ============================================================
#SBATCH --job-name=fe_recon_m10s_swin
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

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
# Compute on IscrC_MM (IscrC_AHNetBio budget exhausted), but WORK/FAST must point to
# IscrC_AHNetBio where repo, venv, datasets and HF cache live (user-owned, world-readable).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# DATA_DIR overridable via env (default = mixtures) → same wrapper serves stems too.
: "${DATA_DIR:=$FAST/datasets/moisesdb/chunks_mix_original/original}"
: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_moisesdb_10s}"
mkdir -p "$EVAL_ROOT"

echo "[$(date)] FINAL recon MoisesDB-10s — SAGE_e299 — 1 GPU — out=$EVAL_ROOT"

python -u evaluation/evaluate_swin_10s.py \
    --checkpoint  checkpoints/SAGE_e299.ckpt \
    --data-dir    "$DATA_DIR" \
    --output-dir  "$EVAL_ROOT" \
    --device      cuda \
    --num-workers 8 \
    --resume \
    "$@"

echo "[$(date)] DONE → $EVAL_ROOT/SAGE_e299/metrics/"
