#!/bin/bash
# ============================================================
# eval_O8_EMA_recon: FMA-large recon eval of O8_EMA
#   Output: runs/eval_swin_large/O8_EMA/metrics/
#   Launch with: sbatch --dependency=afterany:<O8_EMA_JOB_ID> eval_O8_EMA_recon.slurm
# ============================================================
#SBATCH --job-name=eval_O8_EMA_recon
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a; source $WORK/C-VAE/.env; set +a
cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

CACHE_DIR="$SCRATCH/eval_cache"; mkdir -p "$CACHE_DIR"

# ── resolve latest O8_EMA checkpoint ───────────────────────────────────────────
CK="$(ls -t "$WORK/C-VAE/runs/O8_EMA"/*/checkpoints/last.ckpt 2>/dev/null | head -1)"
if [ -z "$CK" ] || [ ! -f "$CK" ]; then
    echo "[ERR] O8_EMA checkpoint not found under runs/O8_EMA" >&2; exit 2
fi
echo "[eval_O8_EMA_recon] ckpt=$CK"

LN="checkpoints/O8_EMA.ckpt"
ln -sfn "$CK" "$LN"

srun python evaluation/evaluate_swin_varT.py \
    --checkpoint "$LN" \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_large \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 16 \
    --max-files 2500 \
    --resume
