#!/bin/bash
# ============================================================
# eval_P_recon.slurm  —  recon eval (VarT) for the P-sweep into runs/P_sweep_eval/recon
# Usage:  sbatch eval_P_recon.slurm <TAG> <RUN_NAME|CKPT_PATH>
#   TAG               : output name (stem) e.g. P_clapW05 / O8_epoch49
#   RUN_NAME|CKPT     : a run dir name under runs/ (resolves */checkpoints/last.ckpt)
#                       OR a direct .ckpt path (used for baselines)
# ============================================================
#SBATCH --job-name=eval_P_recon
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

set -euo pipefail
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

TAG="$1"; SRC="$2"
OUT="$WORK/C-VAE/runs/P_sweep_eval/recon"
CACHE_DIR="$SCRATCH/eval_cache"; mkdir -p "$CACHE_DIR" "$OUT"

# ── resolve checkpoint ───────────────────────────────────────────────────────
if [ -f "$SRC" ]; then
    CK="$(readlink -f "$SRC")"
else
    CK="$(readlink -f "$(ls -t runs/$SRC/*/checkpoints/last.ckpt 2>/dev/null | head -1)")"
fi
if [ -z "$CK" ] || [ ! -f "$CK" ]; then echo "[ERR] checkpoint not found for '$SRC'"; exit 2; fi

# ── properly-named symlink so output stem == TAG (not 'last') ────────────────
LN="checkpoints/${TAG}.ckpt"
ln -sfn "$CK" "$LN"
echo "[eval_P_recon] TAG=$TAG  ckpt=$CK  -> $OUT/$TAG"

srun python evaluation/evaluate_swin_varT.py \
    --checkpoint "$LN" \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir "$OUT" \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 16 \
    --max-files 2500 \
    --resume
