#!/bin/bash
# ============================================================
# test_stradaB_10s.slurm — small-sample proof of "Strada B":
#   self-consistent reconstruction-FAD on chunks_mix_original (10s),
#   computing OUR OWN MERT reference (framewise) instead of the
#   colleague's pre-computed stats. Expected: fad_mert back to a
#   NORMAL scale (~0.1-0.5) instead of ~53.
#
#   sbatch --export=NONE evaluation/test_stradaB_10s.slurm                       # defaults: LARGE2, N=80
#   sbatch --export=NONE evaluation/test_stradaB_10s.slurm checkpoints/sa2Disc-step=500000.ckpt 80
# ============================================================
#SBATCH --job-name=stradaB_10s
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stradaB_10s_%j.out
#SBATCH --error=logs/stradaB_10s_%j.err

set -uo pipefail
mkdir -p logs

CKPT="${1:-checkpoints/LARGE2_mrmel05_e99.ckpt}"
N="${2:-80}"

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

DATA_DIR="$FAST/datasets/moisesdb/chunks_mix_original/original"
EVAL_ROOT="$WORK/C-VAE/runs/eval_swin_10s_stradaB_test"
mkdir -p "$EVAL_ROOT"

echo "============================================================"
echo "STRADA B TEST — ckpt=$CKPT  N=$N files"
echo "data: $DATA_DIR"
echo "============================================================"

echo "[$(date)] STEP 1/2 — recompute OUR MERT reference (framewise) on $N files..."
python -u evaluation/recompute_ref_stats_10s.py \
    --data-dir "$DATA_DIR" \
    --device   cuda \
    --max-files "$N" \
    --save-embeddings

echo "[$(date)] STEP 2/2 — eval swin on the SAME $N files (auto-uses stats_ours/)..."
python -u evaluation/evaluate_swin_10s.py \
    --checkpoint "$CKPT" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$EVAL_ROOT" \
    --device     cuda \
    --num-workers 8 \
    --max-files  "$N"

CK=$(basename "${CKPT%.ckpt}")
echo "============================================================"
echo "RESULT — Strada B fad_mert (OUR self-consistent reference):"
cat "$EVAL_ROOT/$CK/metrics/fad_mert.csv" 2>/dev/null
echo "(confronto: con le stats del collega usciva ~53)"
echo "============================================================"
echo "[$(date)] DONE"
