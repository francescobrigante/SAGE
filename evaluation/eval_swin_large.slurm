#!/bin/bash
# ============================================================
# eval_swin_large.slurm
# Full FMA-large evaluation — VarT Swin (single forward pass).
# Metrics: SI-SDR, STFT, CLAP cosine, FAD (MERT + CLAP-audio).
# CDPAM skipped at scale (too slow); run debug job for it.
# Use --resume to restart safely after preemption.
# ============================================================
#SBATCH --job-name=eval_O4b_O5_large
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

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

CACHE_DIR="$SCRATCH/eval_cache"
mkdir -p "$CACHE_DIR"

CKPTS=()
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --*)
            if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                EXTRA_ARGS+=("$1" "$2")
                shift 2
            else
                EXTRA_ARGS+=("$1")
                shift 1
            fi
            ;;
        *)
            CKPTS+=("$1")
            shift 1
            ;;
    esac
done

if [ ${#CKPTS[@]} -eq 0 ]; then
    CKPTS=(
        checkpoints/O4b_real_distill_mert.ckpt
        checkpoints/O5_real_chroma_ild.ckpt
    )
fi
echo "Evaluating checkpoints: ${CKPTS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]}"

srun $WORK/.envs/c_vae/bin/python3 evaluation/evaluate_swin_varT.py \
    --checkpoint "${CKPTS[@]}" \
    --target-dir $FAST/datasets/fma/fma_large \
    --output-dir $WORK/C-VAE/runs/eval_swin_large \
    --cache-dir  "$CACHE_DIR" \
    --num-workers 16 \
    --max-files 2500 \
    "${EXTRA_ARGS[@]}" \
    --resume
