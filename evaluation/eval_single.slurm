#!/bin/bash
# ===============================================================================
# eval_single.slurm
# Evaluate a single checkpoint. Launch multiple jobs in parallel:
#   sbatch evaluation/eval_single.slurm checkpoints/run_A/epoch=10.ckpt
#   sbatch evaluation/eval_single.slurm checkpoints/run_B/epoch=20.ckpt
# ===============================================================================
#SBATCH --job-name=eval_%x
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err

CHECKPOINT_PATH="$1"

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: no checkpoint path provided."
    echo "Usage: sbatch evaluation/eval_single.slurm <path/to/checkpoint.ckpt>"
    exit 1
fi

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

cd $WORK/C-VAE

TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_small"
EVAL_ROOT="runs/eval_sweep"

mkdir -p logs "$EVAL_ROOT"

# Derive a short job name from the checkpoint stem for the log header
CKPT_STEM=$(basename "$CHECKPOINT_PATH" .ckpt)
echo "============================================================"
echo "EVALUATING CHECKPOINT: $CKPT_STEM"
echo "PATH: $CHECKPOINT_PATH"
echo "============================================================"

python3 evaluation/eval_sweep.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target-dir "$TARGET_DIR" \
    --eval-root "$EVAL_ROOT" \
    --batch-size 16 \
    --device cuda

echo "============================================================"
echo "DONE: $CKPT_STEM"
echo "============================================================"
