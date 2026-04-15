#!/bin/bash
#SBATCH --job-name=eval_sweep
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_sweep_%j.out
#SBATCH --error=logs/eval_sweep_%j.err

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

cd $WORK/C-VAE

# Paths
TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_small"
CHECKPOINTS_DIR="checkpoints"
EVAL_ROOT="runs/eval_sweep"

# Ensure dirs exist
mkdir -p logs "$EVAL_ROOT"

echo "============================================================"
echo "RUNNING FULL EVALUATION SWEEP (ISOLATED CACHE)"
echo "============================================================"
# This will iterate over all checkpoints and use a private cache per-checkpoint
python3 evaluation/eval_sweep.py \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --target-dir "$TARGET_DIR" \
    --eval-root "$EVAL_ROOT" \
    --batch-size 16 \
    --device cuda

echo "============================================================"
echo "SWEEP COMPLETED"
echo "============================================================"
