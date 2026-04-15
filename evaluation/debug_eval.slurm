#!/bin/bash
#SBATCH --job-name=debug_eval
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/debug_eval_%j.out
#SBATCH --error=logs/debug_eval_%j.err

module load profile/deeplrn cineca-ai/4.3.0
source $WORK/.envs/c_vae/bin/activate

cd $WORK/C-VAE

TARGET_DIR="/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma/fma_small"
PREDS_DIR="runs/eval_full/real64x"
CACHE_DIR="runs/eval_full/real64x/cache_debug"

# Ensure logs dir exists
mkdir -p logs

echo "============================================================"
echo "DEBUGGING CLAP SCORES (AUDIO & MUSIC)"
echo "============================================================"
python3 evaluation/compute_clap_score.py \
    --target-dir $TARGET_DIR \
    --preds-dir $PREDS_DIR \
    --model both \
    --max-files 20 \
    --batch-size 8 \
    --cache-dir $CACHE_DIR
