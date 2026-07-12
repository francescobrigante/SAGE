#!/bin/bash
# =====================================================================
# watcher_swiglu_ep99: CPU-only sentinel on lrd_all_serial.
# Polls until MULTICORPUS_swiglu_500ep epoch_099.ckpt appears, then
# exits 0. Chain the eval job with --dependency=afterok:<THIS_ID>.
#
# Launch:
#   W=$(sbatch --parsable next_slurms/watcher_swiglu_ep99.slurm)
#   sbatch --dependency=afterok:$W next_slurms/eval_swiglu_ep99.slurm
# =====================================================================
#SBATCH --job-name=watch_swiglu_ep99
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

CKPT="$WORK/C-VAE/runs/MULTICORPUS_swiglu_500ep/2026-07-03_14-43-41/checkpoints/epoch_099.ckpt"
echo "[$(date '+%T')] Watching: $CKPT"
while [ ! -f "$CKPT" ]; do
    sleep 120
done
echo "[$(date '+%T')] Found epoch_099.ckpt — done."
