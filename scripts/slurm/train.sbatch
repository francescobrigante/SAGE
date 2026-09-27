#!/bin/bash
# =============================================================================
# SAGE training recipes on a SLURM cluster. The paper used 4 nodes x 4 A100
# (the resources below); the global batch of 128 is split over all GPUs, so
# fewer GPUs run the same recipe with a larger per-GPU batch.
#
#   sbatch -A <account> -p <partition> scripts/slurm/train.sbatch pretrain
#   sbatch -A <account> -p <partition> scripts/slurm/train.sbatch decoder_ft +init_from=<phase-1 checkpoint>
#
# Arguments after the recipe are Hydra overrides. Activate the environment
# (`uv sync --extra train`, then `source .venv/bin/activate`) before sbatch; the
# job inherits it. On a time limit Lightning saves and the job is requeued; to
# resume by hand set RESUME_CKPT=<checkpoint> and resubmit.
# =============================================================================
#SBATCH --job-name=sage-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120           # Lightning checkpoints and requeues before the limit
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

RECIPE="${1:?usage: sbatch scripts/slurm/train.sbatch <pretrain|decoder_ft> [hydra overrides]}"
shift
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

srun python train.py +experiment="${RECIPE}" \
    trainer.trainer.num_gpus="${SLURM_NTASKS_PER_NODE}" ++trainer.trainer.num_nodes="${SLURM_NNODES}" \
    ${RESUME_CKPT:+"+ckpt_path=${RESUME_CKPT}"} "$@"
