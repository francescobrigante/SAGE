#!/bin/bash
# =============================================================================
# Any evaluation script on a SLURM cluster: `python -m evaluation.<script> <Hydra overrides>`.
#
#   # reconstruction on FMA, sharded over 4 GPUs (the final merge runs automatically)
#   sbatch -A <account> -p <partition> --ntasks-per-node=4 --gres=gpu:4 scripts/slurm/eval.sbatch \
#       reconstruction dataset=fma model=sage
#   # references of a clip set, then its reconstruction (one GPU)
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch build_references dataset=musiccaps
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch reconstruction dataset=musiccaps model=same-s
#   # semantic probing (one GPU)
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch maeb encoder=sage
#
# Paths (data, weights, outputs) come from configs/paths/: export its environment
# variables or pass `paths=local`. Activate the environment (`uv sync --extra eval`,
# plus `--extra baselines` for the baselines) before sbatch. A reconstruction run
# killed by the time limit resumes when the same command is resubmitted.
# =============================================================================
#SBATCH --job-name=sage-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one task per GPU; only reconstruction shards over tasks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

SCRIPT="${1:?usage: sbatch scripts/slurm/eval.sbatch <reconstruction|build_references|maeb> [overrides]}"
shift
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1

srun python -m "evaluation.${SCRIPT}" "$@"
if [ "${SCRIPT}" = "reconstruction" ]; then
    python -m evaluation.reconstruction "$@" merge=true   # final CSVs and FADs over all shards (idempotent)
fi
