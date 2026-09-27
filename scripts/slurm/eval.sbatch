#!/bin/bash
# =============================================================================
# Any evaluation script on a SLURM cluster: `python -m evaluation.<script> <args>`.
#
#   # reconstruction on FMA, sharded over 4 GPUs (the final merge runs automatically)
#   sbatch -A <account> -p <partition> --ntasks-per-node=4 --gres=gpu:4 scripts/slurm/eval.sbatch \
#       reconstruction --model sage --checkpoint SAGE_FTe992.ckpt --protocol fma \
#       --data-dir <fma_large> --fma-csv <fma_metadata/tracks.csv> --cache-dir <cache> --output-dir runs/recon_fma
#   # references of a clip set, then its reconstruction (one GPU)
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch build_references --data-dir <clips>
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch \
#       reconstruction --model same-s --protocol clips --data-dir <clips> --output-dir runs/recon_clips
#   # semantic probing (one GPU)
#   sbatch -A <account> -p <partition> scripts/slurm/eval.sbatch \
#       maeb --encoder sage --checkpoint SAGE_FTe992.ckpt --with-moisesdb
#
# Activate the environment (`uv sync --extra eval`, plus `--extra baselines` for
# the baselines) before sbatch. A reconstruction run killed by the time limit
# resumes when the same command is resubmitted.
# =============================================================================
#SBATCH --job-name=sage-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one task per GPU; only reconstruction shards over tasks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

SCRIPT="${1:?usage: sbatch scripts/slurm/eval.sbatch <reconstruction|build_references|maeb|tables> [args]}"
shift
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

srun python -m "evaluation.${SCRIPT}" "$@"
if [ "${SCRIPT}" = "reconstruction" ]; then
    python -m evaluation.reconstruction "$@" --merge   # final CSVs and FADs over all shards (idempotent)
fi
