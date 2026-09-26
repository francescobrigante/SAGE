#!/bin/bash
# ============================================================
# final_eval/maeb_clap.slurm — FINAL paper eval, CLAP-ORACLE reference row.
# All 19 MAEB tasks for the FROZEN distillation teacher, 1 GPU, same two-pass
# structure and same output root as final_eval/maeb_sota.slurm, so the oracle is
# comparable row by row with runs/final_eval/maeb/maeb_SAGE_e992_tri2/ (which was
# scored with max_files=0 — do NOT pass --max-files on the real run).
#
# This is a REFERENCE, not a baseline: CLAP is not invertible and reconstructs no
# audio, so it never appears on the fidelity axis. Tabulate it with both caveats
# recorded in the summary.json (512-d vs 64-d; AudioSet/captioned-music
# pretraining plausibly covers GTZAN/MusicGenre/NSynth) — the oracle is an
# OPTIMISTIC upper bound on what distillation could have transferred.
#
# Uses the ISOLATED maeb_dl venv (laion_clap 1.1.7 is ALREADY installed there —
# nothing to pip-install, so that venv's torch pin is not at risk).
#
# Walltime 02:30:00 on `normal`, NOT the 30-min debug qos: a from-scratch 19-task
# 2-pass run on the lightest encoder (SAGE, 1 forward/clip) already TIMED OUT at
# 33 min on dbg (job 58360573); the same 2-pass structure on the SOTA codec
# encoders (48302912-48302918, also from scratch) took 46min-1h41m on `normal`.
# CLAP does 3 forwards per 30-s clip (one per 10-s window, see maeb/clap_encoder.py)
# vs 1 for those encoders, so 2:30 gives ~50min of margin over that worst case.
# MAEB resumes per-task from disk, so a timeout here is recoverable regardless.
# `all19` does both passes in ONE job — sequential on purpose, since two
# concurrent jobs would race on the shared summary.json.
#
#   sbatch evaluation/final_eval/maeb_clap.slurm all19
#   # smoke:
#   MAEB_CLAP_OUT=$WORK/C-VAE/runs/final_eval/maeb/_smoke_clap_oracle \
#   sbatch --qos=boost_qos_dbg --time=00:25:00 \
#          evaluation/final_eval/maeb_clap.slurm --tasks FMAGenreClassification --max-files 100
# ============================================================
#SBATCH --job-name=fe_maeb_clap
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

cd "$WORK/C-VAE"
mkdir -p logs

# Isolated venv: do NOT module-load cineca-ai; purge modules and unset PYTHONPATH (CINECA.md §5.4).
module purge || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source $WORK/.envs/maeb_dl/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${MAEB_CLAP_OUT:-$WORK/C-VAE/runs/final_eval/maeb/maeb_clap_oracle}"
EXTRA=("$@")

echo "============================================================"
echo "MAEB CLAP-oracle (frozen teacher) — 1 GPU — out=$OUT_DIR"
echo "extra args: ${EXTRA[*]-}"
echo "============================================================"

run_pass () {
    echo "[$(date '+%T')] --- pass: $* ---"
    python evaluation/maeb_clap.py \
        --output-dir "$OUT_DIR" \
        --device cuda \
        "$@"
}

if [ "${EXTRA[0]-}" = "all19" ]; then
    run_pass --with-moisesdb                 # 6 FMA + 7 MoisesDB
    run_pass --maeb-original-music-only      # 6 upstream MAEB music
else
    run_pass ${EXTRA[@]+"${EXTRA[@]}"}
fi

echo "============================================================"
echo "[$(date)] MAEB CLAP-oracle DONE — results in $OUT_DIR"
echo "============================================================"
