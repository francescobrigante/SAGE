#!/bin/bash
# ============================================================
# final_eval/recon_fma_sota_chain.slurm — FMA-large test recon (full 11 263) for
# ONE SOTA baseline as a CHAIN of 4 h 1-GPU jobs instead of one long 2-GPU job
# (starts much sooner in the queue). Same evaluator, cache and output root as
# eval_sota_full.slurm (FINAL_EVAL.md), so results are identical in kind.
# SAME-S runs at ~2.55 s/file on 1 GPU → 11 263 files ≈ 8 h ≈ 2-3 links.
#
# How it resumes: evaluate_sota.py appends every metric per file to
# metrics/parts/*.{rank}.csv and marks the stem in parts/done.{rank}.txt; a relaunch
# skips every stem listed in ANY done.*.txt, and --merge dedups by file. Each link
# runs the evaluator under `timeout` (RUN_MIN), then:
#   exit 0   → every file done → --merge (final CSVs + FAD + _done), chain ends
#   exit 124 → time slice used up → submits the next link (CHAIN+1)
#   other    → error → chain stops (inspect the log, then resubmit by hand)
#
#   sbatch --export=NONE evaluation/final_eval/recon_fma_sota_chain.slurm same-s
#   sbatch --export=NONE --account=IscrC_PDR evaluation/final_eval/recon_fma_sota_chain.slurm same-s
# Next links inherit the account of the link that submits them.
#
# Timing caveat: every link reloads the model, so timing.csv gets one warm-up
# outlier per link (median unaffected). SAME/SAME-S also recompile flex_attention
# for each new clip length → FMA timing is not a clean RTF for them; use
# runs/bench_rtf_compile/ for speed.
# ============================================================
#SBATCH --job-name=fe_recon_fma_chain
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

cd "$WORK/C-VAE" || exit 1
mkdir -p logs

MODEL="${1:?Usage: sbatch recon_fma_sota_chain.slurm <adapter key>}"
: "${CHAIN:=1}"                       # link number (set by the previous link)
: "${MAX_CHAIN:=20}"                  # safety cap on links
: "${RUN_MIN:=225}"                   # evaluator time slice of the 4 h link; the rest is merge headroom
: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_fma}"
SELF=$WORK/C-VAE/evaluation/final_eval/recon_fma_sota_chain.slurm

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH
source $WORK/.envs/sota/bin/activate

export HF_HOME=$FAST/hf_cache
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FMA=$FAST/datasets/fma
CACHE_DIR=/leonardo_scratch/large/userexternal/fbrigant/eval_cache   # shared with SAGE / other SOTA
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

COMMON=(
    --model        "$MODEL"
    --target-dir   "$FMA/fma_large"
    --fma-csv-path "$FMA/fma_metadata/tracks.csv"
    --output-dir   "$EVAL_ROOT"
    --cache-dir    "$CACHE_DIR"
    --num-workers  6
    --max-files    0
)

echo "[$(date)] FMA SOTA chain — MODEL=$MODEL — link $CHAIN/$MAX_CHAIN — slice ${RUN_MIN}m — out=$EVAL_ROOT"

# Recovery: the inline merge (world==1) runs under the slice timeout; if the FADs
# don't fit, `touch metrics/MERGE_ONLY` → the next link skips the evaluator, runs
# only --merge (no timeout, no resubmit) and removes the marker on success.
MERGE_ONLY_FLAG="$EVAL_ROOT/$MODEL/metrics/MERGE_ONLY"
if [ -f "$MERGE_ONLY_FLAG" ]; then
    echo "[$(date)] MERGE_ONLY marker found → merge only"
    python -u evaluation/sota_models/evaluate_sota.py "${COMMON[@]}" --merge || exit $?
    rm -f "$MERGE_ONLY_FLAG"
    echo "[$(date)] DONE → $EVAL_ROOT/$MODEL/metrics/"
    exit 0
fi

timeout -k 60 "${RUN_MIN}m" python -u evaluation/sota_models/evaluate_sota.py "${COMMON[@]}"
RC=$?

case $RC in
  0)
    echo "[$(date)] all files done → merge"
    python -u evaluation/sota_models/evaluate_sota.py "${COMMON[@]}" --merge
    echo "[$(date)] DONE → $EVAL_ROOT/$MODEL/metrics/" ;;
  124|137)
    if [ "$CHAIN" -ge "$MAX_CHAIN" ]; then
        echo "[FATAL] MAX_CHAIN=$MAX_CHAIN reached, not resubmitting" >&2; exit 1
    fi
    NEXT=$(sbatch --parsable --account="$SLURM_JOB_ACCOUNT" \
           --export=NONE,CHAIN=$((CHAIN + 1)),MAX_CHAIN=$MAX_CHAIN,RUN_MIN=$RUN_MIN,EVAL_ROOT=$EVAL_ROOT \
           "$SELF" "$MODEL")
    echo "[$(date)] slice over → next link $((CHAIN + 1)) = job $NEXT" ;;
  *)
    echo "[FATAL] evaluator exit $RC — chain stopped" >&2; exit $RC ;;
esac
