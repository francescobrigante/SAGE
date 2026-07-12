#!/bin/bash
# ============================================================
# eval_ft_internal_10s.slurm — INTERNAL model-selection eval for the
# FT_LARGE2_mr05_decFT_postnet finetune, on MoisesDB chunks_mix_original (10s).
#
# Same idea as eval_ft_internal.slurm BUT on the colleague's 10s set
# (chunks_mix_original/original, 1998 WAVs) with the PRE-COMPUTED reference
# embeddings + Fréchet stats (embeddings/ + stats_ours/). Because it uses the
# SAME data + SAME ref-stats as evaluate_sota_10s.py, these numbers are
# DIRECTLY COMPARABLE to the SOTA results in runs/eval_sota_10s/* — no subset,
# no --max-files. Output lands in runs/eval_swin_10s/ next to LARGE2_e99 and
# the SOTA models, so it shows up in results_viz cell 9 with one extra row.
#
# Differences vs eval_ft_internal.slurm:
#   * script  : evaluate_swin_10s.py (single --checkpoint) → we LOOP, one call
#               per epoch (the 10s script does NOT take a checkpoint list).
#   * data    : --data-dir chunks_mix_original/original  (FULL 1998 files).
#   * no       --max-files / --target-dir / --fma-csv-path / --cache-dir.
#   * output  : runs/eval_swin_10s/<stem>/metrics/  (shared with the baseline).
#
# Usage:
#   sbatch evaluation/eval_ft_internal_10s.slurm [EPOCH ...]   # default: 199
#   sbatch evaluation/eval_ft_internal_10s.slurm 179 199
#   sbatch evaluation/eval_ft_internal_10s.slurm base 199      # +LARGE2 baseline
#   SRC_RUN=cooldown sbatch evaluation/eval_ft_internal_10s.slurm 29 39 49  # cooldown run
#
# Robust to requeue: globs epoch_<EP>.ckpt across ALL timestamped run dirs and
# symlinks it to checkpoints/FT_LARGE2_mr05_e<EP>.ckpt (same stems as the FMA
# internal eval — harmless to reuse). --resume skips checkpoints already done.
# NOTE: ~1.5-2h per checkpoint on the full set → keep ≤3 epochs per job.
# ============================================================
#SBATCH --job-name=eval_ft_internal_10s
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

DATA_DIR="$FAST/datasets/moisesdb/chunks_mix_original/original"
EVAL_ROOT="$WORK/C-VAE/runs/eval_swin_10s"     # shared with baseline + SOTA-comparable
mkdir -p "$EVAL_ROOT"

# SRC_RUN selects which run's checkpoints to evaluate (default = finetune).
#   SRC_RUN=cooldown  → evaluate the WSD cooldown run instead.
# Both write to the SAME top-level EVAL_ROOT (runs/eval_swin_10s/<stem>/) so the
# results land next to the SOTA + FT entries with NO manual move needed.
SRC_RUN="${SRC_RUN:-finetune}"
case "$SRC_RUN" in
    cooldown)
        RUN_BASE="$WORK/C-VAE/runs/COOLDOWN_LARGE2_mr05"
        STEM_PREFIX="COOLDOWN_LARGE2_mr05_e" ;;
    finetune|*)
        RUN_BASE="$WORK/C-VAE/runs/FT_LARGE2_mr05_decFT_postnet"
        STEM_PREFIX="FT_LARGE2_mr05_e" ;;
esac
echo "[SRC] SRC_RUN=$SRC_RUN → RUN_BASE=$RUN_BASE (stem prefix: ${STEM_PREFIX})"

EPOCHS=("$@")
[ ${#EPOCHS[@]} -eq 0 ] && EPOCHS=(199)

# ─── Resolve each epoch checkpoint + symlink into checkpoints/ ────────────────
CKPTS=()
for E in "${EPOCHS[@]}"; do
    if [ "$E" = "base" ] || [ "$E" = "large2" ]; then
        LINK="$WORK/C-VAE/checkpoints/LARGE2_mrmel05_e99.ckpt"
        if [ ! -e "$LINK" ]; then
            echo "[WARN] LARGE2 baseline ckpt not found: $LINK — skipping" >&2
            continue
        fi
        echo "[BASE] $LINK (LARGE2 start → stem LARGE2_mrmel05_e99)"
        CKPTS+=("$LINK")
        continue
    fi
    EP=$(printf "%03d" "$E")
    SRC=$(ls -t "$RUN_BASE"/*/checkpoints/epoch_${EP}.ckpt 2>/dev/null | head -1)
    if [ -z "$SRC" ]; then
        echo "[WARN] epoch_${EP}.ckpt not found under $RUN_BASE/*/checkpoints/ — skipping" >&2
        continue
    fi
    LINK="$WORK/C-VAE/checkpoints/${STEM_PREFIX}${E}.ckpt"
    ln -sf "$SRC" "$LINK"
    echo "[LINK] $LINK -> $SRC"
    CKPTS+=("$LINK")
done

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[FATAL] no checkpoints resolved; nothing to evaluate" >&2
    exit 1
fi

echo "============================================================"
echo "[$(date)] INTERNAL 10s recon eval — FULL set (no max-files)"
echo "data:        $DATA_DIR"
echo "checkpoints: ${CKPTS[*]}"
echo "output:      $EVAL_ROOT"
echo "============================================================"

# ─── One call per checkpoint (evaluate_swin_10s.py takes a single ckpt) ───────
rc=0
for CKPT in "${CKPTS[@]}"; do
    echo "------------------------------------------------------------"
    echo "[$(date '+%T')] eval → $(basename "$CKPT")"
    python -u evaluation/evaluate_swin_10s.py \
        --checkpoint  "$CKPT" \
        --data-dir    "$DATA_DIR" \
        --output-dir  "$EVAL_ROOT" \
        --device      cuda \
        --num-workers 8 \
        --resume || rc=$?
done

echo "============================================================"
echo "[$(date)] DONE → $EVAL_ROOT/<stem>/metrics/  (rc=$rc)"
echo "============================================================"
exit $rc
