#!/bin/bash
# ============================================================
# eval_sota_full.slurm — full FMA-large test eval for ONE SOTA baseline,
# file-sharded across the 2 GPUs of one node (srun, 2 tasks), then merged.
# Same shared eval cache as Swin LARGE1 => FAD directly comparable.
# Est. on 2 GPUs: codicodec ~7h, music2latent ~11h, SAME ~32h -> set --time per model:
#   sbatch --time=08:00:00 --export=ALL,MODEL=codicodec    eval_sota_full.slurm
#   sbatch --time=14:00:00 --export=ALL,MODEL=music2latent eval_sota_full.slurm
#   sbatch --time=40:00:00 --export=ALL,MODEL=same         eval_sota_full.slurm
# Override MAXF for a quick shard test:
#   sbatch --qos=boost_qos_dbg --time=00:20:00 \
#          --export=ALL,MODEL=codicodec,MAXF=12,EVAL_ROOT=$WORK/C-VAE/runs/eval_sota_shardtest \
#          eval_sota_full.slurm
# ============================================================
#SBATCH --job-name=eval_sota_full
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

: "${MODEL:?set MODEL=<adapter key>}"
: "${MAXF:=0}"                                       # 0 => full FMA-large test (11,263)
: "${EVAL_ROOT:=$WORK/C-VAE/runs/eval_sota_large}"

module load profile/deeplrn cineca-ai/4.3.0
unset PYTHONPATH                                     # isolate the sota venv
source $WORK/.envs/sota/bin/activate

export HF_HOME=$FAST/hf_cache
export HF_HUB_OFFLINE=1                              # all weights already cached
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE
FMA=/leonardo_scratch/fast/IscrC_AHNetBio/datasets/fma
CACHE_DIR="$SCRATCH/eval_cache"                      # SHARED with Swin LARGE1 / SAO
mkdir -p "$EVAL_ROOT" "$CACHE_DIR"

COMMON=(
    --model        "$MODEL"
    --target-dir   "$FMA/fma_large"
    --fma-csv-path "$FMA/fma_metadata/tracks.csv"
    --output-dir   "$EVAL_ROOT"
    --cache-dir    "$CACHE_DIR"
    --num-workers  6
    --max-files    "$MAXF"
)

echo "[$(date)] eval_sota_full — MODEL=$MODEL MAXF=$MAXF EVAL_ROOT=$EVAL_ROOT (2-GPU shard)"

# Stage 1: 2 ranks each process files[rank::2] on cuda:LOCALID -> metrics/parts/
srun python -u evaluation/sota_models/evaluate_sota.py "${COMMON[@]}"

# Stage 2: single-process merge of partials -> final CSVs + FAD + _done
python -u evaluation/sota_models/evaluate_sota.py "${COMMON[@]}" --merge

echo "[$(date)] DONE -> $EVAL_ROOT/$MODEL/metrics/"
