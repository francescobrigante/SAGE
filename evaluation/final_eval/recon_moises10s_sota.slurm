#!/bin/bash
# ============================================================
# final_eval/recon_moises10s_sota.slurm — FINAL paper eval.
# ONE SOTA baseline reconstruction on the FULL MoisesDB-10s set
# (chunks_mix_original/original, 1998 WAVs, max-files 0), ALL metrics,
# file-sharded across 2 GPUs then merged. Uses the dataset-bundled
# reference stats (stats_ours/ + embeddings/) → FAD identical for every model.
# Writes into runs/final_eval/recon_moisesdb_10s/<model>/metrics/.
#
#   sbatch evaluation/final_eval/recon_moises10s_sota.slurm sao-vae
#   sbatch evaluation/final_eval/recon_moises10s_sota.slurm same
#   # smoke:
#   sbatch --qos=boost_qos_dbg --time=00:20:00 \
#          --export=ALL,EVAL_ROOT=$WORK/C-VAE/runs/final_eval/_smoke/recon_moisesdb_10s \
#          evaluation/final_eval/recon_moises10s_sota.slurm codicodec --max-files 12
# ============================================================
#SBATCH --job-name=fe_recon_m10s_sota
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

MODEL="${1:?Usage: sbatch recon_moises10s_sota.slurm <sao-vae|codicodec|music2latent|same> [extra args]}"
shift
EXTRA=("$@")

module load profile/deeplrn cineca-ai/4.3.0
# Compute on IscrC_MM (IscrC_AHNetBio budget exhausted), but WORK/FAST must point to
# IscrC_AHNetBio where repo, venv, datasets and HF cache live (user-owned, world-readable).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio
unset PYTHONPATH
source $WORK/.envs/sota/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/C-VAE

# DATA_DIR overridable via env (default = mixtures) → same wrapper serves stems too.
: "${DATA_DIR:=$FAST/datasets/moisesdb/chunks_mix_original/original}"
: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_moisesdb_10s}"
mkdir -p "$EVAL_ROOT"

COMMON=(
    --model      "$MODEL"
    --data-dir   "$DATA_DIR"
    --output-dir "$EVAL_ROOT"
    --device     cuda
    --num-workers 8
    --max-files  0
)

echo "[$(date)] FINAL recon MoisesDB-10s — MODEL=$MODEL — 2-GPU shard — out=$EVAL_ROOT"

# Stage 1: 2 ranks each process files[rank::2] on cuda:LOCALID → metrics/parts/
srun python -u evaluation/sota_models/evaluate_sota_10s.py "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"}

# Stage 2: single-process merge of partials → final CSVs + FAD + _done
python -u evaluation/sota_models/evaluate_sota_10s.py "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --merge

echo "[$(date)] DONE → $EVAL_ROOT/$MODEL/metrics/"
