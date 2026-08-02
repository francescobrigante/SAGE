#!/bin/bash
# ============================================================
# evaluation/ms_ablation_eval_e99.slurm — STEREO M/S ablation eval.
# MULTICORPUS_swiglu_ms_{replace,add} @ epoch 99 on the FULL MoisesDB-10s set
# (chunks_mix_original/original, 1998 WAVs), ALL metrics + --compute-ms-metrics
# (width_bias, d_width, sisdr_s, sisdr_m). Sequential on 1 GPU (~2h/ckpt).
# Writes runs/ms_ablation_eval/recon_moisesdb_10s/MC_swiglu_ms_{replace,add}_e99/metrics/
# → read by the "STEREO M/S ABLATION" section of results_viz.ipynb.
# Checkpoint symlinks (tag = stem): checkpoints/MC_swiglu_ms_{replace,add}_e99.ckpt.
# QOS normal: the two 16-GPU disc-arm trainings saturate the IscrC_MM lprod
# per-account CPU cap (256), so eval must ride the normal QOS.
#
#   sbatch evaluation/ms_ablation_eval_e99.slurm
# ============================================================
#SBATCH --job-name=ms_abl_eval_swiglu_e99
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

module load profile/deeplrn cineca-ai/4.3.0
# Compute on IscrC_MM, but repo/venv/datasets live in IscrC_AHNetBio ($WORK/$FAST
# env now point to IscrC_MM after the budget switch → pin them explicitly).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login05:3150'
export https_proxy='http://login05:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

DATA_DIR=$FAST/datasets/moisesdb/chunks_mix_original/original
EVAL_ROOT=$WORK/C-VAE/runs/ms_ablation_eval/recon_moisesdb_10s
mkdir -p "$EVAL_ROOT"

for TAG in MC_swiglu_ms_replace_e99 MC_swiglu_ms_add_e99; do
    echo "[$(date)] M/S ablation eval — $TAG — out=$EVAL_ROOT/$TAG"
    python -u evaluation/evaluate_swin_10s.py \
        --checkpoint  checkpoints/$TAG.ckpt \
        --data-dir    "$DATA_DIR" \
        --output-dir  "$EVAL_ROOT" \
        --device      cuda \
        --num-workers 8 \
        --compute-ms-metrics \
        --resume \
        "$@"
done

echo "[$(date)] DONE → $EVAL_ROOT/MC_swiglu_ms_{replace,add}_e99/metrics/"
