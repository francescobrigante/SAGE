#!/bin/bash
# =====================================================================
# eval_swiglu_ep99: recon (FMA + MoisesDB-10s) + MAEB for
# MULTICORPUS_swiglu_500ep @ epoch_099.
#
# Output:
#   runs/multicorpus_eval/recon_fma/MULTICORPUS_swiglu_ep99/
#   runs/multicorpus_eval/recon_moisesdb_10s/MULTICORPUS_swiglu_ep99/
#   runs/multicorpus_eval/maeb/maeb_MULTICORPUS_swiglu_ep99/
#
# Launch (after watcher exits):
#   W=$(sbatch --parsable next_slurms/watcher_swiglu_ep99.slurm)
#   sbatch --dependency=afterok:$W next_slurms/eval_swiglu_ep99.slurm
# =====================================================================
#SBATCH --job-name=eval_swiglu_ep99
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export HF_HOME=$FAST/hf_cache
export PYTHONUNBUFFERED=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'

set -a; source $WORK/C-VAE/.env; set +a
cd $WORK/C-VAE

TAG="MULTICORPUS_swiglu_ep99"
CKPT_SRC="$WORK/C-VAE/runs/MULTICORPUS_swiglu_500ep/2026-07-03_14-43-41/checkpoints/epoch_099.ckpt"
CKPT_LINK="checkpoints/$TAG.ckpt"
EVAL_ROOT="$WORK/C-VAE/runs/multicorpus_eval"
CACHE_DIR="$SCRATCH/eval_cache"

mkdir -p logs "$CACHE_DIR"
ln -sfn "$CKPT_SRC" "$CKPT_LINK"
echo "[$(date '+%T')] $TAG — ckpt -> $CKPT_SRC"

# ── 1. FMA recon (2-GPU file-shard) ─────────────────────────────────
echo "[$(date '+%T')] Step 1/3: FMA recon"
source $WORK/.envs/c_vae/bin/activate
srun python evaluation/evaluate_swin_varT.py \
    --checkpoint "$CKPT_LINK" \
    --dataset fma \
    --target-dir $FAST/datasets/fma/fma_large \
    --fma-csv-path $FAST/datasets/fma/fma_metadata/tracks.csv \
    --output-dir "$EVAL_ROOT/recon_fma" \
    --cache-dir "$CACHE_DIR" \
    --num-workers 16 --max-files 0 --shard-files --resume
deactivate

# ── 2. MoisesDB-10s recon (1-GPU) ───────────────────────────────────
echo "[$(date '+%T')] Step 2/3: MoisesDB-10s recon"
source $WORK/.envs/c_vae/bin/activate
python evaluation/evaluate_swin_10s.py \
    --checkpoint "$CKPT_LINK" \
    --data-dir $FAST/datasets/moisesdb/chunks_mix_original/original \
    --output-dir "$EVAL_ROOT/recon_moisesdb_10s" \
    --num-workers 8 --resume
deactivate

# ── 3. MAEB — 15 tasks (1-GPU, maeb_dl venv) ────────────────────────
echo "[$(date '+%T')] Step 3/3: MAEB"
source $WORK/.envs/maeb_dl/bin/activate
python evaluation/maeb_swin.py \
    --ckpt "$CKPT_LINK" \
    --with-extra --with-moisesdb --standardize-bottleneck \
    --output-dir "$EVAL_ROOT/maeb/maeb_$TAG" \
    --device cuda
deactivate

echo "[$(date '+%T')] DONE — $TAG"
