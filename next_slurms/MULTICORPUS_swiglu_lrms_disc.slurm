#!/bin/bash
# ============================================================
# MULTICORPUS_swiglu_lrms_disc: STEREO-COLLAPSE pretrain arm "STEREO-AWARE DISC" on the
# SAGE-paper architecture (swin_real_swiglu_xsa). IDENTICAL to
# MULTICORPUS_swiglu_500ep in EVERY parameter (recipe, inv_gamma=200000,
# save_every=20, 16 GPU, global batch 128) EXCEPT:
#   * epochs 500 → 300  (2-day budget)
#   * --time 3-12:00:00 → 2-00:00:00
#   * recon loss UNCHANGED; discriminator.config.fold_lrms=true (critics see L,R,M,S)
#   * distinct namespace / MASTER_PORT
# Infra fixes carried over from the working stereo slurms (mandatory, not params):
# proxy on login05 (login07 is down) and WORK/FAST pinned to IscrC_AHNetBio (the
# env vars now point to IscrC_MM after the budget switch → repo/venv/data live in
# AHNetBio). Spec: docs/superpowers/specs/2026-07-26-stereo-collapse-sao-loss-design.md
# NOTE QOS: 16 GPU each — two arms saturate the IscrC_MM lprod cap (32 GPU).
# ============================================================
#SBATCH --job-name=MULTICORPUS_swiglu_lrms_disc
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=2-00:00:00             # 300 ep @ 16 GPU ≈ 35h; billed on actual time. + auto-resume.
#SBATCH --requeue                     # node-failure insurance + auto-resume from last.ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

# Compute on IscrC_MM, but repo/venv/datasets live in IscrC_AHNetBio ($WORK/$FAST
# env now point to IscrC_MM after the budget switch → pin them so all $WORK/$FAST
# references below resolve correctly).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

# Multi-node rendezvous: Lightning's SLURMEnvironment derives MASTER_ADDR from the
# node list; pin a MASTER_PORT (distinct from the other multicorpus jobs).
export MASTER_PORT=29584

export http_proxy='http://login05:3150'
export https_proxy='http://login05:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_swiglu_lrms_disc"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_swiglu_lrms_disc"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_swiglu \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        trainer.trainer.epochs=300 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=20 \
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_swiglu_lrms_disc \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_swiglu_lrms_disc \
        $CKPT_ARG

    EXIT=$?
    [ $EXIT -eq 0 ] && break

    echo "[$(date '+%T')] Attempt $attempt failed (exit $EXIT), retrying in 15s..."
    sleep 15
    attempt=$((attempt + 1))

    if [ -f "$RUN_ID_FILE" ]; then
        export WANDB_RESUME=must
        export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    fi
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_swiglu_lrms_disc"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
