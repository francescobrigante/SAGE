#!/bin/bash
# ============================================================
# O3_real_vf_mert: M5 real baseline + VF-Loss (Fase 1, VA-VAE arXiv:2501.01423)
#   - 3-stage Swin V2 Real VAE (swin_real_baseline = M5 baseline)
#   - Reconstruction = M5 baseline: stft_MSE=1.0 (mrstft=0, mrmel=0), kl=1e-5
#   - + semantic_vf (w_hyper=0.1): align latent to MERT per-frame (cosine) +
#     relational (gram), with VA-VAE adaptive grad-norm weighting (engine-side),
#     m1=0.5, m2=0.25, projector D(64)->768, NO detached warmup.
#   - vf_proj trained by opt_aux: AdamW lr=1e-4 CONSTANT (ConstantLR), VA-VAE/MAR.
#   - All experiment config in config/experiment/O3_real_vf_mert.yaml
#     (trainer.yaml NOT modified) → invoked via +experiment=O3_real_vf_mert
#   - Semantic family: measured on MAEB (__std) + reconstruction. vs M5 / N2.
#   - Precision: 32 (FP32). Loads frozen MERT (offline from $FAST) as a submodule.
# ============================================================
#SBATCH --job-name=O3_real_vf_mert
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1                 # MERT loads offline from $FAST on compute nodes

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# ─── Auto-resume: W&B run_id + last checkpoint ──────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/O3_real_vf_mert"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/O3_real_vf_mert"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop (up to 8 attempts — recovers from DataLoader timeouts) ───────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        +experiment=O3_real_vf_mert \
        trainer.trainer.epochs=97 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/O3_real_vf_mert \
        trainer.wandb.name=O3_real_vf_mert \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/O3_real_vf_mert"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
