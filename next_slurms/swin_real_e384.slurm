#!/bin/bash
# ============================================================
# Swin_Real_e384: Real-valued Swin V2 VAE with embed_dim=384
#   - 3-stage Swin V2 Real VAE
#   - embed_dim=384, depths=[2,6,2]
#   - patch_size=[64,1], window_size=[4,32]
#   - num_heads=[8,16,32] enc / [32,16,8] dec (head_dim=48)
#   - Precision: 32 (FP32) to prevent numerical explosion under bf16
# ============================================================
#SBATCH --job-name=Swin_Real_e384
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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/${SLURM_JOB_NAME}"

# Override hardcoded WANDB vars with persisted run_id if available
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# Auto-detect newest last.ckpt (finds pre-crash checkpoint on first launch too)
LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/${SLURM_JOB_NAME}"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        models=swin_real_baseline \
        "models.model.swin.embed_dim=384" \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=1e-5 \
        trainer.loss_config.spectral.weights.stft_mse=0.0 \
        trainer.loss_config.spectral.weights.mrstft=1.0 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/${SLURM_JOB_NAME} \
        trainer.wandb.name=${SLURM_JOB_NAME} \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/${SLURM_JOB_NAME}"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
