#!/bin/bash
# ============================================================
# L6_ps32x1_4s_e128_mse: Swin 4-stage VAE + stft_MSE + PS=32x1
#   IDENTICO a L3_ps64x1_3s_e256_mse, ma con modifiche geometriche per 4 stadi:
#     stft_mse: 1.0
#     mrstft:   0.0
#     mrmel:    0.0  (senza mrmel)
#     patch_size: [32, 1]
#     window_size: [8, 32]
#     embed_dim: 128
#     latent_channels: 16  (per ×64 compression rate su 4-stage con 64 tokens)
#
#   Tutto il resto invariato (kl=1e-5, lr=1e-3).
#
# COSA GUARDARE:
#   Effetto del patch size 32x1 su 4 stadi con stft_MSE.
#
# ARCH (swin_cplx_4s_x64): PS=32x1, window=[8,32], embed=128, depths=[2,2,6,2]
#   Stage 0: 32x128 -> Stage 1: 16x64 -> Stage 2: 8x32 -> Stage 3: 4x16 (64 tokens, x64 complex con 16 ch)
# DATA: FMA-medium (default), no disc.
# ============================================================
#SBATCH --job-name=L6_ps32x1_4s_e128_mse
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/L6_ps32x1_4s_e128_mse"

# Override hardcoded WANDB vars with persisted run_id if available
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# Auto-detect newest last.ckpt
LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/L6_ps32x1_4s_e128_mse"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        models=swin_cplx_baseline \
        "models.model.encoder.depths=[2,2,6,2]" \
        "models.model.encoder.num_heads=[4,8,16,32]" \
        "models.model.decoder.depths=[2,6,2,2]" \
        "models.model.decoder.num_heads=[32,16,8,4]" \
        "models.model.swin.patch_size=[32,1]" \
        "models.model.swin.window_size=[8,32]" \
        "models.model.swin.embed_dim=128" \
        models.model.latent_channels=16 \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=1e-5 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.0 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="bf16-mixed" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/L6_ps32x1_4s_e128_mse \
        trainer.wandb.name=L6_ps32x1_4s_e128_mse \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/L6_ps32x1_4s_e128_mse"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
