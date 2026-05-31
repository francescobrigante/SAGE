#!/bin/bash
# ============================================================
# M7_Perceiver_Cplx: Swin V2 Complex VAE with Perceiver Compression
#   - 4-stage Swin V2 Complex VAE
#   - embed_dim=64, depths=[2,2,6,2]
#   - patch_size=[4,1], window_size=8
#   - Perceiver Compression: 512 tokens -> 128 tokens
#   - Precision: 32 (FP32)
# ============================================================
#SBATCH --job-name=M7_Perceiver_Cplx
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                   # 4x A100 DDP
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=18:00:00
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

# ─── Auto-resume (checkpoint only; fresh W&B run each submit) ──────────────────────────────
LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/${SLURM_JOB_NAME}"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop ───────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_cplx_perceiver \
        data.train_dataset.cac=false \
        data.eval_dataset.cac=false \
        data.demo.istft_params.cac=false \
        data.model_channels=2 \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=1e-5 \
        trainer.loss_config.spectral.weights.stft_mse=0.0 \
        trainer.loss_config.spectral.weights.mrstft=1.0 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="bf16-mixed" \
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

    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/${SLURM_JOB_NAME}"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
