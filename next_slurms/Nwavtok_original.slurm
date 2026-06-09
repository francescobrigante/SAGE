#!/bin/bash
# ============================================================
# Nwavtok_original: Real baseline (M5) + WavTokenizer-full Discriminator (RpGAN)
#   - 3-stage Swin V2 Real VAE (embed_dim=256, depths=[2,6,2])
#   - stft_mse=1.0, mrstft=0.0
#   - WavTokenizer Discriminator: Vocos MPD + Vocos legacy magnitude MRD + DAC Discriminator
#   - loss_type=rpgan (Relativistic GAN for direct comparison against N7)
#   - preprocess=true (Strictly faithful to paper: per-sample DC-removal and peak-normalization on)
#   - disc_lr=2e-4 (WavTokenizer paper learning rate)
#   - generator_lr=1e-3
#   - adv=0.1, fm=0.2 (Correctly scaled weights to keep the paper's fm/adv = 2.0 ratio under stft_mse=1.0)
#   - Precision 32 (FP32)
# ============================================================
#SBATCH --job-name=Nwavtok_original
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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/Nwavtok_original"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/Nwavtok_original"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        models=swin_real_baseline \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=1e-4 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.optimizer.lr=1e-3 \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=2e-4" \
        "++trainer.disc_optimizer.betas=[0.8,0.99]" \
        "++trainer.disc_optimizer.weight_decay=1e-3" \
        "++trainer.loss_config.discriminator.type=wavtokenizer" \
        "++trainer.loss_config.discriminator.config.loss_type=rpgan" \
        "++trainer.loss_config.discriminator.config.periods=[2,3,5,7,11]" \
        "++trainer.loss_config.discriminator.config.fft_sizes=[2048,1024,512]" \
        "++trainer.loss_config.discriminator.config.preprocess=true" \
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=0.2" \
        data.train_dataloader.batch_size=8 \
        data.eval_dataloader.batch_size=8 \
        trainer.trainer.accumulate_grad_batches=2 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/Nwavtok_original \
        trainer.wandb.name=Nwavtok_original \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/Nwavtok_original"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
