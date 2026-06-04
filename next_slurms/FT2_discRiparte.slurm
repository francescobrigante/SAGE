#!/bin/bash
# ============================================================
# FT2_discRiparte: DECODER-FINETUNE from N6 — disc FRESH (random init)
#   Ablation: "full train WITH disc (= N6) → decoder-finetune WITH disc from scratch".
#   - Inits autoencoder weights (EMA) from N6's last checkpoint.
#   - Discriminator is NOT loaded from checkpoint → random init (fresh).
#   - Encoder FROZEN (freeze_encoder=true) → only the decoder trains.
#   - SAO Encodec Discriminator, identical architecture to N6 (filters=64, adv=0.1, fm=5.0).
#   - gen_lr=1e-4: decoder pretrained (N6) → finetune at ~10x below N6's full-train 1e-3.
#   - disc_lr=3e-4: discriminator is fresh (random init) → full SAO default LR.
#   - kl=0 (encoder frozen → KL constant). FP32.
#   - warmup_steps=0: matching FT1 for comparability.
#   Distinct from N6 (full train) and FT1 (from M5): own run dir / ckpt_dir / W&B id.
# ============================================================
#SBATCH --job-name=FT2_discRiparte
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-08:00:00
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

# ─── Source weights: N6's last checkpoint (full train WITH disc) ─────────────
N6_CKPT="$WORK/C-VAE/runs/N6_real_mse_sao_disc/2026-06-03_11-27-08/checkpoints/last.ckpt"
if [ ! -f "$N6_CKPT" ]; then
    echo "[FATAL] N6 checkpoint not found: $N6_CKPT" >&2
    exit 1
fi
echo "[INIT] decoder-finetune init from N6 (EMA only, disc fresh): $N6_CKPT"

# ─── Auto-resume: W&B run_id + own last checkpoint ──────────────────────────
# On a fresh start there is no last.ckpt → weights come from +init_from (N6).
# On resume, +ckpt_path is set and train.py IGNORES +init_from (resume wins).
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT2_discRiparte"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/FT2_discRiparte"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        "+init_from=$N6_CKPT" \
        +init_from_ema=true \
        trainer.trainer.freeze_encoder=true \
        trainer.trainer.warmup_steps=0 \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=0.0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.optimizer.lr=1e-4 \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=3e-4" \
        "++trainer.disc_optimizer.betas=[0.8,0.99]" \
        "++trainer.disc_optimizer.weight_decay=1e-3" \
        "++trainer.loss_config.discriminator.type=encodec" \
        "++trainer.loss_config.discriminator.config.filters=64" \
        "++trainer.loss_config.discriminator.config.n_ffts=[2048,1024,512,256,128]" \
        "++trainer.loss_config.discriminator.config.hop_lengths=[512,256,128,64,32]" \
        "++trainer.loss_config.discriminator.config.win_lengths=[2048,1024,512,256,128]" \
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=5.0" \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/FT2_discRiparte \
        trainer.wandb.name=FT2_discRiparte \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/FT2_discRiparte"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
