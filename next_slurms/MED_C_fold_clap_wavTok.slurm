#!/bin/bash
# ============================================================
# MED_C_fold_clap_wavTok: LARGE1 recipe (mrmel default 0.3) on FMA-MEDIUM,
#                         ×64 with LEARNED freq-fold latent (C).
#   - Derived VERBATIM from LARGE1_clap_wavTok. Changes ONLY (inline overrides
#     on top of swin_real_baseline; NO extra yaml):
#       * latent_ch=64, encoder/decoder fold_freq_to_channels=true
#         (PS=[64,1], depths=[2,6,2] unchanged) -> native 1-D latent
#         (64, 32) = 2048 scalars = ×64
#       * data.fma_subset=medium       (19,922 train / 2,505 val)
#       * epochs=97                     (~120k steps on medium, standard O/N-series)
#       * --time=23:00:00
#   - ⚠️ REQUIRES NEW CODE: `fold_freq_to_channels` flag in SwinEncoder/SwinDecoder.
#     Do NOT submit until the code change is implemented + tested.
#   - Everything else identical to LARGE1 (CLAP distill + WavTokenizer rpgan disc,
#     kl=1e-4, mrmel default 0.3, lr=1e-3, disc_lr=2e-4, adv=0.1, fm=0.2, FP32, 2 GPU).
# ============================================================
#SBATCH --job-name=MED_C_fold
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP (kept = LARGE1 for step-count comparability)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=23:00:00
#SBATCH --requeue                      # node-failure insurance: SLURM re-queues, auto-resume from last.ckpt
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
export HF_HUB_OFFLINE=1

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

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MED_C_fold_clap_wavTok"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MED_C_fold_clap_wavTok"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        +experiment=O4b_real_distill_mert \
        data.fma_subset=medium \
        models.model.latent_channels=64 \
        "++models.model.encoder.fold_freq_to_channels=true" \
        "++models.model.decoder.fold_freq_to_channels=true" \
        "++trainer.loss_config.semantic_distill.teacher_type=clap" \
        "++trainer.loss_config.semantic_distill.config.proj_dim=512" \
        "++trainer.loss_config.semantic_distill.detach_warmup_steps=25000" \
        trainer.trainer.epochs=97 \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=10 \
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
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/MED_C_fold_clap_wavTok \
        trainer.wandb.name=MED_C_fold_clap_wavTok \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MED_C_fold_clap_wavTok"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
