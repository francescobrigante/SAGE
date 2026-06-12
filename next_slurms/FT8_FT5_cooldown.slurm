#!/bin/bash
# ============================================================
# FT8_FT5_cooldown: LR COOLDOWN tail of FT5 (WSD-style)
#   Rationale: FT5's InverseLR ends at ~0.79x of peak; the cooldown closes the
#   "decay-to-zero" gap (Hagele et al. arXiv:2405.18392; WSD schedule MiniCPM;
#   HiFi-GAN/BigVGAN end training at low LR via built-in exponential decay).
#   - Inits LIVE weights (init_from_ema=false: EMA is for eval, training
#     continues from live) + TRAINED disc (init_from_disc=true: a fresh disc
#     at decaying LR never becomes useful and unbalances the adversarial game).
#   - Encoder FROZEN (cooldown = tail of the decoder-FT phase, papers do all
#     final polishing decoder-only). kl=0. FP32.
#   - gen lr=8e-5 (= effective LR where FT5 ended: 1e-4*(1+120k/200k)^-0.5),
#     disc lr=1.6e-4 (2:1 ratio preserved), BOTH cosine-decay to min_lr=1e-6.
#   - 500-step linear warmup: Adam moments restart from zero (init_from drops
#     optimizer state) -> tiny ramp avoids the restart kick.
#   - epochs=20 (~25k batches on medium, ~21% of FT5's 120k). total_steps is in
#     SCHEDULER-step units (~= batches/2 with G/D alternation) and is derived
#     from EPOCHS below — change EPOCHS only, the schedule follows.
#   - Compare EMA vs FT5 EMA on FAD/CDPAM/CLAP.
# ============================================================
#SBATCH --job-name=FT8_FT5_cooldown
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=10:00:00
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

# ─── Source weights: FT5's last checkpoint (live + trained disc) ────────────
FT5_CKPT=$(ls -t "$WORK/C-VAE/runs/FT5_O8_decFT_wavtok"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ ! -f "$FT5_CKPT" ]; then
    echo "[FATAL] FT5 checkpoint not found under runs/FT5_O8_decFT_wavtok" >&2
    exit 1
fi
echo "[INIT] cooldown init from FT5 (live weights + disc): $FT5_CKPT"

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT8_FT5_cooldown"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/FT8_FT5_cooldown"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Cosine schedule length (SCHEDULER-step units) ──────────────────────────
# Each scheduler is stepped only on its own phase's batches (G/D alternate 1:1),
# so it receives ~1 step every 2 batches. total_steps must therefore be the
# number of SCHEDULER steps = total_batches / 2, otherwise the cosine covers
# only half its path by end of run (job 45403541 ended at gen_lr=4.21e-5
# instead of min_lr=1e-6 for exactly this reason).
EPOCHS=20
STEPS_PER_EPOCH=1244   # FMA-medium train (19908 files) @ global BS=16, drop_last
TOTAL_SCHED_STEPS=$(( EPOCHS * STEPS_PER_EPOCH / 2 ))   # 12440 sched steps = 24880 batches
WARMUP_SCHED_STEPS=250                                  # = 500 batches of linear warmup

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_real_baseline \
        "+init_from=$FT5_CKPT" \
        +init_from_ema=false \
        +init_from_disc=true \
        trainer.trainer.freeze_encoder=true \
        trainer.trainer.warmup_steps=0 \
        trainer.trainer.epochs=$EPOCHS \
        trainer.loss_config.bottleneck.weights.kl=0.0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.optimizer.lr=8e-5 \
        "trainer.scheduler._target_=ar_spectra.training.schedulers.CosineLRWithWarmup" \
        "~trainer.scheduler.inv_gamma" \
        "~trainer.scheduler.power" \
        "~trainer.scheduler.warmup" \
        "++trainer.scheduler.total_steps=$TOTAL_SCHED_STEPS" \
        "++trainer.scheduler.warmup_steps=$WARMUP_SCHED_STEPS" \
        "++trainer.scheduler.min_lr=1e-6" \
        "++trainer.scheduler.warmup_lr_init=1e-6" \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=1.6e-4" \
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
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/FT8_FT5_cooldown \
        trainer.wandb.name=FT8_FT5_cooldown \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/FT8_FT5_cooldown"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
