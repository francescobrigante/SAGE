#!/bin/bash
# ============================================================
# COOLDOWN_LARGE2_mr05: LR COOLDOWN tail of FT_LARGE2 (WSD-style) — LARGE lineage
#   LARGE2_mrmel05 (FMA-large pretrain, 100 ep) → FT_LARGE2_mr05_decFT_postnet
#   (decoder-FT, 340 ep, best @ epoch_299) → THIS cooldown brings LR → 0.
#
#   Same recipe as FT8_FT5_cooldown but re-pointed to the LARGE2 line:
#   - Inits LIVE weights (init_from_ema=false: EMA is for eval; training is the
#     natural continuation of the LIVE trajectory. With LR→0 the live weights
#     settle and the cooldown's own EMA re-converges to them) + TRAINED disc
#     (init_from_disc=true: epoch_299 already carries the trained WavTokenizer
#     disc; a fresh disc at decaying LR would only unbalance the adv game).
#   - Encoder FROZEN, ResidualPostNet ON (use_postnet=true — epoch_299 has its
#     weights). kl=0. FP32. mrmel=0.5 + ema_decay=0.9998 (LARGE2 distinctive).
#   - gen lr=4.5e-5 (= FT_LARGE2's LR at epoch 299), disc lr=9e-5
#     (2:1 ratio preserved), BOTH cosine-decay to min_lr=1e-6.
#   - 250 sched-step (=500 batches) linear warmup: Adam moments restart from
#     zero (init_from drops optimizer state) → tiny ramp up to the FT-left LR
#     avoids the restart kick, then cosine down.
#   - epochs=50 (~17% of the effective 300-ep finetune (best @ e299), ~12.5% of
#     pretrain+finetune → WSD sweet-spot). STEPS_PER_EPOCH is FMA-large; total_steps is in SCHEDULER-step
#     units (= batches/2 with G/D 1:1 alternation) and is derived from EPOCHS
#     below — change EPOCHS only, the schedule follows.
#
#   ⚠ G/D-alternation bug (verified): each scheduler steps once every ~2 batches,
#   so TOTAL_SCHED_STEPS MUST be total_batches/2 — otherwise the cosine covers
#   only half its path (the medium cooldown job 45403541 used total_steps=25000
#   = batch count and ended at gen_lr=4.21e-5 instead of 1e-6). Here we halve it.
#
#   sbatch next_slurms/COOLDOWN_LARGE2_mr05.slurm
# ============================================================
#SBATCH --job-name=COOLDOWN_LARGE2_mr05
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=36:00:00
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
export HF_HUB_OFFLINE=1                   # no teacher/CLAP in decoder-FT → stay offline

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

# ─── Source weights: FT_LARGE2 epoch_299 (live + trained disc + postnet) ──────
FT_CKPT="$WORK/C-VAE/runs/FT_LARGE2_mr05_decFT_postnet/2026-06-27_13-48-00/checkpoints/epoch_299.ckpt"
if [ ! -f "$FT_CKPT" ]; then
    echo "[FATAL] FT_LARGE2 source checkpoint not found: $FT_CKPT" >&2
    exit 1
fi
echo "[INIT] cooldown init from FT_LARGE2 epoch_299 (live weights + disc): $FT_CKPT"

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/COOLDOWN_LARGE2_mr05"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/COOLDOWN_LARGE2_mr05"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Cosine schedule length (SCHEDULER-step units) ──────────────────────────
# G/D alternate 1:1 → each scheduler steps ~1 every 2 batches, so total_steps
# must be total_batches/2 (see header bug note), else the cosine only reaches
# half its path by end of run.
EPOCHS=50
STEPS_PER_EPOCH=5263   # FMA-large train @ global BS=16, drop_last (observed in FT_LARGE2)
TOTAL_SCHED_STEPS=$(( EPOCHS * STEPS_PER_EPOCH / 2 ))   # 131575 sched steps = 263150 batches
WARMUP_SCHED_STEPS=250                                  # = 500 batches of linear warmup

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_real_baseline \
        models.model.decoder.use_postnet=true \
        data.fma_subset=large \
        "+init_from=$FT_CKPT" \
        +init_from_ema=false \
        +init_from_disc=true \
        trainer.trainer.freeze_encoder=true \
        trainer.trainer.warmup_steps=0 \
        trainer.trainer.epochs=$EPOCHS \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=10 \
        trainer.loss_config.bottleneck.weights.kl=0.0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.5 \
        trainer.optimizer.lr=4.5e-5 \
        "trainer.scheduler._target_=ar_spectra.training.schedulers.CosineLRWithWarmup" \
        "~trainer.scheduler.inv_gamma" \
        "~trainer.scheduler.power" \
        "~trainer.scheduler.warmup" \
        "++trainer.scheduler.total_steps=$TOTAL_SCHED_STEPS" \
        "++trainer.scheduler.warmup_steps=$WARMUP_SCHED_STEPS" \
        "++trainer.scheduler.min_lr=1e-6" \
        "++trainer.scheduler.warmup_lr_init=1e-6" \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=9e-5" \
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
        trainer.trainer.ckpt_dir=checkpoints/COOLDOWN_LARGE2_mr05 \
        trainer.wandb.name=COOLDOWN_LARGE2_mr05 \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/COOLDOWN_LARGE2_mr05"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
