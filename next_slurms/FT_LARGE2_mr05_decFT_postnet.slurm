#!/bin/bash
# ============================================================
# FT_LARGE2_mr05_decFT_postnet: FINAL decoder-finetune of the large model
#   LARGE2_mrmel05_e99 (FMA-large pretrain) → decoder-finetune with the
#   WavTokenizer discriminator + zero-init ResidualPostNet.
#
#   RECIPE = FT7 (FT5 + postnet), re-pointed to LARGE2 and run on FMA-large
#   with LARGE2's distinctive hyperparameters (mrmel=0.5, ema_decay=0.9998):
#     - init autoencoder weights (EMA) from LARGE2_mrmel05_e99 (strict=False:
#       postnet keys absent in ckpt -> stay zero-init = identity at step 0).
#     - encoder FROZEN; discriminator FROM SCRATCH (init_from_disc=false);
#       gen_lr=1e-4, disc_lr=2e-4; kl=0; FP32; scheduler = default InverseLR
#       (inv_gamma=200000, power=0.5, warmup=0.999) exactly as FT7.
#     - data=large; mrmel=0.5 and ema_decay=0.9998 carried over from LARGE2.
#     - LONG run: epochs=500 (~3.95M steps, ~3x the pretrain) → pick the best
#       checkpoint later, then run the FT8-style cooldown from it.
#
#   GRACEFUL TIMEOUT (method b): #SBATCH --signal=SIGUSR1@120 + PL's default
#   SLURMEnvironment(auto_requeue=True) → on SIGUSR1 PL saves a full
#   hpc_ckpt_*.ckpt (EMA injected via EMACallback.on_save_checkpoint) and
#   `scontrol requeue`s the job. The resume block below restores from the
#   FRESHEST of {hpc_ckpt_* , checkpoints/last.ckpt} across all timestamped run
#   dirs (PL's own hpc auto-detect only fires when ckpt_path is None AND in the
#   current run_dir, whose timestamp changes each restart → we resume manually).
#   → near-zero progress loss, automatic chaining beyond the 96h lprod cap.
#
#   sbatch evaluation/../next_slurms/FT_LARGE2_mr05_decFT_postnet.slurm
# ============================================================
#SBATCH --job-name=FT_LARGE2_mr05_decFT_postnet
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=96:00:00                # lprod cap (4-00:00:00); ~184 ep/job → 500 ep ≈ 3 chained jobs
#SBATCH --requeue                      # required for PL auto-requeue (+ node-failure insurance)
#SBATCH --signal=SIGUSR1@120           # SIGUSR1 to the srun step 120s before TIMEOUT → PL saves + requeues
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
export HF_HUB_OFFLINE=1                 # no teacher/CLAP in decoder-FT → stay fully offline

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

# ─── Source weights: LARGE2_mrmel05_e99 (FMA-large pretrain, epoch 99) ────────
LG2_CKPT="$WORK/C-VAE/runs/LARGE2_mrmel05_clap_wavTok/2026-06-13_11-14-50/checkpoints/epoch_099.ckpt"
if [ ! -f "$LG2_CKPT" ]; then
    echo "[FATAL] LARGE2 checkpoint not found: $LG2_CKPT" >&2
    exit 1
fi
echo "[INIT] decoder-finetune init from LARGE2 (EMA): $LG2_CKPT"

# ─── W&B resume (run-id persisted by train.py to .run_ids/<name>) ─────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_LARGE2_mr05_decFT_postnet"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} ──────────
RUN_BASE="$WORK/C-VAE/runs/FT_LARGE2_mr05_decFT_postnet"
pick_ckpt() {
    ls -t "$RUN_BASE"/*/hpc_ckpt_*.ckpt "$RUN_BASE"/*/checkpoints/last.ckpt 2>/dev/null | head -1
}
LATEST_CKPT=$(pick_ckpt)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop (transient crashes; TIMEOUT is handled by signal→requeue) ─────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_real_baseline \
        models.model.decoder.use_postnet=true \
        data.fma_subset=large \
        "+init_from=$LG2_CKPT" \
        +init_from_ema=true \
        +init_from_disc=false \
        trainer.trainer.freeze_encoder=true \
        trainer.trainer.warmup_steps=0 \
        trainer.trainer.epochs=500 \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=10 \
        trainer.loss_config.bottleneck.weights.kl=0.0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.5 \
        trainer.optimizer.lr=1e-4 \
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
        trainer.trainer.ckpt_dir=checkpoints/FT_LARGE2_mr05_decFT_postnet \
        trainer.wandb.name=FT_LARGE2_mr05_decFT_postnet \
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
    LATEST_CKPT=$(pick_ckpt)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
