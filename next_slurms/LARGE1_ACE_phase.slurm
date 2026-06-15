#!/bin/bash
# ============================================================
# LARGE1_ACE_phase: ACE-Step v1.5 §3.1 "final-phase adversarial sharpening"
#   translated onto LARGE1 (Swin C-VAE), CONTINUING from LARGE1 epoch_089 (~470k).
#
#   Paper (ACE-Step 1.5, arXiv 2602.00744 §3.1): in the final 100k/600k steps they
#   eliminate KL and raise adversarial 0.1->0.5. It is ONE continuous full-model
#   adversarial run (encoder NOT frozen, no separate decoder-FT, optimizer/scheduler
#   continue). We reproduce that here as a continuation of LARGE1, TEMPERED for our
#   case (encoder is the thesis deliverable + wavtok-RPGAN disc + low feature_matching):
#
#   ONLY changes vs LARGE1_clap_wavTok (paper-faithful: only KL + adv change; disc kept):
#     * RESUME weights+optimizer+scheduler+EMA from LARGE1 epoch_089 (best ckpt)
#     * trainer.loss_config.bottleneck.weights.kl       1e-4 -> 0      (paper: eliminate KL)
#     * discriminator.weights.adversarial               0.1  -> 0.5    (paper: ONLY adv x5)
#     * discriminator.weights.feature_matching          0.2  -> 0.2    (UNCHANGED — paper kept fm
#         fixed too; our 0.2 is tuned for the wavtok disc, NOT comparable to encodec's 5.0)
#     * epochs=108                                       (~+18 ep from e89 ≈ ~95-100k steps = paper's 17%)
#     * save_every_n_epochs=3                            (frequent ckpts for early eval)
#   KEEP identical: encoder/decoder/disc weights, optimizer+scheduler state (LR continues
#     ~5e-4 decaying), EMA 0.9998, CLAP distill, wavtok-RPGAN disc, stft_mse=1.0/mrmel=0.3,
#     gen lr base 1e-3, disc lr 2e-4, precision 32. Discriminator is NOT reinitialized.
#   NOTE: EMA is PRESERVED on resume (fix callbacks.py:121, 2026-06-12), NOT reset. The EMA
#     loaded at start = LARGE1's pre-ACE smoothed weights; with decay 0.9998 it LAGS the ACE
#     changes ~3-5 tau -> eval ACE ckpts only after >=8 epochs (~40k steps) so EMA tracks ACE.
# ============================================================
#SBATCH --job-name=LARGE1_ACE
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP (kept = LARGE1 for step comparability)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=24:00:00                # ~18 epochs on large ≈ ~10-11h; margin + requeue
#SBATCH --requeue
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

# ─── Resume logic ────────────────────────────────────────────────────────────
# First launch: start the ACE phase FROM LARGE1 epoch_089 (full state resume).
# Requeue:      resume from this run's own last.ckpt.
RUN_ID_FILE="$WORK/C-VAE/.run_ids/LARGE1_ACE_phase"
BASE_CKPT="$WORK/C-VAE/checkpoints/LARGE1_e89_470k.ckpt"   # symlink -> LARGE1 epoch_089.ckpt

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/LARGE1_ACE_phase"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] own checkpoint: $LATEST_CKPT"
else
    CKPT_ARG="+ckpt_path=$BASE_CKPT"
    echo "[INIT] ACE phase starting FROM LARGE1 e89: $BASE_CKPT"
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        +experiment=O4b_real_distill_mert \
        data.fma_subset=large \
        "++trainer.loss_config.semantic_distill.teacher_type=clap" \
        "++trainer.loss_config.semantic_distill.config.proj_dim=512" \
        "++trainer.loss_config.semantic_distill.detach_warmup_steps=25000" \
        trainer.trainer.epochs=108 \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=3 \
        trainer.loss_config.bottleneck.weights.kl=0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.3 \
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
        "++trainer.loss_config.discriminator.weights.adversarial=0.5" \
        "++trainer.loss_config.discriminator.weights.feature_matching=0.2" \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/LARGE1_ACE_phase \
        trainer.wandb.name=LARGE1_ACE_phase \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/LARGE1_ACE_phase"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
