#!/bin/bash
# ============================================================
# LARGE2_lrms_disc: STEREO-COLLAPSE pretrain arm "STEREO-AWARE DISC".
#   Derived VERBATIM from LARGE2_mrmel05_clap_wavTok, ONLY disc delta:
#     - recon losses UNCHANGED (stft_mse=1.0, no mrstft_sd)
#     - discriminator.config.fold_lrms=true → WavTokenizer critics see
#       [L,R | M,S] folded into the batch instead of channel-blind (b c)→(bc)
#   HYPOTHESIS: no critic in the baseline stack ever sees L/R jointly or M/S
#   (SAO's Encodec disc does, in_channels=2) → an M/S-aware adversarial signal
#   alone may prevent the Side collapse, isolating disc- vs recon-loss causes.
#   Spec: docs/superpowers/specs/2026-07-26-stereo-collapse-sao-loss-design.md
#   BOUNDED 2-DAY RUN on IscrC_MM; SIGUSR1@120 + freshest-of resume → requeue-safe.
#   Monitor offline: evaluate_swin_varT.py --compute-ms-metrics (sisdr_s/sisdr_m).
# ============================================================
#SBATCH --job-name=LARGE2_lrms_disc
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP (kept = LARGE2 for step-count comparability)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=2-00:00:00              # bounded 2-day ablation block (~epoch 70 of 100)
#SBATCH --requeue                      # node-failure insurance + signal-driven requeue chain
#SBATCH --signal=SIGUSR1@120           # SIGUSR1 to the srun step 120s before TIMEOUT → PL saves
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

# C-VAE lives in IscrC_AHNetBio regardless of the accounting project ($WORK may
# point to IscrC_MM after the budget switch → never rely on it here).
WORKDIR=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio   # CLAP teacher in $FAST/models (not on IscrC_MM fast)

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORKDIR/C-VAE/.wandb_cache

set -a
source $WORKDIR/C-VAE/.env
set +a

cd $WORKDIR/C-VAE
source $WORKDIR/.envs/c_vae/bin/activate

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORKDIR/C-VAE/.run_ids/LARGE2_lrms_disc"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} ──────────
RUN_BASE="$WORKDIR/C-VAE/runs/LARGE2_lrms_disc"
pick_ckpt() {
    ls -t "$RUN_BASE"/*/hpc_ckpt_*.ckpt "$RUN_BASE"/*/checkpoints/last.ckpt 2>/dev/null | head -1
}
LATEST_CKPT=$(pick_ckpt)
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
        data.fma_subset=large \
        "++trainer.loss_config.semantic_distill.teacher_type=clap" \
        "++trainer.loss_config.semantic_distill.config.proj_dim=512" \
        "++trainer.loss_config.semantic_distill.detach_warmup_steps=25000" \
        trainer.trainer.epochs=100 \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=10 \
        trainer.loss_config.bottleneck.weights.kl=1e-4 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.5 \
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
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=0.2" \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/LARGE2_lrms_disc \
        trainer.wandb.name=LARGE2_lrms_disc \
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
