#!/bin/bash
# ============================================================
# MED_LARGE2_mert_wavTok: LARGE2 recipe on FMA-MEDIUM, MERT teacher (A/B, layer 4)
#   - CLAP-vs-MERT semantic-distill ablation. This is the MERT arm.
#   - Derived VERBATIM from LARGE2_mrmel05_clap_wavTok, changing ONLY:
#       * data.fma_subset=large  → medium
#       * epochs=100             → 97
#       * account IscrC_AHNetBio → IscrC_MM (AHNetBio budget exhausted)
#       * time 80h               → 24h  (medium ≈ ¼ of large)
#       * distinct run/ckpt/W&B namespace
#   - Teacher = MERT-v1-95M layer 4: teacher_type=mert, proj_dim=768 ((B,768,T), time-pooled).
#   - Full LARGE2 recipe kept: mrmel=0.5, ema_decay=0.9998, detach_warmup=25000,
#     WavTokenizer rpgan disc, disc_lr=2e-4/adv=0.1/fm=0.2, kl=1e-4, lr=1e-3, precision 32.
#   - The CLAP arm (MED_LARGE2_clap_wavTok) differs ONLY in teacher_type + proj_dim.
# ============================================================
#SBATCH --job-name=MED_LARGE2_mert
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP (kept = LARGE2 for step-count comparability)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=24:00:00                # medium ≈ ¼ of large (~15h run + margin)
#SBATCH --requeue                      # node-failure insurance: SLURM re-queues, auto-resume from last.ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

# Pin project spaces explicitly (repo/venv on IscrC_AHNetBio work; datasets+models on IscrC_AHNetBio scratch)
# so the job is independent of the submitting shell's default-account $WORK/$FAST.
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MED_LARGE2_mert_wavTok"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MED_LARGE2_mert_wavTok"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        "++trainer.loss_config.semantic_distill.teacher_type=mert" \
        "++trainer.loss_config.semantic_distill.config.proj_dim=768" \
        "++trainer.loss_config.semantic_distill.detach_warmup_steps=25000" \
        trainer.trainer.epochs=97 \
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
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=0.2" \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/MED_LARGE2_mert_wavTok \
        trainer.wandb.name=MED_LARGE2_mert_wavTok \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MED_LARGE2_mert_wavTok"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
