#!/bin/bash
# ============================================================
# L3_ps64x1_3s_e256_mse: Swin baseline (champion) + stft_MSE invece di MRSTFT
#   IDENTICO a EXAMPLE.slurm, unica variabile:
#     stft_mse: 0.0 -> 1.0
#     mrstft:   1.0 -> 0.0
#   Tutto il resto invariato (kl=1e-5, lr=1e-3, swin_cplx_baseline).
#
# IPOTESI (Q2): isolare l'effetto di MSE sull'architettura champion 64x1.
#   Dati: H_ps32x1_e128 (MSE) = SI-SDR best assoluto (-0.14) + KL bassa/sana,
#   ma perceptual pessimo (val/stft 6.47, "solo basse freq") -- pero' su ALTRA arch.
#   Qui lo testiamo sul baseline per confronto pulito (1 variabile) con:
#     H baseline 64x1 + MRSTFT -> SI-SDR -2.12, val/stft 2.16, latent_std 0.70.
#
# COSA GUARDARE:
#   SI-SDR      -> atteso molto migliore (MSE supervisiona fase/waveform)
#   val/stft,mel-> atteso peggiore (MSE pesa le basse freq, trascura le alte)
#   latent_std/KL -> atteso sano (MSE tiene KL bassa)
#
# NEXT se confermato: +mrmel e/o +MRSTFT (una alla volta) per recuperare il perceptual.
#
# ARCH (swin_cplx_baseline): PS=64x1, window=[4,32], embed=256, depths=[2,6,2]
#   Stage 0: 16x128 -> Stage 1: 8x64 -> Stage 2: 4x32 (128 token, x64 complex)
# DATA: FMA-medium (default), no disc.
# ============================================================
#SBATCH --job-name=L3_ps64x1_3s_e256_mse
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
export WANDB_RESUME=must
export WANDB_RUN_ID=b5kujbgf

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# ─── Auto-resume: W&B run_id + last checkpoint ──────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/L3_ps64x1_3s_e256_mse"

# Override hardcoded WANDB vars with persisted run_id if available
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# Auto-detect newest last.ckpt (finds pre-crash checkpoint on first launch too)
LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/L3_ps64x1_3s_e256_mse"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop (up to 8 attempts — recovers from DataLoader timeouts) ───────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_cplx_baseline \
        trainer.trainer.epochs=97 \
        trainer.loss_config.bottleneck.weights.kl=1e-5 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="bf16-mixed" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/L3_ps64x1_3s_e256_mse \
        trainer.wandb.name=L3_ps64x1_3s_e256_mse \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/L3_ps64x1_3s_e256_mse"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
