#!/bin/bash
# ============================================================
# O5_real_chroma_ild — M5 real baseline + SAME §3.3.2 chroma+ILD regression
#   VERSIONE SENZA YAML: tutti i parametri passati come override Hydra CLI.
#
#   Differenze dal trainer.yaml base:
#     models:          SEANet_real_model → swin_real_baseline
#     kl:              1.0e-3  → 1.0e-5
#     mrmel:           0.3     → 0.0
#     semantic_regression: nuovo blocco (3 chroma heads + 1 ILD head)
#     aux_optimizer/scheduler: nuovi blocchi (non presenti in M5 base)
#
#   Loss attive: stft_MSE=1.0, kl=1e-5 + semantic_regression=1.0 (warmup 25k)
#   Config: solo questo script (nessun config/experiment/O5*.yaml richiesto).
# ============================================================
#SBATCH --job-name=O5_real_chroma_ild
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                   # 2x A100 DDP
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

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/O5_real_chroma_ild"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

CKPT_ARG=""

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_real_baseline \
        trainer.trainer.epochs=97 \
        trainer.optimizer.lr=1e-3 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=O5_real_chroma_ild \
        trainer.trainer.ckpt_dir=checkpoints/O5_real_chroma_ild \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.bottleneck.weights.kl=1.0e-4 \
        "+trainer.loss_config.semantic_regression.weights.regression=0.1" \
        "+trainer.loss_config.semantic_regression.config.latent_dim=64" \
        "+trainer.loss_config.semantic_regression.detach_warmup_steps=25000" \
        "+trainer.aux_optimizer._target_=torch.optim.AdamW" \
        "+trainer.aux_optimizer.lr=1.0e-3" \
        "+trainer.aux_optimizer.betas=[0.9,0.98]" \
        "+trainer.aux_optimizer.weight_decay=1.0e-4" \
        "+trainer.aux_scheduler._target_=torch.optim.lr_scheduler.ConstantLR" \
        "+trainer.aux_scheduler.factor=1.0" \
        "+trainer.aux_scheduler.total_iters=1" \
        "+trainer.aux_scheduler.interval=step" \
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
    CKPT_ARG=""
done
