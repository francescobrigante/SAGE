#!/bin/bash
# ============================================================
# PDR_swin_swiglu_xsa_hr3
#
# Standard baseline optimizer launch for SwiGLU+XSA on CINECA/PDR:
#   - model: swin_real_swiglu_xsa
#   - swiglu_hidden_ratio=3.0 from the model config
#   - attention_variant=xsa from the model config
#
# Uses the verified FMA-medium dataset path on this user's SCRATCH.
# ============================================================
#SBATCH --job-name=PDR_swin_sx_hr3
#SBATCH --account=IscrC_PDR
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=80:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load profile/deeplrn cineca-ai/4.3.0

export CVAE_WORK=/leonardo_work/IscrC_PDR/lcerovaz
export CVAE_FAST=/leonardo_scratch/fast/IscrC_PDR/lcerovaz
export CVAE_REPO="$CVAE_WORK/C-VAE"
export FAST="$CVAE_FAST"
export DATA_PATH=/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium
export FMA_METADATA=/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_metadata/tracks.csv

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HOME="$CVAE_FAST/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3

cd "$CVAE_REPO"
mkdir -p logs .run_ids .wandb_cache

if [ -f "$CVAE_REPO/.env" ]; then
    set -a
    source "$CVAE_REPO/.env"
    set +a
fi

# Keep PDR/data choices authoritative even if .env was copied from another account.
export FAST="$CVAE_FAST"
export DATA_PATH=/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium
export FMA_METADATA=/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_metadata/tracks.csv
export HF_HOME="$CVAE_FAST/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
export WANDB_DIR="$CVAE_REPO/.wandb_cache"

if [ ! -f "$CVAE_REPO/.venv/bin/activate" ]; then
    echo "[ERROR] Missing $CVAE_REPO/.venv. Run uv sync during the interactive debug step before sbatch."
    exit 2
fi
source "$CVAE_REPO/.venv/bin/activate"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CVAE_REPO/.venv/bin/python"

if ! "$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"[ENV] python={sys.executable}")
print(f"[ENV] torch={torch.__file__}")
PY
then
    echo "[WARN] Python/Torch preflight failed; continuing into retry-wrapped training command."
fi

EXPERIMENT="final_clap_distill_swiglu_xsa"
GEN_LR="1e-3"
RUN_NAME="final_clap_distill_swiglu_xsa_hr3"

export WANDB_TAGS="$SLURM_JOB_NAME,$RUN_NAME,swiglu,xsa,hr3"
RUN_ID_FILE="$CVAE_REPO/.run_ids/$RUN_NAME"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID
    WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(
    { find "$CVAE_REPO/runs/$RUN_NAME" -path "*/checkpoints/last.ckpt" -type f -printf "%T@ %p\n" 2>/dev/null || true; } \
        | sort -nr \
        | awk 'NR == 1 {print $2}'
)
CKPT_ARG=()
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG=("+ckpt_path=$LATEST_CKPT")
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

MAX_ATTEMPTS=8
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "[$(date '+%F %T')] ${RUN_NAME}: training attempt ${attempt}/${MAX_ATTEMPTS}"

    set +e
    srun --kill-on-bad-exit=1 "$PYTHON_BIN" train.py \
        "+experiment=${EXPERIMENT}" \
        data.fma_subset=medium \
        "++trainer.loss_config.semantic_distill.teacher_type=clap" \
        "++trainer.loss_config.semantic_distill.config.proj_dim=512" \
        "++trainer.loss_config.semantic_distill.detach_warmup_steps=25000" \
        trainer.trainer.epochs=100 \
        "+trainer.trainer.max_steps=100000" \
        trainer.trainer.check_val_every_n_epoch=1 \
        trainer.trainer.save_every_n_epochs=10 \
        trainer.loss_config.bottleneck.weights.kl=1e-4 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.5 \
        trainer.optimizer.lr="$GEN_LR" \
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
        trainer.trainer.num_gpus="$SLURM_NTASKS_PER_NODE" \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision=32 \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        "trainer.trainer.ckpt_dir=checkpoints/${RUN_NAME}" \
        "trainer.wandb.name=${RUN_NAME}" \
        "${CKPT_ARG[@]}"
    exit_code=$?
    set -e
    [ "$exit_code" -eq 0 ] && break

    echo "[$(date '+%F %T')] ${RUN_NAME}: attempt ${attempt} failed with exit ${exit_code}; retrying in 15s"
    sleep 15
    attempt=$((attempt + 1))

    if [ -f "$RUN_ID_FILE" ]; then
        export WANDB_RESUME=must
        export WANDB_RUN_ID
        WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    fi
    LATEST_CKPT=$(
        { find "$CVAE_REPO/runs/$RUN_NAME" -path "*/checkpoints/last.ckpt" -type f -printf "%T@ %p\n" 2>/dev/null || true; } \
            | sort -nr \
            | awk 'NR == 1 {print $2}'
    )
    if [ -n "$LATEST_CKPT" ]; then
        CKPT_ARG=("+ckpt_path=$LATEST_CKPT")
    fi
done

if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "[ERROR] ${RUN_NAME}: failed after ${MAX_ATTEMPTS} attempts"
    exit 1
fi

echo "[$(date '+%F %T')] ${RUN_NAME}: completed successfully"
