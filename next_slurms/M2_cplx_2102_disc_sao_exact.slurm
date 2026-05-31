#!/bin/bash
# ============================================================
# M2_cplx_2102_disc_sao_exact:
#   Identico a K13, ma con stages [2,10,2] per encoder e decoder.
#   - 3-stage Swin V2 Complex VAE + discriminator con parametri SAO
#   - depths=[2,10,2], embed_dim=256
#   - patch_size=[64,1], window_size=[4,32]
# ============================================================
#SBATCH --job-name=M2_cplx_2102_disc_sao_exact
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-08:00:00
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

# ─── Auto-resume: W&B run_id + last checkpoint ──────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/M2_cplx_2102_disc_sao_exact"

# Override hardcoded WANDB vars with persisted run_id if available
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# Auto-detect newest last.ckpt
LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/M2_cplx_2102_disc_sao_exact"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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

    srun python train.py \
        models=swin_cplx_baseline \
        "models.model.encoder.depths=[2,10,2]" \
        "models.model.decoder.depths=[2,10,2]" \
        "models.model.swin.patch_size=[64,1]" \
        "models.model.swin.window_size=[4,32]" \
        "models.model.swin.embed_dim=256" \
        models.model.latent_channels=8 \
        "data.train_dataset.custom_metadata_kwargs.subset=medium" \
        "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
        trainer.trainer.epochs=120 \
        trainer.trainer.kl_annealing_epochs=0 \
        trainer.loss_config.bottleneck.weights.kl=1e-4 \
        trainer.loss_config.spectral.weights.stft_mse=0.0 \
        trainer.loss_config.spectral.weights.mrstft=1.0 \
        "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
        trainer.trainer.clip_grad_norm=20.0 \
        trainer.optimizer.lr=1e-3 \
        "++trainer.optimizer.weight_decay_exclude_1d=true" \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=3e-4" \
        "++trainer.disc_optimizer.betas=[0.8,0.99]" \
        "++trainer.disc_optimizer.weight_decay=1e-3" \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="bf16-mixed" \
        "++trainer.trainer.compile=false" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/M2_cplx_2102_disc_sao_exact \
        trainer.wandb.name=M2_cplx_2102_disc_sao_exact \
        "++trainer.loss_config.discriminator.type=encodec" \
        "++trainer.loss_config.discriminator.config.filters=64" \
        "++trainer.loss_config.discriminator.config.n_ffts=[2048,1024,512,256,128]" \
        "++trainer.loss_config.discriminator.config.hop_lengths=[512,256,128,64,32]" \
        "++trainer.loss_config.discriminator.config.win_lengths=[2048,1024,512,256,128]" \
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=5.0" \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/M2_cplx_2102_disc_sao_exact"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
