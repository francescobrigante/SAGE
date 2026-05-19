#!/bin/bash
# ============================================================
# H_ps4x1_5s_e64: PS=(4,1), Win=(16,8), 5 stage, depths=[2,2,2,6,2], embed=64
#
# GRID (PatchEmbed → PatchMerge ×4):
#   Stage 0: 256×128  (1024/4  ×  128/1)
#   Stage 1: 128×64
#   Stage 2:  64×32
#   Stage 3:  32×16
#   Stage 4:  16×8    (bottleneck) — 128 token
#
# GLOBAL ATTENTION — window=[16,8]:
#   Stage 0-2: local windows
#   Stage 3: 2×2=4 windows
#   Stage 4: win [16,8] == grid [16,8] → global ✓
#
# RATIO & PARAMS:
#   embed=64: ratio 8× patch input (4×1×2=8 complex) — Molto sovra-parametrizzato al PatchEmbed
#   Params: ~196M (Iso-parameter con H_ps32x1_e128 4-stage)
#   Latent: 128×8 (seq=128, channels=8 complex) — Ottimo per LDM, KL atteso ~71
#
# CONTINUATION (step 120k→240k):
#   Resume: runs/H_ps4x1_5s_e64/2026-05-17_00-44-13/checkpoints/last.ckpt
#   W&B:    run ID 9llioltk (stesso run, continua i log)
#   LR al resume (step~120k su 240k): ~5.2e-4 → cosine a 0 su +120k step
# ============================================================
#SBATCH --job-name=H_ps4x1_5s_e64
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/H_ps4x1_5s_e64
module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_RESUME=must
export WANDB_RUN_ID=9llioltk
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    +ckpt_path=$WORK/C-VAE/runs/H_ps4x1_5s_e64/2026-05-17_00-44-13/checkpoints/last.ckpt \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[4,1]" \
    "models.model.swin.window_size=[16,8]" \
    "models.model.swin.embed_dim=64" \
    "models.model.encoder.depths=[2,2,2,6,2]" \
    "models.model.encoder.num_heads=[4,8,16,32,64]" \
    "models.model.decoder.depths=[2,2,6,2,2]" \
    "models.model.decoder.num_heads=[64,32,16,8,4]" \
    models.model.latent_channels=8 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=194 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    trainer.scheduler.total_steps=240000 \
    trainer.scheduler.warmup_steps=6000 \
    trainer.optimizer.lr=1e-3 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/H_ps4x1_5s_e64 \
    trainer.wandb.name=H_ps4x1_5s_e64
