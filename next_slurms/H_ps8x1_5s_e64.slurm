#!/bin/bash
# ============================================================
# H_ps8x1_5s_e64: PS=(8,1), Win=[16,16], 5 stage, depths=[2,2,2,6,2], embed=64
#
# GRID (PatchEmbed → PatchMerge ×4):
#   Stage 0: 128×128  (1024/8 × 128/1)  — 16384 token
#   Stage 1:  64×64                      —  4096 token
#   Stage 2:  32×32                      —  1024 token
#   Stage 3:  16×16  — win [16,16] == grid → global ✓
#   Stage 4:   8×8   (bottleneck)        —    64 token
#
# ATTENTION — window=[16,16]:
#   Stage 0: win [16,16] < grid [128,128] → locale (8×8=64 finestre)
#   Stage 1: win [16,16] < grid [64,64]   → locale (4×4=16 finestre)
#   Stage 2: win [16,16] < grid [32,32]   → locale (2×2=4 finestre)
#   Stage 3: win [16,16] == grid [16,16]  → global ✓ (1 finestra)
#   Stage 4: win [16,16] > grid [8,8]     → clamped [8,8] → global ✓
#
# RATIO & PARAMS:
#   embed=64: ratio 4× patch input (8×1×2=16 complex)
#   head_dim=16 per tutti gli stage (64/4=16)
#   Params: ~177M (iso-parameter con H_ps4x1_5s_e64 e H_ps16x1_5s_e64)
#   Latent: 64×16 (seq=64, channels=16 complex) — seq×ch=1024, ×64 ✓
#   KL atteso: ~71 (come D_ps16x1 che ha stessa latent 128×8)
#
# CONFRONTO DIRETTO:
#   vs H_ps16x1_5s_e64 (5-stage): PS 16→8 (2× finer), grid 4×8→8×8, ch 32→16
#   vs H_ps4x1_5s_e64  (5-stage): PS 4→8  (2× coarser), grid 16×8→8×8, ch 8→16
#   vs H_ps32x1_e128   (4-stage): 5s vs 4s; stessa latent shape 64×16
#   Global agli ultimi 2 stage (n-1 e n) — più aggressivo della G-suite (solo n-1)
# ============================================================
#SBATCH --job-name=H_ps8x1_5s_e64
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

mkdir -p logs checkpoints/H_ps8x1_5s_e64
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
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

# latent_tokens = 8×8 = 64  →  latent_channels = 1024/64 = 16  (×64 complex ✓)
srun python train.py \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[8,1]" \
    "models.model.swin.window_size=[16,16]" \
    "models.model.swin.embed_dim=64" \
    "models.model.encoder.depths=[2,2,2,6,2]" \
    "models.model.encoder.num_heads=[4,8,16,32,64]" \
    "models.model.decoder.depths=[2,2,6,2,2]" \
    "models.model.decoder.num_heads=[64,32,16,8,4]" \
    models.model.latent_channels=16 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=97 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    trainer.scheduler.total_steps=120000 \
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
    trainer.trainer.ckpt_dir=checkpoints/H_ps8x1_5s_e64 \
    trainer.wandb.name=H_ps8x1_5s_e64
