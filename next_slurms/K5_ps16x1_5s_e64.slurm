#!/bin/bash
# ============================================================
# K5_ps16x1_22622_e64: Suite K — PS=(16,1), Win=[8,16], 5 stage, embed=64
# Depths: encoder=[2,2,6,2,2]  decoder=[2,2,6,2,2]  (symmetric)
#
# ABLATION: encoder simmetrico al decoder (enc=dec).
#   H baseline: enc=[2,2,2,6,2]  dec=[2,2,6,2,2]  — asimmetrico
#   K4:         enc=[2,2,2,6,2]  dec=[2,6,2,2,2]  — reverse
#   K5:         enc=dec=[2,2,6,2,2]                — deep al pos 3 per entrambi
#   → testa se la simmetria enc/dec è vantaggiosa rispetto all'asimmetria di H.
#     In K5 il deep stage si sposta da pos 4 (H enc) a pos 3 — stage a
#     risoluzione media con attenzione locale (4 finestre 8×16).
#
# GRID (PatchEmbed → PatchMerge ×4):
#   Stage 0:  64×128  (enc stage 0 / dec stage 4)
#   Stage 1:  32×64
#   Stage 2:  16×32   ← K5 deep enc+dec (6 blocks, locale 2×2 finestre)
#   Stage 3:   8×16   win [8,16] == grid → global ✓
#   Stage 4:   4×8    (bottleneck) → clamped → global ✓
#
# ENC attention [win=[8,16]]:
#   Stage 0: locale (8×8=64 finestre)
#   Stage 1: locale (4×4=16 finestre)
#   Stage 2: locale (2×2=4 finestre) ← 6 blocks con attenzione locale
#   Stage 3: global ✓
#   Stage 4: global (clamped) ✓
# DEC attention [specchio di enc]:
#   Stage 0 (deepest): global ✓ (clamped)
#   Stage 1: global ✓
#   Stage 2: locale (2×2=4 finestre) ← 6 blocks
#   Stage 3: locale (4×4=16 finestre)
#   Stage 4: locale (8×8=64 finestre)
#
# PARAMS: ~177M  |  Latent: 32×32 (×64 ✓)  |  KL atteso: ~231
# LOSS: MRSTFT(1.0)
# DATA: FMA-medium (19,922 train tracks)
# ============================================================
#SBATCH --job-name=K5_ps16x1_22622_e64
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

srun python train.py \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[16,1]" \
    "models.model.swin.window_size=[8,16]" \
    "models.model.swin.embed_dim=64" \
    "models.model.encoder.depths=[2,2,6,2,2]" \
    "models.model.encoder.num_heads=[4,8,16,32,64]" \
    "models.model.decoder.depths=[2,2,6,2,2]" \
    "models.model.decoder.num_heads=[64,32,16,8,4]" \
    models.model.latent_channels=32 \
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
    trainer.optimizer.lr=1e-3 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/K5_ps16x1_22622_e64 \
    trainer.wandb.name=K5_ps16x1_22622_e64
