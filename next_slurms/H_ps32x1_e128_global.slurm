#!/bin/bash
# ============================================================
# H_ps32x1_e128_global: PS=(32,1), Win=(32,128), 4 stage, embed=128
#
# GRID (PatchEmbed → PatchMerge ×3):
#   Stage 0:  32×128  (1024/32  ×  128/1)   — 4096 token
#   Stage 1:  16×64                           — 1024 token
#   Stage 2:   8×32                           —  256 token
#   Stage 3:   4×16   (bottleneck)            —   64 token
#
# GLOBAL ATTENTION — window=[32,128]:
#   Stage 0: win [32,128] == grid [32,128] → 1 finestra = global ✓ (4096 tok/head)
#   Stage 1: win [32,128] > grid [16,64]  → clamped [16,64]  → global ✓
#   Stage 2: win [32,128] > grid [8,32]   → clamped  [8,32]  → global ✓
#   Stage 3: win [32,128] > grid [4,16]   → clamped  [4,16]  → global ✓
#   Tutti gli stage: ogni token attende tutti gli altri nella propria mappa → full global.
#   NB: stage 0 ha 4096 token → matrice attenzione 4096² per head (~512 MB) — borderline A100.
#
# DIFFERENZE rispetto a H_ps32x1_e128 (win=[8,32], globale solo da stage 2):
#   window [8,32] → [32,128]: global portato dallo stage 2 allo stage 0
#
# PROX STEP (identico a H_ps32x1_e128):
#   embed 64→128 | LR 1e-4→1e-3 | MRSTFT | no KL annealing
#
# embed=128: ratio 2× patch input (32×1×2=64 complex) — canonical Swin-Tiny ratio
# head_dim=32 per tutti gli stage (embed/num_heads[0] = 128/4)
# Params: ~196M, Latent: 64×16, KL atteso ~131
# ============================================================
#SBATCH --job-name=H_ps32x1_e128_global
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/H_ps32x1_e128_global
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
    "models.model.swin.patch_size=[32,1]" \
    "models.model.swin.window_size=[32,128]" \
    "models.model.swin.embed_dim=128" \
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
    trainer.trainer.ckpt_dir=checkpoints/H_ps32x1_e128_global \
    trainer.wandb.name=H_ps32x1_e128_global
