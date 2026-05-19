#!/bin/bash
# ============================================================
# H_ps4x1_5s_e96_real: PS=(4,1), Win=[32,16], 5 stage, depths=[2,2,2,6,2], embed=96
# Domain: REAL (cac=true, in_channels=4: Re_L, Im_L, Re_R, Im_R)
#
# GRID (PatchEmbed → PatchMerge ×4):
#   Stage 0: 256×128  (1024/4 × 128/1)  — 32768 token
#   Stage 1: 128×64                      —  8192 token
#   Stage 2:  64×32                      —  2048 token
#   Stage 3:  32×16  — win [32,16] == grid → global ✓  (n-1)
#   Stage 4:  16×8   (bottleneck)        —   128 token
#
# ATTENTION — window=[32,16]:
#   Stage 0: win [32,16] < grid [256,128] → locale (8×8=64 finestre da 32×16=512 tok)
#   Stage 1: win [32,16] < grid [128,64]  → locale (4×4=16 finestre)
#   Stage 2: win [32,16] < grid [64,32]   → locale (2×2=4 finestre)
#   Stage 3: win [32,16] == grid [32,16]  → global ✓ (1 finestra)  ← n-1
#   Stage 4: win [32,16] > grid [16,8]    → clamped [16,8] → global ✓  ← n
#
# RATIO & PARAMS:
#   embed=96: ratio 6× patch input (4×1×4ch=16 real) — tra canonical 4× e 8× del ℂ
#   head_dim=24 costante (96/4=24, 192/8=24, 384/16=24, 768/32=24, 1536/64=24)
#   Params: ~200M
#   Latent: 128×16 real (seq=128, channels=16 real) — seq×ch=2048, ×64 ✓
#   KL atteso: ~71 (stessa latent shape di H_ps4x1_5s_e64 ℂ in termini di seq)
#
# CONFRONTO DIRETTO:
#   vs H_ps4x1_5s_e64 (ℂ): stesso PS/grid, real vs complex, embed 64→96, ch 8→16, win [16,8]→[32,16]
#   vs swin_real_4s_x64 (4s iso-param): +1 stage, embed 92→96, latent_ch 16→16 (invariato), win→rettangolare
#   Global agli ultimi 2 stage (n-1 e n) — evidenza B2w8: +2.4 dB; G-suite: +0.6/+1.9 dB
# ============================================================
#SBATCH --job-name=H_ps4x1_5s_e96_real
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

mkdir -p logs checkpoints/H_ps4x1_5s_e96_real
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

# latent_tokens = 16×8 = 128  →  latent_channels = 2048/128 = 16 real  (×64 ✓)
srun python train.py \
    models=swin_real_4s_x64 \
    "models.model.swin.patch_size=[4,1]" \
    "models.model.swin.window_size=[32,16]" \
    "models.model.swin.embed_dim=96" \
    "models.model.encoder.depths=[2,2,2,6,2]" \
    "models.model.encoder.num_heads=[4,8,16,32,64]" \
    "models.model.decoder.depths=[2,2,6,2,2]" \
    "models.model.decoder.num_heads=[64,32,16,8,4]" \
    models.model.latent_channels=16 \
    data.train_dataset.cac=true \
    data.eval_dataset.cac=true \
    data.demo.istft_params.cac=true \
    data.model_channels=4 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=97 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=1.0 \
    trainer.loss_config.spectral.weights.mrstft=0.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    trainer.scheduler.total_steps=120000 \
    trainer.scheduler.warmup_steps=6000 \
    trainer.optimizer.lr=1e-3 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/H_ps4x1_5s_e96_real \
    trainer.wandb.name=H_ps4x1_5s_e96_real
