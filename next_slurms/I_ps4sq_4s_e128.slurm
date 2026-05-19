#!/bin/bash
# ============================================================
# I_ps4sq_4s_e128: PS=(4,4) square, Win=(64,8), 4 stage, embed=128
# Grid: 256×32 → 128×16 → 64×8 (n-1) → 32×4 (deepest)
# Win (64,8): global a n-1 (64×8) ✓
#             global al deepest (32×4, clamped a (32,4)) ✓
# embed=128 (4× patch input 4×4×2ch=32): ratio canonico Swin
# Latent: seq=128, ch=8, 128×8, KL atteso ~70
# Params: ~196M
#
# Suite I — PS square modernizzato (LR=1e-3, H-series scale):
#   PS=(4,4) fine-grain su entrambi gli assi T e F
#   Gap non esplorato: patch quadrato piccolo + latent 128×8
#   Confronto diretto con H_ps4x1_5s_e64 (stessa latent, PS_T identico)
# ============================================================
#SBATCH --job-name=I_ps4sq_4s_e128
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/I_ps4sq_4s_e128
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
    "models.model.swin.patch_size=[4,4]" \
    "models.model.swin.window_size=[64,8]" \
    "models.model.swin.embed_dim=128" \
    models.model.latent_channels=8 \
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
    trainer.trainer.ckpt_dir=checkpoints/I_ps4sq_4s_e128 \
    trainer.wandb.name=I_ps4sq_4s_e128
