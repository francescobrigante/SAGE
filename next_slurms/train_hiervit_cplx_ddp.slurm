#!/bin/bash
# ============================================================
# HierViT Complex VAE — 2 GPU DDP, 100k training steps on FMA medium
#
# Dataset: FMA medium (~25k tracks → ~20k train samples after split).
# DDP across 2 A100s with B=16 per rank → 32 effective batch.
# Steps/epoch ≈ 20000 / 32 = 625. 100k steps ≈ ~160 epochs.
# Stop condition: max_steps=100000 (max_epochs set high as effective infinity).
#
# Compression: ×16 (latent_channels=128, N_lat=32 complex tokens).
# ============================================================

#SBATCH --job-name=train_hiervit_cplx_ddp
#SBATCH --account=IscrC_PDR
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/hiervit_cplx_x16_ddp

# Do NOT load cineca-ai/profile modules: our .venv has its own self-contained
# torch 2.7.1+cu126 stack (bundled CUDA runtime libs). Mixing CINECA's torch
# with our torchaudio causes an ABI mismatch (undefined symbol c10::Error::what()).

# Single-thread BLAS per DataLoader worker — see swin DDP slurm for rationale.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# DDP / NCCL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Compute-node internet access via CINECA squid proxy (probed: 3133 is live).
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

# W&B: project comes from --trainer.wandb.project; entity from env.
# Online mode (default) → live streaming to wandb.ai through the squid proxy above.
export WANDB_ENTITY=ar_spectra
export WANDB_TAGS="$SLURM_JOB_NAME,hiervit,cplx,fma_medium"
export WANDB_DIR=$WORK/lcerovaz/C-VAE/.wandb_cache
mkdir -p $WANDB_DIR

cd $WORK/lcerovaz/C-VAE
source .venv/bin/activate

echo "============================================================"
echo "HierViT Complex VAE — 2 GPU DDP, 100k steps"
echo "============================================================"

srun python train.py \
    models=hiervit_cplx_x64 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    +trainer.trainer.max_steps=100000 \
    trainer.trainer.epochs=1000 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.scheduler.total_steps=100000 \
    trainer.scheduler.warmup_steps=5000 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.3 \
    trainer.trainer.ckpt_dir=checkpoints/hiervit_cplx_x16_ddp \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_false \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.project=c-vae \
    trainer.wandb.name=HierViT_cplx_x16_2GPU_100k
