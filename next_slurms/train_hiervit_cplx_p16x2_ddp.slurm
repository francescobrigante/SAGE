#!/bin/bash
# ============================================================
# HierViT Complex VAE — 2 GPU DDP, 100k steps on FMA medium — patch (16, 2)
#
# Square stem grid (64, 64) → finer time, equal-aspect token grid. After
# 3× PatchMerging the final grid is (8, 8) = 64 latent tokens (vs 32 in the
# main run with patch (16, 4)). Stage 1 attention is 4× heavier (4096² vs
# 2048² scores per head). All other hparams identical to the main run.
#
# Compression: ×8 (latent_channels=128, N_lat=64 complex tokens, 16384 scalars).
# ============================================================

#SBATCH --job-name=train_hiervit_cplx_p16x2_ddp
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
mkdir -p checkpoints/hiervit_cplx_p16x2_ddp

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
echo "HierViT Complex VAE — 2 GPU DDP, 100k steps — patch (16, 2) ablation"
echo "============================================================"

srun python train.py \
    models=hiervit_cplx_x64 \
    models.model.hier_vit.patch_size=[16,2] \
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
    trainer.trainer.ckpt_dir=checkpoints/hiervit_cplx_p16x2_ddp \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_false \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.project=c-vae \
    trainer.wandb.name=HierViT_cplx_p16x2_2GPU_100k
