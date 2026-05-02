#!/bin/bash
# ============================================================
# Swin Real VAE - 4-stage, x64, 50 epochs, 2 GPUs DDP (A100)
#
# Hardware: 1 Nodo, 2 GPU
# DDP (DistributedDataParallel) taglia il dataset in due:
#   6400 campioni / 2 GPU = 3200 campioni a testa.
#   3200 campioni / 16 batch_size = ~200 batch per rango per epoca!
# Steps totali = 50 epoche * 200 batch = 10000 step attesi.
# ============================================================

#SBATCH --job-name=train_real_ddp
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # DDP: 1 task per GPU
#SBATCH --gres=gpu:2                   # Alloca 2x A100 (64GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
mkdir -p checkpoints/real_x64_ddp

module load profile/deeplrn cineca-ai/4.3.0

# OMP=1: each DataLoader worker runs single-threaded STFT. Without this,
# num_workers=8 workers × OMP=8 = 64 threads on 8 CPUs → context-switch thrashing.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

echo "============================================================"
echo "Starting REAL-VAL training on 2 GPUs (DDP)"
echo "============================================================"

# Calcolo Scheduler: 50 epoche * 200 batch (perchè dimezzato dal DDP) = 10000 total_steps
# Warmup: 5% di 10000 = 500 step

srun python train.py \
    models=swin_real_4s_x64 \
    trainer.trainer.epochs=50 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.scheduler.total_steps=10000 \
    trainer.scheduler.warmup_steps=500 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.3 \
    trainer.trainer.ckpt_dir=checkpoints/real_x64_ddp \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_false \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.name=TEST_real_x64_2GPU  # <--- CAMBIA QUI IL NOME VISIBILE SU W&B
# ─────────────────────────────────────────────────────────────────────────────
#   trainer.trainer.strategy=ddp_find_unused_parameters_false
#       Era: trainer.trainer.strategy=ddp (default Lightning, alias di
#       ddp_find_unused_parameters_true). La variante _false skippa la scansione
#       DDP dei parametri non usati ad ogni backward.
#       Sicuro perché TUTTI i parametri Swin sono usati. Speedup ~5-10% per step.
#   trainer.trainer.precision="bf16-mixed"
#       Era: precision=32-true (default in trainer.yaml). Mixed precision bf16
#       sui Tensor Cores A100. Sicuro sul real (parametri float32, autocast
#       converte i matmul/conv in bf16 mantenendo accumulatori fp32). Speedup
#       atteso ~30-60% sui matmul/conv (più del complex perché qui TUTTI gli
#       op sono in float32 e quindi tutti accelerati dal cast bf16).
#   +trainer.trainer.compile=true
#       Attiva torch.compile(mode="reduce-overhead", dynamic=False) in train.py.
#       CUDA Graphs sicure perché input shape è fissa (target_frames=128, F=1024).
#       Speedup atteso ~10-30%; fallback eager con log se compile fallisce.
# ─────────────────────────────────────────────────────────────────────────────
