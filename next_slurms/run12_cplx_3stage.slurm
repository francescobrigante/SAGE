#!/bin/bash
# ============================================================
# Run 12 — Hierarchical Bias: 3 stadi vs 4, emb=80 (iso-param ~50M)
# depths: [2,6,2], num_heads: [4,8,16], PS=8, stessa latent 128×8
# DDP 2×A100: 16000 step, 800 warmup
# ============================================================
#SBATCH --job-name=r12_3stage
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/run12_3stage
module load profile/deeplrn cineca-ai/4.3.0
# OMP=1: each DataLoader worker runs single-threaded STFT. Without this,
# num_workers=8 workers × OMP=8 = 64 threads on 8 CPUs → context-switch thrashing.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache
cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    models=swin_cplx_4s_x64 \
    models.model.swin.embed_dim=80 \
    models.model.swin.patch_size=8 \
    "models.model.encoder.depths=[2,6,2]" \
    "models.model.encoder.num_heads=[4,8,16]" \
    "models.model.decoder.depths=[2,6,2]" \
    "models.model.decoder.num_heads=[16,8,4]" \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    trainer.trainer.epochs=80 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.scheduler.total_steps=16000 \
    trainer.scheduler.warmup_steps=800 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.3 \
    trainer.trainer.ckpt_dir=checkpoints/run12_3stage \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.name=r12_cplx_3stage
# OPT (vs versione pre-speedup): aggiunte 3 flag perf
#   strategy=ddp_find_unused_parameters_true  (era: strategy=ddp)
#   precision="bf16-mixed"                     (era: 32-true default)
#   +trainer.trainer.compile=true              (nuova: torch.compile opt-in)
