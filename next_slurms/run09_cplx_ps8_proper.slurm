#!/bin/bash
# ============================================================
# Run 9 — Bottleneck Ablation: proper (C=0) su compact latent
# = Run 3 config + bottleneck proper (ptp=2)
# DDP 2×A100: 16000 step, 800 warmup
# ============================================================
#SBATCH --job-name=r09_proper
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

mkdir -p logs checkpoints/run09_proper
module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache
mkdir -p $WORK/C-VAE/.wandb_cache
cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    models=swin_cplx_4s_x64 \
    models.model.swin.patch_size=8 \
    models.model.latent_channels=32 \
    models.model.parameters_to_predict=2 \
    models.model.bottleneck.proper=true \
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
    trainer.trainer.ckpt_dir=checkpoints/run09_proper \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_false \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.name=r09_cplx_ps8_proper
# OPT (vs versione pre-speedup): aggiunte 3 flag perf
#   strategy=ddp_find_unused_parameters_false  (era: strategy=ddp)
#   precision="bf16-mixed"                     (era: 32-true default)
#   +trainer.trainer.compile=true              (nuova: torch.compile opt-in)
