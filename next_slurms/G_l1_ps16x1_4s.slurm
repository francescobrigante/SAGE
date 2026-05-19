#!/bin/bash
# ============================================================
# G_l1_ps16x1_4s: identico a D_ps16x1_4s +
#   time_domain L1 loss (weight=0.1) per correggere amplitude drift.
#
# Ipotesi: il path del gradiente di mrstft passa attraverso stft(istft(S))
#   (proiezione ortogonale, non identità), attenuando il segnale di ampiezza.
#   L1 waveform aggiunge un gradiente diretto su decoded vs reals,
#   senza proiezione STFT, analogamente a come opera un discriminatore.
# Variabile isolata rispetto a D: solo aggiunta di l1_time_loss=0.1.
# 12h: run diagnostica per verificare se le ampiezze convergono prima.
# ============================================================
#SBATCH --job-name=G_l1_ps16x1
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs checkpoints/G_l1_ps16x1_4s
module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_RUN_ID=hq9b9gpx
export WANDB_RESUME=must
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
    models.model.latent_channels=8 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=97 \
    trainer.trainer.kl_annealing_epochs=10 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    "++trainer.loss_config.time.weights.l1=0.3" \
    trainer.scheduler.total_steps=120000 \
    trainer.scheduler.warmup_steps=6000 \
    trainer.optimizer.lr=1e-4 \
    trainer.optimizer.weight_decay=0.05 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    trainer.trainer.clip_grad_norm=5.0 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/G_l1_ps16x1_4s \
    trainer.wandb.name=G_l1_ps16x1_4s \
    +ckpt_path=$WORK/C-VAE/runs/G_l1_ps16x1_4s/2026-05-14_08-08-01/checkpoints/last.ckpt
