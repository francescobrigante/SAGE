#!/bin/bash
# ============================================================
# J1_ps64x1_large_e256: definitive long run — 500k steps target
#
# BASE: PS=(64,1), Win=(4,32), 3 stages, embed=256
#       encoder depths=[2,18,2], decoder depths=[2,18,2], ~300M params
#       Latent: 128×8 complex, KL~1e-5
#
# LOSS: MRSTFT(1.0) + L1-waveform(0.5) + MR-mel(0.5)  [J4 winner]
# DATA: FMA-large (84,353 train tracks)
# BS:   global=64 (32 per GPU × 2 GPUs)
# LR:   3e-3  (conservative 3× linear scale vs BS-16 baseline at 1e-3)
#
# STEPS: 84353 / 64 = 1318 steps/epoch (drop_last)
#        380 epochs × 1318 = 500,840 steps total
#
# ⚠️  TIMING WARNING:
#     J4 [2,6,2] benchmark: ~60k steps/day on 2 GPUs.
#     [2,18,2] is ~2.2× heavier → estimated 25–55k steps per 2-day job.
#     This script will NOT complete 500k steps in a single run.
#     Chain multiple jobs resuming from last checkpoint:
#       Add to srun args: +ckpt_path=checkpoints/J1_ps64x1_large_e256/last.ckpt
# ============================================================
#SBATCH --job-name=J1_ps64x1_large_e256
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export WANDB_MODE=online
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_TAGS="$SLURM_JOB_NAME"
export WANDB_DIR=$WORK/C-VAE/.wandb_cache

set -a
source $WORK/C-VAE/.env
set +a

cd $WORK/C-VAE
source $WORK/.envs/c_vae/bin/activate

srun python train.py \
    models=swin_cplx_4s_x64 \
    "models.model.swin.patch_size=[64,1]" \
    "models.model.swin.window_size=[4,32]" \
    "models.model.swin.embed_dim=256" \
    "models.model.encoder.depths=[2,18,2]" \
    "models.model.encoder.num_heads=[8,16,32]" \
    "models.model.decoder.depths=[2,18,2]" \
    "models.model.decoder.num_heads=[32,16,8]" \
    models.model.latent_channels=8 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=large" \
    "data.eval_dataset.custom_metadata_kwargs.subset=large" \
    data.train_dataloader.batch_size=32 \
    data.eval_dataloader.batch_size=32 \
    trainer.trainer.epochs=190 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-5 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    trainer.loss_config.mrmel.weights.mrmel=0.5 \
    "++trainer.loss_config.time.weights.l1=0.5" \
    trainer.optimizer.lr=2e-3 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.trainer.check_val_every_n_epoch=2 \
    trainer.trainer.save_every_n_epochs=10 \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=false" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/J1_ps64x1_large_e256 \
    trainer.wandb.name=J1_ps64x1_large_e256
