#!/bin/bash
# ============================================================
# K13_ps64x1_3s_e256_disc_sao_exact:
#   H arch + MS-STFT disc con parametri IDENTICI a SAO:
#     - disc_lr=3e-4  (SAO default, separato da gen_lr=1e-3)
#     - disc betas=(0.8, 0.99), disc weight_decay=1e-3
#     - adv=0.1, fm=5.0, hinge, 5 scale STFT (invariati)
#     - warmup: InverseLR warmup=0.999 (no hard warmup_steps, = SAO)
#     - kl=1e-4 (da K8 — meglio di 1e-5 per latent_std)
#
# CONFRONTO SAO vs K8:
#   disc_lr:  3e-4   (SAO) vs 1e-3  (K8) → 3.3x piu basso
#   betas:    (0.8,0.99) vs (0.9,0.98) → beta1 piu bassa, piu stabile GAN
#   wd:       1e-3   vs  1e-4          → 10x piu alta
#   gen_lr:   1e-3   invariato
#
# GRID (PatchEmbed -> PatchMerge x2):
#   Stage 0:  16x128  (1024/64  x  128/1)
#   Stage 1:   8x64
#   Stage 2:   4x32   (bottleneck) -- 128 token
#
# LOSS: MRSTFT(1.0) + Encodec disc (adv=0.1, fm=5.0)
# DATA: FMA-medium (19,922 train tracks)
# ============================================================
#SBATCH --job-name=K13_ps64x1_3s_e256_disc_sao_exact
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-08:00:00
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
    "models.model.encoder.depths=[2,6,2]" \
    "models.model.encoder.num_heads=[8,16,32]" \
    "models.model.decoder.depths=[2,6,2]" \
    "models.model.decoder.num_heads=[32,16,8]" \
    models.model.latent_channels=8 \
    data.train_dataset.cac=false \
    data.eval_dataset.cac=false \
    data.demo.istft_params.cac=false \
    data.model_channels=2 \
    "data.train_dataset.custom_metadata_kwargs.subset=medium" \
    "data.eval_dataset.custom_metadata_kwargs.subset=medium" \
    trainer.trainer.epochs=120 \
    trainer.trainer.kl_annealing_epochs=0 \
    trainer.loss_config.bottleneck.weights.kl=1e-4 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.optimizer.lr=1e-3 \
    "+trainer.optimizer.weight_decay_exclude_1d=true" \
    "+trainer.disc_optimizer._target_=torch.optim.AdamW" \
    "+trainer.disc_optimizer.lr=3e-4" \
    "+trainer.disc_optimizer.betas=[0.8,0.99]" \
    "+trainer.disc_optimizer.weight_decay=1e-3" \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=false" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/K13_ps64x1_3s_e256_disc_sao_exact \
    trainer.wandb.name=K13_ps64x1_3s_e256_disc_sao_exact \
    "+trainer.loss_config.discriminator.type=encodec" \
    "+trainer.loss_config.discriminator.config.filters=64" \
    "+trainer.loss_config.discriminator.config.n_ffts=[2048,1024,512,256,128]" \
    "+trainer.loss_config.discriminator.config.hop_lengths=[512,256,128,64,32]" \
    "+trainer.loss_config.discriminator.config.win_lengths=[2048,1024,512,256,128]" \
    "+trainer.loss_config.discriminator.weights.adversarial=0.1" \
    "+trainer.loss_config.discriminator.weights.feature_matching=5.0"
