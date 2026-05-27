#!/bin/bash
# ============================================================
# K12_ps64x1_3s_e256_disc_trb: Swin H + SAME Transformer (Config-2) + RPGAN
#   Fix rispetto alla run originale interrotta:
#     kl:             1e-5 -> 1e-4
#     gen opt:        betas=(0.9,0.95) wd=1e-4 -- SAME paper (Sec 4.2)
#     disc opt:       separato dal gen -- SAO-exact (lr=3e-4, b=(0.8,0.99), wd=1e-3)
#     wavelet_kwargs: rimosso (non in SAME) -> mfb_kwargs con use_HIL=True (PQMF conv)
#     chroma_kwargs:  aggiunto (TRB chroma, 3 scale ottave 1/5/9)
#     patched depths: rimossa override [2,2,2] -> default SAT [3,3,3]
#     time:           7h -> 18h
#
# DISCRIMINATORE (SAME Config-2 = MultiTransformerDiscriminator):
#   3 TRB-STFT      (FFT 4096/1024/128)
#   3 TRB-Patched   (patch 29/443/953, prime sizes)
#   6 conv-PQMF     (MultiFilterBankDiscriminator, periodi [1,2,3,5,7,11]) <- use_HIL=True
#   3 TRB-Chroma    (ottave 1/5/9, ChromaPretransform n_chroma=64)
#   Totale: 15 rami (= SAME "10" contati a livello componente: STFT+Patched+PQMF+Chroma)
#   Objective: rpgan
#
# CONFRONTO con K11 (SAME Config-1 convoluzionale):
#   K11: HIL disc (STFT conv + PQMF conv + Chroma conv)
#   K12: MultiTransformer disc (STFT TRB + Patched TRB + PQMF conv + Chroma TRB) <- questo
#
# GRID (PatchEmbed -> PatchMerge x2):
#   Stage 0:  16x128  (1024/64  x  128/1)
#   Stage 1:   8x64
#   Stage 2:   4x32   (bottleneck) -- 128 token
#
# LOSS: MRSTFT(1.0) + RPGAN (adv=0.1, fm=5.0)
# DATA: FMA-medium (19,922 train tracks)
# ============================================================
#SBATCH --job-name=K12_ps64x1_3s_e256_disc_trb
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
    "+trainer.optimizer.betas=[0.9,0.95]" \
    "+trainer.optimizer.weight_decay=1e-4" \
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
    trainer.trainer.ckpt_dir=checkpoints/K12_ps64x1_3s_e256_disc_trb \
    trainer.wandb.name=K12_ps64x1_3s_e256_disc_trb \
    "+trainer.loss_config.discriminator.type=transformer" \
    "+trainer.loss_config.discriminator.config.loss_type=rpgan" \
    "+trainer.loss_config.discriminator.config.stft_kwargs.enabled=true" \
    "+trainer.loss_config.discriminator.config.patched_kwargs.enabled=true" \
    "+trainer.loss_config.discriminator.config.mfb_kwargs.enabled=true" \
    "+trainer.loss_config.discriminator.config.mfb_kwargs.use_HIL=true" \
    "+trainer.loss_config.discriminator.config.chroma_kwargs.enabled=true" \
    "+trainer.loss_config.discriminator.config.chroma_kwargs.sample_rate=44100" \
    "+trainer.loss_config.discriminator.weights.adversarial=0.1" \
    "+trainer.loss_config.discriminator.weights.feature_matching=5.0"
