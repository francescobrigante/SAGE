#!/bin/bash
# ============================================================
# K11_ps64x1_3s_e256_disc_hil: Swin H + SAME Convoluzionale (HIL) + RPGAN
#   K11b relaunch (run sana, contro prior escape latent_std ~ 3):
#     kl:       1e-4 -> 5e-4  (HIL ~3x piu' aggressivo di MS-STFT: K8 kl=1e-4->std 1.5, HIL kl=1e-4->std 3)
#     disc_lr:  3e-4 -> 2e-4  (SAO-exact; rallenta HIL -> disc_loss verso equilibrio rpgan ~0.69, meno escape)
#     gen opt:  INVARIATO betas (0.9,0.98) -- betas baseline swin
#     adv/fm:   INVARIATI (0.1 / 5.0) -- fm = driver qualita' percettiva; 1a leva escalation se std resta >2
#     NB: monitora latent_std nelle prime ~15 epoch; se >2 -> fm 5.0->3.0 (o kl=1e-3). Annealing OFF.
#     time:     7h -> 18h (run completa 120 epoch)
#
# DISCRIMINATORE (SAME convoluzionale = HILDiscriminator):
#   Port esatto di stable-audio-tools (discriminators.py)
#   MS-STFT (5 scale) + PQMF FilterBank (6 periodi [1,2,3,5,7,11]) + Chroma (48-bin)
#   Objective: rpgan (Relativistic Paired GAN, softplus -- come SAME)
#   Weights: adversarial=0.1, feature_matching=5.0
#
# CONFRONTO con K13 (encodec/hinge):
#   K13: encodec disc (MS-STFT x5, hinge loss), kl=1e-4, disc_lr=3e-4
#   K11: HIL disc (STFT+PQMF+Chroma, rpgan) <- questo, kl=5e-4, disc_lr=2e-4
#   -> NB: non piu' ablation pulita (kl/disc_lr divergono); K11b ottimizzata best-performing
#
# GRID (PatchEmbed -> PatchMerge x2):
#   Stage 0:  16x128  (1024/64  x  128/1)
#   Stage 1:   8x64
#   Stage 2:   4x32   (bottleneck) -- 128 token
#
# LOSS: MRSTFT(1.0) + RPGAN (adv=0.1, fm=5.0)
# DATA: FMA-medium (19,922 train tracks)
# ============================================================
#SBATCH --job-name=K11_ps64x1_3s_e256_disc_hil_conservativo
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=1-5:00:00
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
    trainer.loss_config.bottleneck.weights.kl=5e-4 \
    trainer.loss_config.spectral.weights.stft_mse=0.0 \
    trainer.loss_config.spectral.weights.mrstft=1.0 \
    "trainer.loss_config.spectral.mrstft.config.max_loss_clamp=20.0" \
    trainer.trainer.clip_grad_norm=20.0 \
    trainer.optimizer.lr=1e-3 \
    "trainer.optimizer.weight_decay_exclude_1d=true" \
    "+trainer.disc_optimizer._target_=torch.optim.AdamW" \
    "+trainer.disc_optimizer.lr=2e-4" \
    "+trainer.disc_optimizer.betas=[0.8,0.99]" \
    "+trainer.disc_optimizer.weight_decay=1e-3" \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_true \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=false" \
    trainer.wandb.use_wandb=true \
    trainer.trainer.ckpt_dir=checkpoints/K11_ps64x1_3s_e256_disc_hil_conservativo \
    trainer.wandb.name=K11_ps64x1_3s_e256_disc_hil_conservativo \
    "+trainer.loss_config.discriminator.type=hil" \
    "+trainer.loss_config.discriminator.config.loss_type=rpgan" \
    "+trainer.loss_config.discriminator.config.filters=64" \
    "+trainer.loss_config.discriminator.config.n_ffts=[2048,1024,512,256,128]" \
    "+trainer.loss_config.discriminator.config.hop_lengths=[512,256,128,64,32]" \
    "+trainer.loss_config.discriminator.config.win_lengths=[2048,1024,512,256,128]" \
    "+trainer.loss_config.discriminator.config.n_chroma=48" \
    "+trainer.loss_config.discriminator.weights.adversarial=0.1" \
    "+trainer.loss_config.discriminator.weights.feature_matching=5.0"
