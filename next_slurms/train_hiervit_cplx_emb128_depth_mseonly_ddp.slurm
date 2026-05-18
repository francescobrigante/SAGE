#!/bin/bash
# ============================================================
# HierViT Complex VAE — 2 GPU DDP, 100k steps on FMA medium
# MSE-ONLY + dimension-preserving stem (embed_dim=128) + deeper depths
#
# Identical to train_hiervit_cplx_emb128_mseonly_ddp.slurm except:
#   - encoder.depths: [2, 2, 6, 2] → [2, 2, 6, 4]    (14 blocks, was 12)
#   - decoder.depths: [2, 6, 2, 2] → [4, 6, 2, 4]    (16 blocks, was 12)
#       asimmetria pro-ricostruzione: ultimo stage del decoder a 4 blocchi
#       (alla risoluzione più alta, subito prima del PatchUnembed).
#
# drop_path schedule si adatta automaticamente al numero di blocchi.
# Reconstruction loss: ONLY complex MSE in STFT space; mrmel disabled.
# ============================================================

#SBATCH --job-name=train_hiervit_cplx_emb128_depth_mseonly_ddp
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
mkdir -p checkpoints/hiervit_cplx_emb128_depth_mseonly_ddp

# Single-thread BLAS per DataLoader worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# DDP / NCCL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Compute-node internet access via CINECA squid proxy.
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'

# W&B
export WANDB_ENTITY=ar_spectra
export WANDB_TAGS="$SLURM_JOB_NAME,hiervit,cplx,fma_medium,emb128,deeper"
export WANDB_DIR=$WORK/lcerovaz/C-VAE/.wandb_cache
mkdir -p $WANDB_DIR

cd $WORK/lcerovaz/C-VAE
source .venv/bin/activate

echo "============================================================"
echo "HierViT Complex VAE — 2 GPU DDP, 100k steps"
echo "MSE-ONLY + emb128 + enc[2,2,6,4]/dec[4,6,2,4]"
echo "============================================================"

srun python train.py \
    models=hiervit_cplx_x64 \
    models.model.hier_vit.embed_dim=128 \
    models.model.encoder.num_heads=[8,16,32,64] \
    models.model.decoder.num_heads=[64,32,16,8] \
    models.model.encoder.depths=[2,2,6,4] \
    models.model.decoder.depths=[4,6,2,4] \
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
    trainer.loss_config.mrmel.weights.mrmel=0.0 \
    trainer.trainer.ckpt_dir=checkpoints/hiervit_cplx_emb128_depth_mseonly_ddp \
    trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
    trainer.trainer.strategy=ddp_find_unused_parameters_false \
    trainer.trainer.precision="bf16-mixed" \
    "+trainer.trainer.compile=true" \
    trainer.wandb.use_wandb=true \
    trainer.wandb.project=c-vae \
    trainer.wandb.name=HierViT_cplx_emb128_depth_mseonly_2GPU_100k
