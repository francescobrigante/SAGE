#!/bin/bash
# ============================================================
# FT_MULTICORPUS_swiglu_ep359: FINAL decoder-finetune of the multicorpus model.
#   MULTICORPUS_swiglu_500ep @ep359 (best pretrain checkpoint) → decoder-finetune
#   with the WavTokenizer discriminator + zero-init ResidualPostNet.
#
#   RECIPE = FT_LARGE2 (= FT7 + postnet), re-pointed to the multicorpus swiglu
#   model and run on the multi-corpus dataset at 16 GPU. The FT recipe is IDENTICAL
#   to FT_LARGE2; only the scale, dataset, init source and two knobs differ:
#     - init autoencoder weights (EMA) from MULTICORPUS_swiglu_500ep/epoch_359
#       (strict=False: postnet keys absent → stay zero-init = identity at step 0).
#     - encoder FROZEN; discriminator FROM SCRATCH (init_from_disc=false);
#       gen_lr=1e-4 (AdamW, pretrain 1e-3 /10); disc_lr=2e-4; kl=0; FP32.
#     - mrmel=0.5 and ema_decay=0.9998 carried over from the pretrain recipe.
#
#   inv_gamma=200000 (SAME as the swiglu 500ep pretrain — NOT rescaled): mirrors the
#   FT_LARGE2 precedent, which kept LARGE2's inv_gamma unchanged and simply ran
#   longer. The scheduler restarts fresh (init_from = weights only) and decays over
#   1500 ep to ~0.47×peak; best checkpoint picked mid-run via offline eval.
#
#   LONG run: epochs=1500 (3× the pretrain's 500-epoch budget). save every 25 ep
#   (offline recon/MAEB eval cadence); val every 2 ep (FMA-only, fast → dense W&B
#   curve). Pick the best checkpoint later, then run an FT8-style cooldown from it.
#
#   NOTE (drop_path on the frozen encoder): freeze_encoder sets requires_grad=False
#   but NOT .eval(), so the encoder's timm DropPath (drop_path_rate=0.1) stays active
#   in train mode → the decoder trains on latents perturbed by stochastic depth,
#   deterministic at inference. This is IDENTICAL to FT_LARGE2 (which produced SAGE),
#   kept here for faithful reproduction. LayerNorm (not BatchNorm) → no running-stat
#   drift; the mismatch is "train noisy / infer clean" (the benign direction).
#
#   GRACEFUL TIMEOUT (method b): #SBATCH --signal=SIGUSR1@120 + PL's default
#   SLURMEnvironment(auto_requeue=True) → on SIGUSR1 PL saves a full hpc_ckpt_*.ckpt
#   (EMA injected via EMACallback.on_save_checkpoint) and `scontrol requeue`s the job.
#   The resume block restores from the FRESHEST of {hpc_ckpt_*, checkpoints/last.ckpt}
#   across all timestamped run dirs. 1500 ep ≈ 7-8 days at 16 GPU → auto-chains beyond
#   the 4-day lprod cap with near-zero progress loss.
#
#   sbatch next_slurms/FT_MULTICORPUS_swiglu_ep359.slurm
# ============================================================
#SBATCH --job-name=FT_MULTICORPUS_swiglu_ep359
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=4-00:00:00             # full lprod cap → fewest requeues for a ~7-8 day run
#SBATCH --requeue                     # required for PL auto-requeue (+ node-failure insurance)
#SBATCH --signal=SIGUSR1@120          # SIGUSR1 to the srun step 120s before TIMEOUT → PL saves + requeues
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
export HF_HUB_OFFLINE=1                 # no teacher/CLAP in decoder-FT → stay fully offline

# Multi-node rendezvous: pin a MASTER_PORT distinct from the other multicorpus jobs.
export MASTER_PORT=29558

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

# ─── Source weights: MULTICORPUS_swiglu_500ep @ep359 (best pretrain checkpoint) ─
SRC_CKPT="$WORK/C-VAE/runs/MULTICORPUS_swiglu_500ep/2026-07-03_14-43-41/checkpoints/epoch_359.ckpt"
if [ ! -f "$SRC_CKPT" ]; then
    echo "[FATAL] source checkpoint not found: $SRC_CKPT" >&2
    exit 1
fi
echo "[INIT] decoder-finetune init from swiglu ep359 (EMA): $SRC_CKPT"

# ─── W&B resume (fresh run; id persisted by train.py to .run_ids/<name>) ──────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_MULTICORPUS_swiglu_ep359"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} ──────────
RUN_BASE="$WORK/C-VAE/runs/FT_MULTICORPUS_swiglu_ep359"
pick_ckpt() {
    ls -t "$RUN_BASE"/*/hpc_ckpt_*.ckpt "$RUN_BASE"/*/checkpoints/last.ckpt 2>/dev/null | head -1
}
LATEST_CKPT=$(pick_ckpt)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop (transient crashes; TIMEOUT is handled by signal→requeue) ─────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        models=swin_real_swiglu_xsa \
        models.model.decoder.use_postnet=true \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        "+init_from=$SRC_CKPT" \
        +init_from_ema=true \
        +init_from_disc=false \
        trainer.trainer.freeze_encoder=true \
        trainer.trainer.warmup_steps=0 \
        trainer.trainer.epochs=1500 \
        trainer.trainer.check_val_every_n_epoch=2 \
        trainer.trainer.save_every_n_epochs=25 \
        trainer.loss_config.bottleneck.weights.kl=0.0 \
        trainer.loss_config.spectral.weights.stft_mse=1.0 \
        trainer.loss_config.spectral.weights.mrstft=0.0 \
        trainer.loss_config.mrmel.weights.mrmel=0.5 \
        trainer.optimizer.lr=1e-4 \
        trainer.scheduler.inv_gamma=200000 \
        "++trainer.disc_optimizer._target_=torch.optim.AdamW" \
        "++trainer.disc_optimizer.lr=2e-4" \
        "++trainer.disc_optimizer.betas=[0.8,0.99]" \
        "++trainer.disc_optimizer.weight_decay=1e-3" \
        "++trainer.loss_config.discriminator.type=wavtokenizer" \
        "++trainer.loss_config.discriminator.config.loss_type=rpgan" \
        "++trainer.loss_config.discriminator.config.periods=[2,3,5,7,11]" \
        "++trainer.loss_config.discriminator.config.fft_sizes=[2048,1024,512]" \
        "++trainer.loss_config.discriminator.config.preprocess=true" \
        "++trainer.loss_config.discriminator.weights.adversarial=0.1" \
        "++trainer.loss_config.discriminator.weights.feature_matching=0.2" \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.trainer.strategy=ddp_find_unused_parameters_true \
        trainer.trainer.precision="32" \
        "+trainer.trainer.compile=false" \
        "++trainer.trainer.ema_decay=0.9998" \
        trainer.wandb.use_wandb=true \
        trainer.trainer.ckpt_dir=checkpoints/FT_MULTICORPUS_swiglu_ep359 \
        trainer.wandb.name=FT_MULTICORPUS_swiglu_ep359 \
        $CKPT_ARG

    EXIT=$?
    [ $EXIT -eq 0 ] && break

    echo "[$(date '+%T')] Attempt $attempt failed (exit $EXIT), retrying in 15s..."
    sleep 15
    attempt=$((attempt + 1))

    if [ -f "$RUN_ID_FILE" ]; then
        export WANDB_RESUME=must
        export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    fi
    LATEST_CKPT=$(pick_ckpt)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
