#!/bin/bash
# ============================================================
# FT_MULTICORPUS_swiglu_ep359_ms_add: STEREO-COLLAPSE ablation ARM "ADD".
#   IDENTICAL to FT_MULTICORPUS_swiglu_ep359 in every respect EXCEPT the recon loss:
#     - stft_mse  = 1.0   (ComplexMSE on STFT — KEPT INTACT; best-in-class Mid)
#     - mrstft_sd = 1.0   (SAO-faithful sum/difference + L/R MR-STFT — ADDED)
#
#   HYPOTHESIS (doc raccomandazione 1+3): SAGE sounds near-mono because a complex-STFT
#   MSE is a quadratic form → by the parallelogram identity M/S is a no-op, so the
#   low-energy Side channel gets ~0 loss budget and collapses (w_rec ≈ 0.15·w_ref,
#   broadband). SAGE already has the BEST Mid of the 5 models (MRSTFT_M 1.25), so we
#   ADD the missing stereo-aware term next to the untouched ComplexMSE instead of
#   replacing it — repairing Side without risking the Mid. mrstft_sd's per-resolution
#   spectral convergence normalizes by target energy PER CHANNEL → gives Side its own
#   budget. This is the SAFE arm; FT_..._ms_replace is the exploratory (MSE-off) arm.
#
#   Everything else (init from swiglu ep359 EMA, frozen encoder, WavTokenizer disc
#   from scratch, gen_lr=1e-4, kl=0, postnet, inv_gamma=200000, ema=0.9998, FP32,
#   epochs=1500 + val/2 + save/25, mrmel=0.5) is COPIED VERBATIM from the mse FT so the
#   ONLY variable vs baseline is the recon loss → clean single-knob ablation at
#   identical LR schedule (compare ms_add@e399 vs baseline@e399). See the header of
#   next_slurms/FT_MULTICORPUS_swiglu_ep359.slurm for the full recipe rationale.
#
#   BOUNDED 2-DAY EXPERIMENT: --time=2-00:00:00. Graceful SIGUSR1@120 → PL saves a full
#   checkpoint (last.ckpt / hpc_ckpt_*) ~120 s before TIMEOUT so no progress is lost at
#   the 2-day mark; --requeue lets it auto-chain if you want to extend. Stop with
#   `scancel` after the e199/e399 verdict.
#
#   sbatch next_slurms/FT_MULTICORPUS_swiglu_ep359_ms_add.slurm
# ============================================================
#SBATCH --job-name=FT_MULTICORPUS_swiglu_ep359_ms_add
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=2-00:00:00             # bounded 2-day ablation block
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

# Multi-node rendezvous: pin a MASTER_PORT distinct from the mse FT (29558), the
# mrstft FT (29560), and the replace arm (29564) → all can co-schedule in parallel.
export MASTER_PORT=29562

export http_proxy='http://login05:3150'
export https_proxy='http://login05:3150'
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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_MULTICORPUS_swiglu_ep359_ms_add"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} ──────────
RUN_BASE="$WORK/C-VAE/runs/FT_MULTICORPUS_swiglu_ep359_ms_add"
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
        trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0 \
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
        trainer.trainer.ckpt_dir=checkpoints/FT_MULTICORPUS_swiglu_ep359_ms_add \
        trainer.wandb.name=FT_MULTICORPUS_swiglu_ep359_ms_add \
        $CKPT_ARG \
        "$@"

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
