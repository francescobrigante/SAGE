#!/bin/bash
# ============================================================
# FT_MULTICORPUS_swiglu_muon_ep499: decoder-finetune of the MUON pretrain.
#   Source: MULTICORPUS_swiglu_muon_500ep @ epoch_499 (muon variant) → decoder-FT
#   with the WavTokenizer discriminator + zero-init ResidualPostNet.
#
#   RECIPE = identical to FT_MULTICORPUS_swiglu_ep359 (= FT_LARGE2 + postnet), with
#   ONE structural difference: the generator optimizer stays **MuonAdamW** (not plain
#   AdamW), because we are finetuning a muon-pretrained model and want the same
#   optimizer family. Both LR groups are divided by 10 vs the pretrain (FT_LARGE2's
#   gen-lr /10 rule, applied to BOTH param groups so their ratio is preserved):
#     * muon_lr   1e-2 → 1e-3   (2-D Swin matrices, Newton-Schulz orthogonalized)
#     * adamw_lr  1e-3 → 1e-4   (1-D params: bias / norm / embedding / CPB / heads)
#     * lr        1e-3 → 1e-4   (fallback, only used if muon_lr/adamw_lr were None)
#   All other muon knobs (momentum 0.95, nesterov, ns_steps 5, matrix_lr_scale,
#   weight_decay_exclude_1d) copied verbatim from the pretrain experiment.
#
#   MuonAdamW is wired to the model via configure_optimizers (autoencoders.py): it
#   keys off _target_==MuonAdamW and passes model=self.autoencoder, and it SKIPS
#   frozen params — so overriding the optimizer via ++ (WITHOUT loading the muon
#   experiment) is correct AND avoids instantiating the CLAP teacher / aux optimizer
#   (no distill in a decoder-FT → fully offline-safe).
#
#   Everything else is FT_MULTICORPUS_swiglu_ep359 verbatim: init from muon ep499
#   EMA weights, encoder FROZEN, disc FROM SCRATCH (init_from_disc=false), kl=0,
#   mrmel=0.5, mse recon (stft_mse=1.0 / mrstft=0.0), ema=0.9998, FP32, postnet
#   zero-init, inv_gamma=200000 (same as the muon pretrain run), 1500 ep, val/2,
#   save/25, graceful SIGUSR1 requeue. Distinct namespace + MASTER_PORT so it can
#   co-schedule in parallel.
#
#   NOTE: on lprod the IscrC_MM account is capped at 256 CPU total. Two 4-node FTs
#   (128 CPU each) already saturate it → to RUN this one on lprod, stop one of the
#   running FTs first (e.g. the worse-performing mrstft) to free 128 CPU + 4 nodes.
#
#   sbatch next_slurms/FT_MULTICORPUS_swiglu_muon_ep499.slurm
# ============================================================
#SBATCH --job-name=FT_MULTICORPUS_swiglu_muon_ep499
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

# Multi-node rendezvous: pin a MASTER_PORT distinct from the other multicorpus jobs
# (mse FT=29558, mrstft FT=29560, muon pretrain=29557).
export MASTER_PORT=29561

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

# ─── Source weights: MULTICORPUS_swiglu_muon_500ep @ep499 (muon pretrain) ──────
SRC_CKPT="$WORK/C-VAE/runs/MULTICORPUS_swiglu_muon_500ep/2026-07-04_16-15-14/checkpoints/epoch_499.ckpt"
if [ ! -f "$SRC_CKPT" ]; then
    echo "[FATAL] source checkpoint not found: $SRC_CKPT" >&2
    exit 1
fi
echo "[INIT] decoder-finetune init from muon ep499 (EMA): $SRC_CKPT"

# ─── W&B resume (fresh run; id persisted by train.py to .run_ids/<name>) ──────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_MULTICORPUS_swiglu_muon_ep499"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} ──────────
RUN_BASE="$WORK/C-VAE/runs/FT_MULTICORPUS_swiglu_muon_ep499"
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
        "++trainer.optimizer._target_=ar_spectra.training.optimizers.MuonAdamW" \
        "++trainer.optimizer.lr=1e-4" \
        "++trainer.optimizer.muon_lr=1e-3" \
        "++trainer.optimizer.adamw_lr=1e-4" \
        "++trainer.optimizer.weight_decay=1e-4" \
        "++trainer.optimizer.weight_decay_exclude_1d=true" \
        "++trainer.optimizer.betas=[0.9,0.98]" \
        "++trainer.optimizer.eps=1e-8" \
        "++trainer.optimizer.muon_momentum=0.95" \
        "++trainer.optimizer.muon_nesterov=true" \
        "++trainer.optimizer.ns_steps=5" \
        "++trainer.optimizer.matrix_lr_scale=true" \
        "++trainer.optimizer.log_param_coverage=true" \
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
        trainer.trainer.ckpt_dir=checkpoints/FT_MULTICORPUS_swiglu_muon_ep499 \
        trainer.wandb.name=FT_MULTICORPUS_swiglu_muon_ep499 \
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
