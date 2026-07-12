#!/bin/bash
# ============================================================
# MULTICORPUS_swiglu_muon_1000ep: CONTINUATION of MULTICORPUS_swiglu_muon_500ep.
# NOT a fresh run — resumes the SAME local run and the SAME W&B run (id c02nf4iq)
# from its last.ckpt (epoch 499 / global_step 717000) and trains to epoch 1000.
#
#   SAME NAMESPACE as the 500ep run (this is the whole point of "continue"):
#     * run_id file  .run_ids/MULTICORPUS_swiglu_muon_500ep  → W&B resume=must (c02nf4iq)
#     * ckpt_dir     checkpoints/MULTICORPUS_swiglu_muon_500ep
#     * resume glob  runs/MULTICORPUS_swiglu_muon_500ep/*/checkpoints/{hpc_ckpt_*,last.ckpt}
#     * wandb.name   MULTICORPUS_swiglu_muon_500ep      (same chart continues)
#   Only the SLURM job-name differs (…_1000ep) for sacct clarity.
#
#   inv_gamma stays 200000 (VERBATIM from the 500ep run): the scheduler's last_epoch
#   is restored from the checkpoint, so the InverseLR decay CONTINUES with no
#   discontinuity (≈0.54×peak at ep500 → ≈0.42×peak at ep1000). Changing it would
#   introduce a jump. Everything else (Muon lr, kl, mrmel, disc, ema, batch 128,
#   16 GPU) is identical — same run, more epochs, not a new recipe.
#
#   GRACEFUL TIMEOUT (added vs the 500ep slurm): 500 more epochs ≈ 2.5-2.8 days at
#   16 GPU, so the 3-day cap will likely TIMEOUT before ep1000. #SBATCH
#   --signal=SIGUSR1@120 + PL's SLURMEnvironment(auto_requeue=True) → on SIGUSR1 PL
#   writes a full hpc_ckpt_*.ckpt (EMA injected) and `scontrol requeue`s the job;
#   the resume block below restores from the FRESHEST of {hpc_ckpt_*, last.ckpt}
#   across all timestamped run dirs → near-zero progress loss on requeue (vs up to
#   20 epochs lost with save_every_n_epochs alone). Stop early if it stops improving.
#
#   sbatch next_slurms/MULTICORPUS_swiglu_muon_1000ep.slurm
# ============================================================
#SBATCH --job-name=MULTICORPUS_swiglu_muon_1000ep
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=3-00:00:00             # 3-day cap; billed on actual time, exits at epoch 1000.
#SBATCH --requeue                     # node-failure insurance + auto-resume from ckpt
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
export HF_HUB_OFFLINE=1

# Multi-node rendezvous: Lightning's SLURMEnvironment derives MASTER_ADDR from the
# node list; pin a MASTER_PORT (distinct from the other multicorpus jobs).
export MASTER_PORT=29557

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

# ─── W&B resume: SAME run as the 500ep training (id persisted to .run_ids) ─────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_swiglu_muon_500ep"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: freshest of {hpc_ckpt_* (signal), last.ckpt (periodic)} across dirs ─
RUN_BASE="$WORK/C-VAE/runs/MULTICORPUS_swiglu_muon_500ep"
pick_ckpt() {
    ls -t "$RUN_BASE"/*/hpc_ckpt_*.ckpt "$RUN_BASE"/*/checkpoints/last.ckpt 2>/dev/null | head -1
}
LATEST_CKPT=$(pick_ckpt)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
else
    echo "[FATAL] no last.ckpt/hpc_ckpt found under $RUN_BASE — nothing to continue." >&2
    exit 1
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_swiglu_muon \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        trainer.trainer.epochs=1000 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=20 \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_swiglu_muon_500ep \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_swiglu_muon_500ep \
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
