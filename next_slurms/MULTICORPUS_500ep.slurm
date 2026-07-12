#!/bin/bash
# ============================================================
# MULTICORPUS_500ep: identical to MULTICORPUS_baseline (16 GPU = 4 nodes × 4 A100,
# global batch 128, LARGE2 recipe on the multi-corpus dataset), but trained for
# 500 epochs instead of 100.
#
#   WHY inv_gamma=200000 (LARGE2 verbatim, NOT the 36000 of the 100-epoch run):
#   500 epochs × 956 steps/epoch = 478,000 total steps ≈ LARGE2's 527,000 steps.
#   Since the step budget now matches LARGE2, the ORIGINAL inv_gamma=200000
#   reproduces LARGE2's LR decay almost exactly (end LR (1+478k/200k)^-0.5 = 0.54×peak
#   vs LARGE2's 0.52). We override it here to shadow the YAML's 36000 (which is
#   calibrated for the 95.6k-step / 100-epoch run and would over-decay at 478k).
#
#   detach_warmup_steps stays 25000 (absolute, from the YAML). save_every_n_epochs
#   raised to 20 to bound checkpoint disk (save_top_k=-1: 25 ckpts × 2.9GB ≈ 72GB
#   instead of 100 × 2.9GB ≈ 290GB at every=5).
#
#   DISTINCT namespace (job-name / ckpt_dir / run_id / W&B) so it does NOT collide
#   with the 100-epoch MULTICORPUS_baseline run. Starts fresh unless a last.ckpt of
#   its OWN namespace exists. Wall-clock ~2.5-2.8 days at 16 GPU → fits one lprod job.
# ============================================================
#SBATCH --job-name=MULTICORPUS_500ep
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=3-12:00:00             # lprod cap. ~2.5-2.8 days expected for 500 ep @ 16 GPU;
                                      # billed on actual time, exits at epoch 500. + auto-resume.
#SBATCH --requeue                     # node-failure insurance + auto-resume from last.ckpt
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

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_500ep"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_500ep"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_baseline \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        trainer.trainer.epochs=500 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=20 \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_500ep \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_500ep \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_500ep"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
