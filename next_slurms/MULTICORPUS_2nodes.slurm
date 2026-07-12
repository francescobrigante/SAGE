#!/bin/bash
# ============================================================
# MULTICORPUS_2nodes: identical recipe to MULTICORPUS_baseline, on 2 nodes × 4 A100
# = 8 GPU (instead of 4 nodes × 4 = 16). Global batch stays 128 → per-device 16.
#
#   WHY: on a jammed cluster (only ~13 fully-idle nodes, most "mix" nodes are
#   GPU-full), a 2-node job needs just 2 free nodes → far easier to backfill than
#   a 4-node one, so it starts sooner. Same math (global batch 128, DDP world 8),
#   ~2× wall-clock. Per-device 16 is smoke-validated (job 48153463 ran exactly this).
#
#   DISTINCT namespace (job-name / ckpt_dir / run_id / W&B) from MULTICORPUS_baseline
#   so the two can sit in the queue together without colliding: whichever STARTS
#   FIRST wins → cancel the other. Do NOT let both run at once (double billing).
#
#   Recipe = config/experiment/MULTICORPUS_baseline.yaml (LARGE2 verbatim; inv_gamma
#   36000, detach as set in the YAML). See TRAIN_MULTICORPUS.md.
# ============================================================
#SBATCH --job-name=MULTICORPUS_2nodes
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 8 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=4-00:00:00             # lprod cap. ~24-40h expected at 8 GPU (2× the 16-GPU run);
                                      # billed on actual time, exits at epoch 100. + auto-resume.
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
# node list; pin a MASTER_PORT (distinct from the 16-GPU job to avoid any clash).
export MASTER_PORT=29556

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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_2nodes"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_2nodes"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        trainer.trainer.epochs=100 \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=2" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_2nodes \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_2nodes \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_2nodes"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
