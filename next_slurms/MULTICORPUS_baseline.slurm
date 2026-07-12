#!/bin/bash
# ============================================================
# MULTICORPUS_baseline: LARGE2 recipe on the multi-corpus dataset, IscrC_MM, 16 GPU.
#
#   Recipe lives in config/experiment/MULTICORPUS_baseline.yaml (LARGE2 verbatim +
#   inv_gamma 200000→36000 and detach 25000→5000 rescaled for global batch 128).
#   See TRAIN_MULTICORPUS.md for the full rationale and the delta-vs-LARGE2 table.
#
#   Dataset: FMA-full(large) + Jamendo split-0 + M4Singer (~122k/epoch). data=multicorpus.
#   Global batch 128 = 16 GPU (4 nodes × 4 A100) × per-device 8 (= LARGE2 footprint).
#
#   PREREQUISITES (first multi-node run in this repo) — ✅ BOTH SATISFIED:
#     1) train.py passes `num_nodes` to Trainer and divides the global batch by the
#        world size (num_nodes × num_gpus). DONE (train.py:295-297, 609).
#     2) 2-node smoke (job 48153463, scripts/smoke_multicorpus_2node.slurm) PASSED:
#        NCCL nranks=8 over IB, split 128/8=16, recipe composed, train+val+EMA ran,
#        filelist cache built & persisted, exit 0. Production run reloads cache instantly.
# ============================================================
#SBATCH --job-name=MULTICORPUS_baseline
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=1-00:00:00             # lprod cap (4 days). ~12-20h expected; full-length
                                      # files + partial-read I/O is the unknown → max headroom,
                                      # billed on actual time, exits at epoch 100. + auto-resume.
#SBATCH --requeue                     # node-failure insurance + auto-resume from last.ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO                # first multi-node run: keep verbose; lower to WARN later
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

# Multi-node rendezvous: Lightning's SLURMEnvironment derives MASTER_ADDR from the
# node list; we pin a MASTER_PORT to avoid collisions across concurrent jobs.
export MASTER_PORT=29555

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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_baseline"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_baseline"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
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
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_baseline"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
