#!/bin/bash
# ============================================================
# MULTICORPUS_swiglu_lrms_mse_side — arm "STEREO-AWARE DISC + complexMSE_side"
# sull'architettura del paper (swin_real_swiglu_xsa), su multicorpus a 16 GPU.
#
# COS'E'. E' la versione multicorpus di STEREO_F: `lrms + complexMSE_side`, cioe'
# il termine A (SideComplexMSE) SENZA mrstft_side. Colma il buco dichiarato in
# STEREO_COLLAPSE_DIAGNOSIS §11.9: finora A e' misurato solo IN PRESENZA di B
# (arm D = lrms + mrstft_side + complexMSE_side), mai da solo, e mai su multicorpus.
#
# DERIVATO VERBATIM da MULTICORPUS_swiglu_lrms_disc.slurm. Cambiano SOLO tre cose:
#   1. LOSS   : + mse_side=0.5 e side_gate_db=-40.0   (2 righe)
#   2. EPOCHE : 300 -> 240
#   3. TEMPO  : 2-00:00:00 -> 2-06:00:00
# (piu' namespace e MASTER_PORT distinti, che non sono parametri).
# Tutto il resto e' IDENTICO a lrms_disc: LR SCHEDULER NON TOCCATO
# (inv_gamma=200000, InverseLR, warmup 0.999, power 0.5), batch 128, 4x4 GPU,
# save_every_n_epochs=20, fold_lrms=true, +experiment=MULTICORPUS_swiglu.
#
# PERCHE' 240 E NON 300. La schedule LR e' step-based e NON dipende dal totale:
# InverseLR._get_closed_form_lr (src/ar_spectra/training/schedulers.py:40-44) usa
# solo last_epoch e inv_gamma, con interval:step (config/trainer.yaml:36).
# Quindi epochs=240 produce una traiettoria LR IDENTICA a epochs=300 troncata a
# 240: nessun confondente introdotto, e il job si autospegne dopo epoch_239.ckpt
# invece di continuare a consumare budget.
#
# PERCHE' 2-06:00:00 E NON 2-00:00:00. Le due run `lrms` precedenti si sono
# fermate a e219 proprio per il walltime a 48 h (MISURATO: last.ckpt di
# lrms_disc e' 0 byte, scrittura troncata). A 10.96-11.21 min/epoca (MISURATO
# dai timestamp dei checkpoint) e239 arriva a 43.8-44.8 h: 48 h non bastano con
# margine. 54 h lasciano ~9 h di franchigia. Il billing e' sul tempo effettivo,
# quindi chiedere margine non costa nulla. Max di boost_qos_lprod = 4-00:00:00.
#
# COSTO — MISURATO, non stimato:
#   min/epoca con fold_lrms: 10.96 (lrms_disc) / 11.21 (ms_add_lrms), dai mtime
#   dei checkpoint epoch_019 -> epoch_219. mse_side e' una MSE sugli stessi
#   tensori gia' calcolati per stft_mse -> sovracosto atteso trascurabile (INFERITO).
#   billing = 32 x R x 4 nodi, R = max(4/4 GPU, 32/32 CPU, 480/494 GB) = 1.0
#           = 128 local h per ora di wall.
#   -> e239 a 240 epoche = 43.8-44.8 h wall = 5611-5736 local h (~1.9 giorni).
#   Budget IscrC_MM al 2026-08-29: 68503/80000 -> 11497 local h residui.
#
# COMPARABILITA' — ATTENZIONE, leggere prima di fare tabelle:
#   Le altre run NON sono tutte a e239. MISURATO in runs/*/checkpoints/:
#     lrms_disc     -> ultimo ckpt buono epoch_219   (fold_lrms=true)
#     ms_add_lrms   -> ultimo ckpt buono epoch_219   (fold_lrms=true)
#     ms_add        -> epoch_239                     (fold_lrms ASSENTE)
#     ms_replace    -> epoch_239                     (fold_lrms ASSENTE)
#   Quindi:
#   * a e219 il confronto con lrms_disc e' A UN SOLO FATTORE (+complexMSE_side):
#     e' LA replica su multicorpus del contrasto F - N. Questo e' il confronto pulito.
#   * a e239 l'epoca coincide con ms_add/ms_replace, ma quelle due non hanno
#     fold_lrms: due fattori cambiano insieme, non e' interpretabile da solo.
#   Con save_every_n_epochs=20 questa run scrive ENTRAMBI (epoch_219 e epoch_239),
#   quindi copre le due letture senza costi aggiuntivi oltre le 20 epoche finali.
#
# Infra (obbligatoria, non parametri): proxy squid su login05 (login07 e' giu');
# WORK/FAST pinnati a IscrC_AHNetBio PRIMA di ogni uso, perche' dopo il cambio
# budget le env var puntano a IscrC_MM mentre repo/venv/dataset stanno in AHNetBio.
# NOTE QOS: 16 GPU. Cap lprod per account = 256 CPU / 32 GPU / 8 nodi (VERIFICATO
# con sacctmgr) -> resta spazio per uno STEREO_* a 1 nodo/2 GPU in parallelo.
# ============================================================
#SBATCH --job-name=MULTICORPUS_swiglu_lrms_mse_side
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=2-06:00:00              # 240 ep @ 11.2 min/ep ≈ 44.8h MISURATO + margine
#SBATCH --requeue                      # node-failure insurance + auto-resume from last.ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Tracciamento preambolo -------------------------------------------------
# Il 27/08 due job sono restati ~9 min nella sola .batch senza mai lanciare srun
# e con .out a 0 byte: il primo echo era DOPO il preambolo, che era quindi un
# punto cieco totale. Ora ogni riga del preambolo finisce su stderr con l'ora,
# cosi' un eventuale blocco (Lmod? metadata Lustre? venv?) si localizza subito
# invece di presentarsi come un job "RUNNING" che non scrive nulla.
# Non tocca l'addestramento: e' solo bash tracing su stderr.
echo "[$(date '+%T')] batch step avviato su $(hostname) - job $SLURM_JOB_ID"
export PS4='+ [$(date +%T)] riga ${LINENO}: '
set -x

module load profile/deeplrn cineca-ai/4.3.0

# Compute on IscrC_MM, but repo/venv/datasets live in IscrC_AHNetBio ($WORK/$FAST
# env now point to IscrC_MM after the budget switch → pin them so all $WORK/$FAST
# references below resolve correctly).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

# Multi-node rendezvous: Lightning's SLURMEnvironment derives MASTER_ADDR from the
# node list; pin a MASTER_PORT (distinct from every other slurm in next_slurms/).
export MASTER_PORT=29607

# Proxy squid: gira su login05 (login07 e' giu' e fra login node non c'e' ssh).
# VERIFICATO il 2026-08-29 da login05: squid in ascolto su *:3150 (pid 3325783),
# api.wandb.ai/healthz -> HTTP 200 in 0.17s attraverso il proxy. I nodi di calcolo
# non hanno rete diretta: senza questo, WANDB_MODE=online blocca l'avvio.
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

# ─── Auto-resume ────────────────────────────────────────────────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_swiglu_lrms_mse_side"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_swiglu_lrms_mse_side"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
set +x
echo "[$(date '+%T')] preambolo completato in $SECONDS s, entro nel retry loop"
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    # --export=ALL e' una rete di sicurezza, non un cambio di comportamento: con
    # `sbatch` normale srun propaga gia' tutto l'ambiente. Serve se il job venisse
    # sottomesso con `sbatch --export=NONE`, nel qual caso il PYTHONPATH del modulo
    # non arriverebbe a python, il torch del modulo uscirebbe dal path e vincerebbe
    # la dir fantasma .envs/c_vae/lib/python3.11/site-packages/torch/ (solo header,
    # residuo di un install interrotto) -> ModuleNotFoundError: torch.profiler.
    srun --export=ALL --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_swiglu \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        trainer.trainer.epochs=240 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=20 \
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
        "++trainer.loss_config.mse_side.weights.mse_side=0.5" \
        "++trainer.loss_config.mse_side.config.side_gate_db=-40.0" \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_swiglu_lrms_mse_side \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_swiglu_lrms_mse_side \
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
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/MULTICORPUS_swiglu_lrms_mse_side"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
