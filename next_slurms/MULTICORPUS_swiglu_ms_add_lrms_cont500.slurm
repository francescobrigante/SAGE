#!/bin/bash
# ============================================================
# MULTICORPUS_swiglu_ms_add_lrms_cont500 — PROSECUZIONE di
# MULTICORPUS_swiglu_ms_add_lrms da epoch_219 fino a epoch_499.
#
# PERCHE' SERVE UNO SLURM NUOVO E NON SI RILANCIA L'ORIGINALE.
# La run originale (job 51118878) NON e' arrivata a 300 epoche: e' morta a ~e225
# con exit 137 (SIGKILL) dopo 42.29 h, e i tentativi del retry loop sono morti in
# ~1 min l'uno (step .1 FAILED 1:0, .2/.3/.4 CANCELLED 0:15). Lo stato COMPLETED
# del job e' l'exit dello script batch, non del training.
# Conseguenza: **last.ckpt e' TRONCATO** — 67.108.864 byte = esattamente 64 MiB,
# `unzip -l` da' rc=9 "End-of-central-directory signature not found".
# L'auto-resume dell'originale fa `ls -t .../last.ckpt | head -1` e prenderebbe
# proprio quel file => torch.load fallisce all'avvio. Lo stesso glob e' ripetuto
# dentro il retry loop, quindi non basta correggere un punto solo.
#
# L'ultimo checkpoint INTEGRO e' epoch_219.ckpt: 3.000.066.464 byte, `unzip -t` OK,
# 3950 file nell'archivio (identico per struttura a un checkpoint sano noto).
# Contiene lo stato completo: pesi, ottimizzatori, scheduler, EMA.
#
# COSA CAMBIA rispetto a MULTICORPUS_swiglu_ms_add_lrms.slurm — SOLO QUESTO:
#   1. trainer.trainer.epochs 300 -> 500
#   2. selettore di checkpoint robusto (pick_ckpt) al posto del glob su last.ckpt
#   3. --time 2-00:00:00 -> 3-00:00:00  (280 epoche x 11.214 min = 52.3 h MISURATE;
#      con 48 h e' morta la prima volta. Billing sul tempo effettivo: il margine
#      non costa. Cap di boost_qos_lprod = 4-00:00:00, verificato con sacctmgr)
#   4. tracciamento del preambolo (solo stderr, non tocca il training)
# La LOSS, l'architettura, lo scheduler, il batch, i nodi e il namespace sono
# INVARIATI: e' la stessa run che prosegue, non un arm nuovo.
#
# W&B: namespace invariato => .run_ids/MULTICORPUS_swiglu_ms_add_lrms (= vgdto66q)
# viene riletto e la run RIPRENDE con WANDB_RESUME=must. Nota attesa: la history di
# vgdto66q arriva a e225 mentre si riparte da e220, quindi le epoche 220-225 nuove
# verranno scartate da W&B come step non monotoni e resteranno visibili le vecchie.
# Da e226 in poi logga normalmente. Non tocca ne' training ne' checkpoint.
#
# COSTO — MISURATO (11.214 min/epoca su epoch_019 -> epoch_219 della run originale;
# billing 128 local h per ora di wall = 32 x R(=1.0) x 4 nodi):
#   e219 -> e499 = 280 epoche = 52.3 h = 6.698 local h.
# ATTENZIONE BUDGET: al 2026-08-31 restano ~5.916 local h su IscrC_MM (saldo legge
# 68.504/80.000 ma NON include ancora la run lrms_mse_side finita oggi, 5.568 h
# misurati). Servono ~780 h in piu' di quelle disponibili: la run arriverebbe a
# ~e465 portando l'account a ~101%, stato in cui NON si sottomettono piu' job
# (e' quello di IscrC_AHNetBio). Traguardo massimo che lascia spazio agli eval:
# epochs=459 (240 epoche, 44.9 h, 5.742 h). Scelta consapevole dell'utente.
#
#   sbatch next_slurms/MULTICORPUS_swiglu_ms_add_lrms_cont500.slurm
# ============================================================
#SBATCH --job-name=MULTICORPUS_swiglu_ms_add_lrms_cont500
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=24:00:00                # SEGMENTO 3 (04/09): da last.ckpt (~e408) a e499 =
#                                      # 91 ep x 11.386 min = 17.3h MISURATE + margine.
#                                      # Niente manutenzioni fino al 15/10 (e quella tocca solo dcgp).
#SBATCH --requeue                      # node-failure insurance + auto-resume
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Tracciamento preambolo (solo stderr, non tocca il training) -------------
echo "[$(date '+%T')] batch step avviato su $(hostname) - job $SLURM_JOB_ID"
export PS4='+ [$(date +%T)] riga ${LINENO}: '
set -x

module load profile/deeplrn cineca-ai/4.3.0

# Compute on IscrC_MM, but repo/venv/datasets live in IscrC_AHNetBio ($WORK/$FAST
# env now point to IscrC_MM after the budget switch → pin them explicitly).
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
export MASTER_PORT=29586

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

RUN_DIR_GLOB="$WORK/C-VAE/runs/MULTICORPUS_swiglu_ms_add_lrms"

# ─── Auto-resume W&B (namespace invariato: riprende vgdto66q) ────────────────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/MULTICORPUS_swiglu_ms_add_lrms"

if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Selettore di checkpoint robusto ────────────────────────────────────────
# Scorre last.ckpt e gli epoch_*.ckpt dal piu' recente e restituisce il PRIMO che
# passa la validazione. Sostituisce `ls -t .../last.ckpt | head -1`, che qui
# prenderebbe il last.ckpt troncato del crash del 2026-08-02.
# `unzip -l` legge solo il central directory di fine file: 0.07 s su un ckpt sano,
# 0.14 s per rifiutare quello troncato (MISURATO). Il CRC completo (`unzip -t`)
# costerebbe 15 s e non serve: la troncatura si vede dal central directory mancante.
pick_ckpt() {
    local c
    for c in $(ls -t "$RUN_DIR_GLOB"/*/checkpoints/last.ckpt \
                     "$RUN_DIR_GLOB"/*/checkpoints/epoch_*.ckpt 2>/dev/null); do
        if unzip -l "$c" >/dev/null 2>&1; then
            echo "$c"; return 0
        fi
        echo "[RESUME] SCARTATO (archivio non valido): $c" >&2
    done
    return 1
}

LATEST_CKPT=$(pick_ckpt)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
else
    echo "[FATAL] nessun checkpoint integro sotto $RUN_DIR_GLOB" >&2
    exit 1
fi

# ─── Retry loop ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS=8
attempt=1
set +x
echo "[$(date '+%T')] preambolo completato in $SECONDS s, entro nel retry loop"
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --export=ALL --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_swiglu \
        data=multicorpus \
        data.train_dataloader.batch_size=128 \
        data.eval_dataloader.batch_size=128 \
        trainer.trainer.epochs=500 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=20 \
        trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0 \
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
        trainer.trainer.num_gpus=4 \
        "++trainer.trainer.num_nodes=4" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=MULTICORPUS_swiglu_ms_add_lrms \
        trainer.trainer.ckpt_dir=checkpoints/MULTICORPUS_swiglu_ms_add_lrms \
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
    # Stessa guardia anche qui: se il crash ha troncato il last.ckpt appena
    # scritto, si riparte dall'epoch_*.ckpt integro piu' recente invece di
    # ritentare all'infinito su un archivio rotto.
    LATEST_CKPT=$(pick_ckpt)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
