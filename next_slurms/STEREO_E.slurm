#!/bin/bash
# ============================================================
# STEREO_E: lrms + add  (mrstft_sd COMPLETA, ricetta SAO invariata)
#
# Secondo round su FMA-MEDIUM. Domanda: fra i due candidati emersi, conviene
# investire su `add` (mrstft_sd su tutti e quattro i rami M,S,L,R) o su
# `complexMSE_side` (arm STEREO_F)? Controllo comune: STEREO_N.
#
# Differenza da STEREO_N: UNA riga — mrstft_sd.weights.mrstft_sd 0.0 -> 1.0.
# Tutto il resto (w_ms=1.0, w_lr=1.0, w_mid=null, w_side=null, side_gate_db=null)
# resta al default di trainer.yaml:161-181, cioe' SAO invariato: e' esattamente
# la ricetta di MULTICORPUS_swiglu_ms_add_lrms.slurm:96-97, quindi questo arm e'
# confrontabile SIA con N (a variabile singola) SIA con ms_add_lrms e219 (stessa
# loss, dataset e scala diversi).
#
# NOTA sull'asimmetria col gemello STEREO_F: qui side_gate_db resta null perche'
# e' il default SAO usato dalla famiglia multicorpus; in STEREO_F vale -40 perche'
# replica la config di mse_side gia' misurata in STEREO_D. Voluto: ciascun arm e'
# allineato al proprio precedente, non l'uno all'altro.
#
# Derivato da MULTICORPUS_swiglu_ms_add_lrms.slurm — stessa ricetta e stessa
# architettura del paper (swin_real_swiglu_xsa: SwiGLU + XSA) — cambiando SOLO:
#   * data=multicorpus            -> data=fma, fma_subset=medium (19.922 tracce train)
#   * 4 nodi x 4 GPU (16)         -> 1 nodo x 2 GPU
#   * batch_size 128              -> 16 (default FMA). Con 128 su 2 GPU l'epoca
#                                    sarebbe di soli 78 step; a 16 sono 623/epoca,
#                                    confrontabili con i 1054/epoca del multicorpus.
#   * epochs 300                  -> 95
#   * inv_gamma                   -> 200000 RIPRISTINATO = il valore di trainer.yaml
#                                    che gli slurm LARGE2 usano (non fanno override).
#                                    Serve l'override esplicito perche' l'experiment
#                                    MULTICORPUS_swiglu lo abbassa a 36000.
#                                    Nota: a 118.180 step (1244/epoca MISURATI su
#                                    MED_LARGE2 x 95 epoche) l'InverseLR resta a ~78%
#                                    del picco -> niente cooldown. Voluto: e' la
#                                    ricetta LARGE2, e tenerlo uguale fra i 4 arm
#                                    toglie la posizione nello schedule dai confondenti.
#   * --time 2-00:00:00           -> 24:00:00  (~12.7h stimate + margine)
#   * namespace / MASTER_PORT distinti
# detach_warmup_steps resta 25000: e' il valore gia' provato su FMA-medium a 2 GPU
# (MED_LARGE2), non lo riscalo per non inventare un regime non testato.
#
# fold_lrms=true su TUTTI e quattro gli arm: e' la condizione di F4 (il warm start
# parte da lrms_disc_e219, dove e' gia' attivo) e il suo valore marginale in
# presenza di una loss M/S e' gia' misurato (ms_add vs ms_add_lrms: stessa
# immagine, meta' del costo FAD). E' la costante dell'esperimento, non una variabile.
#
# Costo: ~350 local h per arm — MISURATO sui checkpoint di STEREO_N
# (01:48:13 -> last.ckpt 23:42:16 = 21.9 h wall; 13.8 min/epoca su 80 epoche;
# billing 16 = 32 x R x 1 nodo, R = max(2/4 GPU, 16/32 CPU, 240/494 GB) = 0.5).
# La stima originale di 10.7 h nei 4 arm precedenti era sbagliata di 2x.
# ============================================================
#SBATCH --job-name=STEREO_E
#SBATCH --account=IscrC_MM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # 1 task = 1 GPU
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Override Hydra opzionali passati a sbatch, applicati PER ULTIMI (vincono su
# tutto il resto). Usati dallo smoke: limit_train_batches / epochs / nome W&B.
EXTRA=("$@")

# --- Tracciamento preambolo (aggiunto 27/08) -------------------------------
# Il 27/08 STEREO_E/F sono restati ~9 min nella sola .batch senza mai lanciare
# srun e con .out a 0 byte: il primo echo era dopo il preambolo, che era quindi
# un punto cieco totale. Ora ogni riga del preambolo finisce su stderr con
# l'ora, cosi' un eventuale blocco (Lmod? metadata Lustre? venv?) si localizza
# subito invece di presentarsi come un job "RUNNING" che non scrive nulla.
echo "[$(date '+%T')] batch step avviato su $(hostname) - job $SLURM_JOB_ID"
export PS4='+ [$(date +%T)] riga ${LINENO}: '
set -x

module load profile/deeplrn cineca-ai/4.3.0

# Compute su IscrC_MM, ma repo/venv/dataset stanno in IscrC_AHNetBio ($WORK/$FAST
# puntano a IscrC_MM dopo il cambio budget -> vanno fissati PRIMA di ogni uso).
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
export MASTER_PORT=29605

# Proxy squid: gira su login05 (login07 e' giu' e fra login node non c'e' ssh).
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

# --- Auto-resume -----------------------------------------------------------
RUN_ID_FILE="$WORK/C-VAE/.run_ids/STEREO_E"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/STEREO_E"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ARG="+ckpt_path=$LATEST_CKPT"
    echo "[RESUME] Checkpoint: $LATEST_CKPT"
fi

# --- Retry loop ------------------------------------------------------------
MAX_ATTEMPTS=8
attempt=1
set +x
echo "[$(date '+%T')] preambolo completato in $SECONDS s, entro nel retry loop"
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    # --export=ALL e' obbligatorio, non cosmetico: se lo slurm e' sottomesso con
    # sbatch --export=NONE, SLURM imposta SLURM_EXPORT_ENV=NONE e srun NON propaga
    # l'ambiente al task -> il PYTHONPATH messo da `module load` non arriva a python,
    # il torch del modulo esce dal path e vince la dir fantasma
    # .envs/c_vae/lib/python3.11/site-packages/torch/ (solo header, residuo di un
    # install interrotto) -> ModuleNotFoundError: No module named 'torch.profiler'.
    srun --export=ALL --kill-on-bad-exit=1 python train.py \
        +experiment=MULTICORPUS_swiglu \
        data=fma \
        data.fma_subset=medium \
        data.train_dataloader.batch_size=16 \
        data.eval_dataloader.batch_size=16 \
        trainer.trainer.epochs=95 \
        trainer.scheduler.inv_gamma=200000 \
        trainer.trainer.save_every_n_epochs=10 \
        trainer.trainer.check_val_every_n_epoch=1 \
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
        trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0 \
        trainer.trainer.num_gpus=$SLURM_NTASKS_PER_NODE \
        "++trainer.trainer.num_nodes=1" \
        trainer.wandb.use_wandb=true \
        trainer.wandb.name=STEREO_E \
        trainer.trainer.ckpt_dir=checkpoints/STEREO_E \
        $CKPT_ARG ${EXTRA[@]+"${EXTRA[@]}"}

    EXIT=$?
    [ $EXIT -eq 0 ] && break

    echo "[$(date '+%T')] Attempt $attempt failed (exit $EXIT), retrying in 15s..."
    sleep 15
    attempt=$((attempt + 1))

    if [ -f "$RUN_ID_FILE" ]; then
        export WANDB_RESUME=must
        export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    fi
    LATEST_CKPT=$(ls -t "$WORK/C-VAE/runs/STEREO_E"/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done
