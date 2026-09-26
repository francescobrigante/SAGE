#!/bin/bash
# ============================================================
# FT_e499_A_mrstft_sd — BRACCIO A: "continuazione naturale" del pretrain M/S.
#
#   Sorgente: MULTICORPUS_swiglu_ms_add_lrms @ep499 (schedule 500 completata,
#   job 55620293 COMPLETED 0:0 il 2026-09-05). E' il primo FT che parte da un
#   latente con lo STEREO INTATTO: width_bias +0.0024, d_width 0.0604.
#
#   RICETTA = next_slurms/FT_MULTICORPUS_swiglu_ep359.slurm VERBATIM (postnet
#   zero-init + discriminatore WavTokenizer da zero, encoder congelato, EMA
#   0.9998, kl=0, FP32, niente teacher/CLAP) con SOLO tre differenze:
#     1. SRC_CKPT → ms_add_lrms/epoch_499 invece di swiglu_500ep/epoch_359;
#     2. le DUE righe M/S del pretrain, riportate identiche (vedi sotto);
#     3. account EUHPC_D34_099 (IscrC_MM esaurito) + path WORK/FAST hardcodati.
#
#   Le due righe M/S sono copiate carattere per carattere da
#   next_slurms/MULTICORPUS_swiglu_ms_add_lrms_cont500.slurm:
#       trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0
#       ++trainer.loss_config.discriminator.config.fold_lrms=true
#   Entrambe le chiavi esistono nel loss_config di default (config/trainer.yaml:161
#   e :192, peso 0.0 → inerti se non attivate), quindi l'override funziona anche
#   senza +experiment. fold_lrms e' un parametro del discriminatore WavTokenizer
#   (src/ar_spectra/models/discriminators/wavtokenizer.py:41), che e' esattamente
#   quello usato dal FT → supportato.
#
#   COSTO DI fold_lrms, MISURATO sul pretrain (mtime dei checkpoint):
#     ms_add (senza)      7.15 min/ep
#     ms_add_lrms (con)  11.33 min/ep   → +58%
#   Nessun FT precedente lo usava: i 6.56 min/ep misurati su FT_..._ms_add
#   (job 50133705) sono SENZA fold_lrms. Attesa qui ~10.4 min/ep → ~970 epoche
#   in 7 giorni, non 1500. Non e' un problema: inv_gamma e' step-based e
#   indipendente da epochs (src/.../schedulers.py:40-44), quindi epochs= sposta
#   solo il punto di arresto e il checkpoint migliore si sceglie a valle.
#   Per tornare a 6.56 min/ep basta commentare la riga fold_lrms.
#
#   sbatch next_slurms/FT_e499_A_mrstft_sd.slurm
# ============================================================
#SBATCH --job-name=FT_e499_A_mrstft_sd
# 2026-09-10: ripartenza dopo il TIMEOUT dei 4 giorni. Account spostato su
# EUHPC_D34_099: IscrC_MM e' chiuso (101.2%) e la ripartenza del braccio B
# sul suo account originario sarebbe stata rifiutata. EUHPC ha 67.5k h
# residui ma questo mese e' 2.7x sopra la quota mensile, quindi il
# fairshare e' basso: l'ordine A-prima-di-B e' imposto da una dependency
# Slurm (--dependency=after:<jobid A>), non dalla priorita'.
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 1 task = 1 GPU → 16 tasks total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8              # 32 cores/node ÷ 4 tasks
#SBATCH --mem-per-cpu=15G              # 4×8×15 = 480G/node (< 512G)
#SBATCH --time=4-00:00:00              # cap lprod; nessuna MAINT sui nodi boost prima del 15/10
#SBATCH --requeue                      # richiesto dall'auto-requeue di PL
#SBATCH --signal=SIGUSR1@120           # SIGUSR1 120s prima del TIMEOUT → PL salva + requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load profile/deeplrn cineca-ai/4.3.0

# Repo, venv, dataset e checkpoint vivono su IscrC_AHNetBio anche se il billing e'
# EUHPC: senza questo hardcode $WORK segue l'account e il venv non viene trovato.
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1                 # niente teacher/CLAP nel decoder-FT

export MASTER_PORT=29621                # distinto da A/B e dagli altri job multicorpus

export http_proxy='http://login05:3150' # login07 e' morto: proxy su login05
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

# ─── Pesi sorgente: ms_add_lrms @ep499 (fine schedule, stereo intatto) ────────
SRC_CKPT="$WORK/C-VAE/runs/MULTICORPUS_swiglu_ms_add_lrms/2026-09-04_15-40-54/checkpoints/epoch_499.ckpt"
if [ ! -f "$SRC_CKPT" ]; then
    echo "[FATAL] source checkpoint non trovato: $SRC_CKPT" >&2
    exit 1
fi
echo "[INIT] decoder-finetune init da ms_add_lrms ep499 (EMA): $SRC_CKPT"

# ─── W&B (run nuova; l'id viene persistito da train.py in .run_ids/<name>) ────
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_e499_A_mrstft_sd"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: il piu' fresco fra {hpc_ckpt_* (segnale), last.ckpt (periodico)} ─
# Ogni candidato e' validato con `unzip -l`: un kill durante la riscrittura di
# last.ckpt lascia un file troncato a 64 MiB che fa fallire il resume.
RUN_BASE="$WORK/C-VAE/runs/FT_e499_A_mrstft_sd"
pick_ckpt() {
    local c
    for c in $(ls -t "$RUN_BASE"/*/hpc_ckpt_*.ckpt \
                     "$RUN_BASE"/*/checkpoints/last.ckpt 2>/dev/null); do
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
fi

# ─── Retry loop (crash transitori; il TIMEOUT lo gestisce il segnale→requeue) ─
MAX_ATTEMPTS=8
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date '+%T')] Training attempt $attempt/$MAX_ATTEMPTS"

    srun --export=ALL --kill-on-bad-exit=1 python train.py \
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
        trainer.loss_config.mrstft_sd.weights.mrstft_sd=1.0 \
        "++trainer.loss_config.discriminator.config.fold_lrms=true" \
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
        trainer.trainer.ckpt_dir=checkpoints/FT_e499_A_mrstft_sd \
        trainer.wandb.name=FT_e499_A_mrstft_sd \
        $CKPT_ARG

    EXIT=$?
    [ $EXIT -eq 0 ] && break

    echo "[$(date '+%T')] Attempt $attempt failed (exit $EXIT), retry fra 15s..."
    sleep 15
    attempt=$((attempt + 1))

    if [ -f "$RUN_ID_FILE" ]; then
        export WANDB_RESUME=must
        export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    fi
    LATEST_CKPT=$(pick_ckpt)
    [ -n "$LATEST_CKPT" ] && CKPT_ARG="+ckpt_path=$LATEST_CKPT"
done

# Se il loop esce per esaurimento dei tentativi (e non per `break` su exit 0) lo
# script DEVE fallire. Senza, l'ultimo comando eseguito e' l'assegnazione
# `attempt=$((attempt+1))` che vale 0, e Slurm registra COMPLETED 0:0 su un job
# totalmente fallito — successo apparente su 8 tentativi andati a vuoto.
if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "[$(date '+%T')] TUTTI i $MAX_ATTEMPTS tentativi falliti" >&2
    exit 1
fi
exit 0
