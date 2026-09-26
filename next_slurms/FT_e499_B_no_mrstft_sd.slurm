#!/bin/bash
# ============================================================
# FT_e499_B_no_mrstft_sd — BRACCIO B: A senza il termine MRSTFT M/S.
#
#   IDENTICO al braccio A (next_slurms/FT_e499_A_mrstft_sd.slurm) tranne UNA riga:
#   mrstft_sd NON viene attivata, resta al default 0.0 (config/trainer.yaml:163)
#   quindi il termine e' inerte. Nient'altro cambia.
#
#   Ipotesi: lo stereo e' gia' nel latente CONGELATO (freeze_encoder=true,
#   engine.py:85), quindi durante il FT non serve piu' RIALZARE il livello del
#   Side — serve solo non perderlo. Se e' vero, togliere mrstft_sd libera
#   FAD/CLAP dal costo misurato del termine di magnitudine (fad_mert 2.2x a e319)
#   senza far ricollassare l'immagine.
#
#   Cosa resta a tenere il Side, e sono entrambe gia' in A:
#     - stft_mse=1.0  → la ComplexMSE sullo spettrogramma CAC power-norm, che
#       contiene GIA' la fase del Side (identita' del parallelogramma, rapporto
#       misurato 1.00000012);
#     - fold_lrms=true → i critici del discriminatore vedono L,R,M,S, quindi la
#       pressione avversaria sul Side resta attiva. E' un parametro del
#       discriminatore, NON la loss MRSTFT M/S, percio' non si tocca.
#
#   Leggibile presto: width_bias e sisdr_s al primo checkpoint utile (e25, ~4h)
#   dicono gia' se B tiene l'immagine o la perde.
#
#   sbatch next_slurms/FT_e499_B_no_mrstft_sd.slurm
# ============================================================
#SBATCH --job-name=FT_e499_B_no_mrstft_sd
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

export MASTER_PORT=29622                # distinto da A/B e dagli altri job multicorpus

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
RUN_ID_FILE="$WORK/C-VAE/.run_ids/FT_e499_B_no_mrstft_sd"
if [ -f "$RUN_ID_FILE" ]; then
    export WANDB_RESUME=must
    export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
    echo "[RESUME] W&B run: $WANDB_RUN_ID"
fi

# ─── Resume: il piu' fresco fra {hpc_ckpt_* (segnale), last.ckpt (periodico)} ─
# Ogni candidato e' validato con `unzip -l`: un kill durante la riscrittura di
# last.ckpt lascia un file troncato a 64 MiB che fa fallire il resume.
RUN_BASE="$WORK/C-VAE/runs/FT_e499_B_no_mrstft_sd"
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
        trainer.trainer.ckpt_dir=checkpoints/FT_e499_B_no_mrstft_sd \
        trainer.wandb.name=FT_e499_B_no_mrstft_sd \
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
