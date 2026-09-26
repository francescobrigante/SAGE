#!/bin/bash
# ============================================================
# final_eval/recon_moises10s_swin_tag.slurm — como recon_moises10s_swin.slurm
# ma con CHECKPOINT passato da CLI invece di cablato su SAGE_e299, e account
# EUHPC_D34_099 invece di IscrC_MM (98.4% consumato, solo 1292 local-h residui
# a inizio settembre — da non toccare per un job non urgente).
#
# SAGE reconstruction sul set MoisesDB-10s FULL (max-files 0), tutte le metriche.
# SINGLE GPU (evaluate_swin_10s.py non shard-a). Serve sia mixtures che stems:
# DATA_DIR override (gia' presente nell'originale) seleziona quale.
#
# Tempi reali misurati (stesso script, SAGE_e299): mixtures 1998 file ~20 min,
# stems 1546 file ~16 min (scalato). Entrambi entrano nel muro di 30 min di
# boost_qos_dbg — usato qui apposta per la priorita' piu' alta.
#
# Il fix variable-length (tri2) e' applicato di default da EuleroEncodeDecode:
# nessun flag necessario, vedi VARLEN_SEAMS.md §8.4.
#
#   # mixtures (default):
#   sbatch evaluation/final_eval/recon_moises10s_swin_tag.slurm SAGE_e992_tri2
#   # stems:
#   sbatch --export=ALL,DATA_DIR=$FAST/datasets/moisesdb/chunks_stem_original/original,EVAL_ROOT=$WORK/C-VAE/runs/final_eval/recon_moisesdb_stems_10s \
#          evaluation/final_eval/recon_moises10s_swin_tag.slurm SAGE_e992_tri2
# ============================================================
#SBATCH --job-name=fe_recon_m10s_tag
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

cd "$WORK/C-VAE" || exit 1
mkdir -p logs

TAG="${1:?Usage: sbatch recon_moises10s_swin_tag.slurm <CKPT_TAG> [extra args]}"
shift

module load profile/deeplrn cineca-ai/4.3.0
export HF_HOME="$FAST/hf_cache"
export HF_HUB_OFFLINE=1
export http_proxy='http://login07:3150'
export https_proxy='http://login07:3150'
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
source $WORK/C-VAE/.env
set +a

source $WORK/.envs/c_vae/bin/activate

# DATA_DIR overridable via env (default = mixtures) → lo stesso wrapper serve gli stems.
: "${DATA_DIR:=$FAST/datasets/moisesdb/chunks_mix_original/original}"
: "${EVAL_ROOT:=$WORK/C-VAE/runs/final_eval/recon_moisesdb_10s}"
mkdir -p "$EVAL_ROOT"

echo "[$(date)] FINAL recon MoisesDB-10s — $TAG — 1 GPU — data=$DATA_DIR — out=$EVAL_ROOT"

python -u evaluation/evaluate_swin_10s.py \
    --checkpoint  checkpoints/$TAG.ckpt \
    --data-dir    "$DATA_DIR" \
    --output-dir  "$EVAL_ROOT" \
    --device      cuda \
    --num-workers 8 \
    --resume \
    "$@"

echo "[$(date)] DONE → $EVAL_ROOT/$TAG/metrics/"
