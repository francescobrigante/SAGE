#!/bin/bash
# ============================================================
# final_eval/maeb_swin_tag.slurm — come final_eval/maeb_swin.slurm ma con
# CHECKPOINT e CARTELLA DI OUTPUT passati da CLI (l'originale li ha cablati su
# SAGE_e299). Tutte e 19 le task MAEB (6 FMA + 7 MoisesDB-30s + 6 upstream music),
# 1 GPU, protocollo 64-d (--standardize-bottleneck, come le tabelle del paper).
# I pesi caricati sono quelli EMA: EuleroEncodeDecode li preferisce sempre
# (src/ar_spectra/models/inference.py:54-72), come per FAD/CDPAM/CLAP e per i SOTA.
#
# Due passate resume-safe: 1) FMA+MoisesDB, 2) upstream music (merge non distruttivo).
# Output in runs/final_eval/maeb/maeb_<OUT_NAME>/ → il nome <OUT_NAME> e' quello da
# aggiungere a MODELS nella cella "FINAL EVAL — paper tables" del notebook, che
# risolve _maeb_root(m) = runs/final_eval/maeb/maeb_<m>.
#
#   sbatch evaluation/final_eval/maeb_swin_tag.slurm <CKPT_TAG> <OUT_NAME>
#   es.  ... SAGE_msfix_FT_e949 SAGE_msfix_FT_e949
#
# Account EUHPC_D34_099 (AHNetBio e MM chiusi). QOS normal: boost_qos_dbg ha
# priorita' piu' alta ma walltime max 30 min, e le run MAEB misurate durano
# 6-36 min — troppo al limite. Durata attesa qui: ~30 min.
# ============================================================
#SBATCH --job-name=fe_maeb_tag
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Compute on IscrC_MM (IscrC_AHNetBio budget exhausted), but WORK/FAST must point
# to IscrC_AHNetBio where repo, venv, checkpoints and HF cache live (user-owned,
# world-readable → readable from an IscrC_MM job).
export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

mkdir -p logs
cd $WORK/C-VAE

# Isolated maeb_dl venv: purge modules + unset PYTHONPATH (CINECA.md §5.4).
module purge || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source $WORK/.envs/maeb_dl/bin/activate

export HF_HOME="$FAST/hf_cache"
export HF_DATASETS_CACHE="$FAST/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG="${1:?Usage: sbatch maeb_swin_tag.slurm <CKPT_TAG> <OUT_NAME>}"
OUT_NAME="${2:?Usage: sbatch maeb_swin_tag.slurm <CKPT_TAG> <OUT_NAME>}"
shift 2
OUT="$WORK/C-VAE/runs/final_eval/maeb/maeb_$OUT_NAME"
echo "[$(date)] FINAL MAEB (19 task: 6 FMA + 7 MoisesDB + 6 upstream) — $TAG — out=$OUT"

# Pass 1: FMA (6) + MoisesDB (7).
python evaluation/maeb_swin.py \
    --ckpt checkpoints/$TAG.ckpt \
    --with-moisesdb --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

# Pass 2: upstream MAEB music tasks (6) on original datasets — merges into the
# same summary.json ($OUT), non-destructive (runner.py merge).
python evaluation/maeb_swin.py \
    --ckpt checkpoints/$TAG.ckpt \
    --maeb-original-music-only --standardize-bottleneck \
    --output-dir "$OUT" \
    --device cuda \
    "$@"

echo "[$(date)] DONE → $OUT"
