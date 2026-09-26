#!/bin/bash
# ============================================================
# final_eval/recon_musiccaps_sdd_swin_tag.slurm — NEW. SAGE reconstruction on
# one of the two frozen SOTA-comparison sets (MusicCaps or Song Describer),
# read by the "MusicCaps & SDD" notebook cell (evaluation/results_viz.ipynb).
# Same evaluate_swin_10s.py path as MoisesDB-10s, just pointed at a different
# frozen dataset dir; both already ship stats_ours/ + embeddings/ ref caches
# built by build_refs_10s.py, reused unchanged (proven by the existing
# SAGE_e299/SAGE_final_1499/epoch_* rows already under these tags).
# Writes into runs/eval_swin_10s/<DATASET_TAG>/<CKPT_TAG>/metrics/, exactly
# where _registry() in the notebook cell looks for "Ours" rows.
#
# Tempi estrapolati dal tempo di pura inferenza gia' misurato (SAGE_final_1499,
# stesso script): MusicCaps 964 file ~10 min, SDD 8364 file ~80-90 min — SDD
# supera il muro di 30 min di boost_qos_dbg, quindi qui la qos e' `normal`
# per entrambi (uniforme, evita di doverla scegliere per dataset).
#
# Il fix variable-length (tri2) e' applicato di default da EuleroEncodeDecode:
# nessun flag necessario, vedi VARLEN_SEAMS.md §8.4.
#
#   sbatch evaluation/final_eval/recon_musiccaps_sdd_swin_tag.slurm SAGE_e992_tri2 musiccaps
#   sbatch evaluation/final_eval/recon_musiccaps_sdd_swin_tag.slurm SAGE_e992_tri2 sdd
# ============================================================
#SBATCH --job-name=fe_recon_mcsdd_tag
#SBATCH --account=EUHPC_D34_099
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

export WORK=/leonardo_work/IscrC_AHNetBio
export FAST=/leonardo_scratch/fast/IscrC_AHNetBio

cd "$WORK/C-VAE" || exit 1
mkdir -p logs

TAG="${1:?Usage: sbatch recon_musiccaps_sdd_swin_tag.slurm <CKPT_TAG> <musiccaps|sdd>}"
DATASET="${2:?Usage: sbatch recon_musiccaps_sdd_swin_tag.slurm <CKPT_TAG> <musiccaps|sdd>}"
shift 2

case "$DATASET" in
  musiccaps) DATA_DIR=$FAST/datasets/musiccaps/balanced_44k_stereo/audio
             DATASET_TAG=balanced_44k_stereo ;;
  sdd)       DATA_DIR=$FAST/datasets/song_describer/sdd_44k_stereo_10s/audio
             DATASET_TAG=sdd_44k_stereo_10s ;;
  *) echo "DATASET deve essere 'musiccaps' o 'sdd'" >&2; exit 1 ;;
esac

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

: "${EVAL_ROOT:=$WORK/C-VAE/runs/eval_swin_10s/$DATASET_TAG}"
mkdir -p "$EVAL_ROOT"

echo "[$(date)] recon $DATASET_TAG — $TAG — 1 GPU — data=$DATA_DIR — out=$EVAL_ROOT"

python -u evaluation/evaluate_swin_10s.py \
    --checkpoint  checkpoints/$TAG.ckpt \
    --data-dir    "$DATA_DIR" \
    --output-dir  "$EVAL_ROOT" \
    --device      cuda \
    --num-workers 8 \
    --resume \
    "$@"

echo "[$(date)] DONE → $EVAL_ROOT/$TAG/metrics/"
