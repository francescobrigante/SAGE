#!/bin/bash
# ============================================================
# evaluation/watch_muon_ep499.slurm
# One-shot watcher: wait for the MULTICORPUS_swiglu_muon ep499 checkpoint to be
# fully written, then create a distinct-named symlink and submit the FMA +
# MoisesDB-10s recon eval (same clean pattern as the other epochs).
# CPU-only, self-resubmits if it hits the serial wall-time before the ckpt lands.
#   sbatch evaluation/watch_muon_ep499.slurm
# ============================================================
#SBATCH --job-name=watch_muon_ep499
#SBATCH --account=IscrC_AHNetBio
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "$WORK/C-VAE"
mkdir -p logs

SELF="evaluation/watch_muon_ep499.slurm"
CK_SRC="$WORK/C-VAE/runs/MULTICORPUS_swiglu_muon_500ep/2026-07-04_16-15-14/checkpoints/epoch_499.ckpt"
LINK="checkpoints/MULTICORPUS_swiglu_muon_ep499.ckpt"
FMA_ROOT="$WORK/C-VAE/runs/multicorpus_eval/recon_fma"
M10S_ROOT="$WORK/C-VAE/runs/multicorpus_eval/recon_moisesdb_10s"

# Stop polling ~20 min before the wall so a resubmit can take over cleanly.
DEADLINE=$(( $(date +%s) + 3*3600 + 25*60 ))

# Already evaluated? (idempotent — nothing to do)
if [ -f "$FMA_ROOT/MULTICORPUS_swiglu_muon_ep499/metrics/_done" ]; then
    echo "[$(date)] muon ep499 FMA already done — exiting."
    exit 0
fi

# True only when the checkpoint exists and its size is stable over 60s
# (i.e. fully flushed, not mid-write).
ckpt_ready() {
    [ -f "$CK_SRC" ] || return 1
    local s1 s2
    s1=$(stat -c%s "$CK_SRC" 2>/dev/null) || return 1
    sleep 60
    s2=$(stat -c%s "$CK_SRC" 2>/dev/null) || return 1
    [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]
}

echo "[$(date)] watching $CK_SRC (deadline $(date -d @$DEADLINE '+%H:%M'))"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ckpt_ready; then
        echo "[$(date)] checkpoint ready — submitting eval."
        ln -sfn "$CK_SRC" "$LINK"
        sbatch --job-name=mc_fma_swiglu_muon_ep499 --time=08:00:00 \
            --export=ALL,EVAL_ROOT="$FMA_ROOT" \
            evaluation/final_eval/recon_fma_swin.slurm --checkpoint "$LINK"
        sbatch --job-name=mc_m10s_swiglu_muon_ep499 --time=06:00:00 \
            --export=ALL,EVAL_ROOT="$M10S_ROOT" \
            evaluation/final_eval/recon_moises10s_swin.slurm --checkpoint "$LINK"
        echo "[$(date)] eval submitted → recon_fma/recon_moisesdb_10s/MULTICORPUS_swiglu_muon_ep499"
        exit 0
    fi
    sleep 180
done

echo "[$(date)] deadline reached, checkpoint not ready — resubmitting watcher."
sbatch "$SELF"
