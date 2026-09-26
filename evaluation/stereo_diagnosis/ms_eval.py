# ===============
# Shared stereo-imaging row builder for the 10s reconstruction evaluators.
#
# evaluate_swin_10s.py (SAGE) and sota_models/evaluate_sota_10s.py (SAO, SAME,
# CoDiCodec, Music2Latent) both build their ms_metrics row through this module,
# so our model and the baselines we compare against are measured by *identical*
# code. That is the precondition for reading the two tables side by side — the
# published SOTA stereo numbers are not on our eval set, so the only usable
# target row is one we compute ourselves.
#
# stereo_imaging.py — the metric core behind every number in
# STEREO_COLLAPSE_DIAGNOSIS.md — is deliberately NOT modified here: this file
# only composes it, so previously reported values keep their provenance.
# ===============
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from stereo_imaging import align, stereo_imaging_distance   # torch-only, no heavy deps

# Per-file schema. The first five columns are byte-identical to the legacy
# --compute-ms-metrics schema (old ms_metrics.csv stay readable, and readers that
# select columns by name are unaffected). The rest are already computed by
# stereo_imaging_distance and were previously thrown away.
MS_COLUMNS = [
    "file",
    "width_bias", "d_width",      # Side LEVEL (signed) / stereo image error (unsigned)
    "sisdr_s", "sisdr_m",         # Side CONTENT / Mid guardrail
    "d_pan", "pan_bias",          # L/R balance
    "score",                      # (d_width + d_pan) / 2
    "sm_ref_db", "sm_rec_db",     # global 10*log10(E_S/E_M), reference and reconstruction
    "ref_mono",                   # 1 = degenerate reference Side, see below
]

# A reference whose Side sits this far below its Mid carries no stereo
# information at all: in practice a mono file duplicated onto two channels
# (utils/audio.py:110-113 `wav.repeat(2, 1)`). Such files exist in every corpus
# we evaluate on — 14 of the 964 MusicCaps clips are mono→stereo duplicates by
# construction (EVALUATION.md §5.1), and M4Singer is 100% mono in training.
# For them S is exactly 0, so width_bias and sisdr_s are degenerate and must not
# enter a Side average. Flagged per file rather than dropped, so the caller
# decides and the count stays visible.
MONO_REF_DB = -80.0


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-10) -> float:
    """Scale-invariant SDR (dB), matching stereo_diagnosis/measure_imaging.si_sdr."""
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est * ref).sum() / (ref.square().sum() + eps)
    proj = a * ref
    return float((10 * torch.log10(proj.square().sum() / ((est - proj).square().sum() + eps))).item())


def ms_metrics_row(ref: torch.Tensor, pred: torch.Tensor, sr: int) -> Dict[str, float]:
    """Stereo-imaging metrics on a (2, T) ref/pred pair, delay-aligned first.

    Two quantities carry the stereo verdict and they are near-orthogonal
    (Spearman rho = +0.02 across the M/S ablation set), so neither substitutes
    for the other:

      width_bias  Side *level* error (→0; <0 squash, >0 over-widening)
      sisdr_s     Side *content*, SI-SDR on S=(L-R)/sqrt(2) — scale-invariant,
                  so it is blind to level and sees only whether the Side is the
                  right signal.

    The ms_replace arm is the cautionary case: +0.5 dB of Side level (the best
    of any arm) with sisdr_s = -14.66, i.e. baseline-grade content. Reading
    width_bias alone would have declared it the winner.

    sisdr_m is the Mid guardrail: whatever fixes the Side must leave the Mid —
    the only thing FAD/CLAP/CDPAM actually measure — intact.
    """
    ref_a, pred_a = align(ref.float(), pred.float())      # delay-compensate (~no-op for our recon)
    d = stereo_imaging_distance(ref_a, pred_a, sample_rate=sr)

    Sr = (ref_a[0] - ref_a[1]) / 2 ** 0.5                 # (T,) reference side
    Sx = (pred_a[0] - pred_a[1]) / 2 ** 0.5               # (T,) predicted side
    Mr = (ref_a[0] + ref_a[1]) / 2 ** 0.5                 # (T,) reference mid
    Mx = (pred_a[0] + pred_a[1]) / 2 ** 0.5               # (T,) predicted mid

    return {
        "width_bias": d["width_bias"], "d_width": d["d_width"],
        "sisdr_s": si_sdr(Sx, Sr), "sisdr_m": si_sdr(Mx, Mr),
        "d_pan": d["d_pan"], "pan_bias": d["pan_bias"], "score": d["score"],
        "sm_ref_db": d["sm_ref_db"], "sm_rec_db": d["sm_rec_db"],
        "ref_mono": int(d["sm_ref_db"] < MONO_REF_DB),
    }
