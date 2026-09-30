# ===============
# Smoke tests of the evaluation pipeline on the tiny SAGE checkpoint and synthetic audio:
# the reconstruction evaluator on a clip set end to end (SI-SDR/SDR/STFT/mel), the Fréchet distance,
# and the MAEB encoder protocol (64-d time-pooled latent, 19 tasks). No pretrained embedders:
# FAD/CLAP/CDPAM with real weights live behind the `weights` marker.
# ===============
from __future__ import annotations

import csv
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import REPO

pytestmark = pytest.mark.smoke


def test_reconstruction_evaluator_end_to_end(tiny_pretrain_ckpt, audio_dir, tmp_path):
    """SAGE samples z, and the per-file seed makes two runs give the same numbers."""
    spectral = []
    for run in ("a", "b"):
        out = tmp_path / run
        cmd = [sys.executable, "-m", "evaluation.reconstruction", "model=sage", "dataset=musiccaps",
               f"checkpoint={tiny_pretrain_ckpt}", f"dataset.data_dir={audio_dir}",
               f"output_dir={out}", "device=cpu", "num_workers=0", "sdr_only=true"]
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        csvs = list(out.rglob("metrics/*.csv"))
        assert csvs, f"no metric CSV written under {out}"
        rows = [r for f in csvs for r in csv.DictReader(open(f))]
        assert rows and all(np.isfinite(float(v)) for r in rows for k, v in r.items()
                            if k not in ("file", "model") and v not in ("", None))
        spectral.append((out / tiny_pretrain_ckpt.stem / "metrics" / "spectral.csv").read_text())
    assert spectral[0] == spectral[1]


def test_metric_functions_on_synthetic_audio():
    from evaluation.metrics import signal as losses
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(2, 44100, generator=g)
    assert losses.si_sdr(ref, ref) > 60                                  # identical signals
    assert losses.si_sdr(ref, ref + 0.1 * torch.randn(2, 44100, generator=g)) > losses.si_sdr(
        ref, ref + torch.randn(2, 44100, generator=g))                   # less noise, higher SI-SDR


def test_frechet_distance():
    from fadtk.fad import calc_frechet_distance
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2000, 16)), rng.normal(size=(2000, 16))
    stats = lambda x: (x.mean(0), np.cov(x, rowvar=False))
    same = calc_frechet_distance(*stats(a), *stats(b))
    shifted = calc_frechet_distance(*stats(a), *stats(b + 1.0))
    assert same < 0.1 < shifted


def test_maeb_encoder_gives_64d_embeddings(tiny_pretrain_ckpt):
    from evaluation.maeb.sage_encoder import SAGELatentEncoder
    enc = SAGELatentEncoder(str(tiny_pretrain_ckpt), device="cpu", standardize_bottleneck=True)
    emb = enc._encode_item(0.1 * torch.randn(2, 44100 * 3))
    assert emb.shape == (64,) and torch.isfinite(emb).all()          # 16 channels x 4 freq bands


def test_maeb_suite_has_the_19_tasks_of_the_paper():
    from evaluation.maeb.tasks import ALLOWED_TASKS, FMA_SUITE, MAEB_ORIGINAL_MUSIC, MOISESDB_SUITE
    assert (len(FMA_SUITE), len(MOISESDB_SUITE), len(MAEB_ORIGINAL_MUSIC)) == (6, 7, 6)
    assert len(set(ALLOWED_TASKS)) == 19
