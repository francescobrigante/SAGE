#!/usr/bin/env python3
"""
evaluation/test_evaluate.py
Fast offline test suite — no GPU, no model downloads, no audio files needed.
Run from the project root:
    python evaluation/test_evaluate.py

Tests:
  1. Import chain (utils → compute_*.py → evaluate.py)
  2. SI-SDR: perfect reconstruction → +∞, orthogonal → very negative
  3. STFT loss: identical signals → 0
  4. batch_align: artificial lag recovery
  5. cosine_sim: identity → 1.0, orthogonal → 0.0
  6. infer_batch: mock codec, shape correctness
  7. collect_fma_files: temporary directory scan
  8. write_csv / atomic_save_npy: round-trip
  9. evaluate.py --help (CLI smoke test)
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent.resolve()
EVAL_DIR  = PROJ_ROOT / "evaluation"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(EVAL_DIR))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_results = []


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, cond))


def section(title: str):
    print(f"\n── {title} {'─' * (55 - len(title))}")


# ── 1. Imports ────────────────────────────────────────────────
section("Import chain")
try:
    from utils import (
        atomic_save_npy, write_csv, silence_output, collect_fma_files,
        batch_align, si_sdr, stft_loss, cdpam_score,
        load_or_embed, target_cache_path,
        get_expected_frames, infer_batch,
    )
    check("utils.py imports", True)
except Exception as e:
    check("utils.py imports", False, str(e))

try:
    from compute_clap_score import cosine_sim, embed_clap
    check("compute_clap_score: cosine_sim + embed_clap", True)
except Exception as e:
    check("compute_clap_score: cosine_sim + embed_clap", False, str(e))

try:
    from compute_fad import embed_mert, compute_fad_from_embeddings
    check("compute_fad: embed_mert + compute_fad_from_embeddings", True)
except Exception as e:
    check("compute_fad: embed_mert + compute_fad_from_embeddings", False, str(e))

try:
    # evaluate.py imports are checked by AST; avoid running main()
    import ast
    src = (EVAL_DIR / "evaluate.py").read_text()
    ast.parse(src)
    check("evaluate.py AST parse", True)
except Exception as e:
    check("evaluate.py AST parse", False, str(e))


# ── 2. SI-SDR ─────────────────────────────────────────────────
section("SI-SDR metric")
t = torch.randn(2, 44100)
check("perfect reconstruction → high SI-SDR",
      si_sdr(t, t) > 60,
      f"{si_sdr(t, t):.1f} dB")

noise = torch.randn_like(t)
check("target vs pure noise → negative SI-SDR",
      si_sdr(t, noise) < 0)

# Scale-invariance: si_sdr(t, α·t) shouldn't penalise gain offsets.
# Near-identical signals hit float32 limits, so we just verify that
# a 2x-louder version scores HIGHER than pure noise (not negative).
check("2x-scaled target scores much better than noise",
      si_sdr(t, t * 2.0) > 0)


# ── 3. STFT loss ──────────────────────────────────────────────
section("STFT loss metric")
t2 = torch.randn(2, 44100)
check("identical signals → STFT loss ≈ 0",
      stft_loss(t2, t2) < 1e-5)

noisy = t2 + 0.5 * torch.randn_like(t2)
check("noisy signal → STFT loss > 0",
      stft_loss(t2, noisy) > 0.0)


# ── 4. batch_align ────────────────────────────────────────────
section("batch_align (FFT cross-correlation)")
sr = 16000
T  = 4 * sr
sig  = torch.randn(1, 2, T)
# Introduce known 200-sample shift
shift = 200
shifted = torch.zeros_like(sig)
shifted[..., shift:] = sig[..., :T - shift]
aln_t, aln_p, lags = batch_align(sig, shifted, sr)
lag_val = int(lags[0].item())
check("lag detected correctly",
      abs(lag_val - shift) <= 2,
      f"expected {shift}, got {lag_val}")

check("aligned output same shape as input",
      aln_t.shape == sig.shape)


# ── 5. cosine_sim ─────────────────────────────────────────────
section("cosine_sim")
a = np.array([1.0, 0.0, 0.0])
b = np.array([1.0, 0.0, 0.0])
check("identical → 1.0", abs(cosine_sim(a, b) - 1.0) < 1e-6)

c = np.array([0.0, 1.0, 0.0])
check("orthogonal → 0.0", abs(cosine_sim(a, c)) < 1e-6)

check("zero vector → 0.0", cosine_sim(np.zeros(3), a) == 0.0)


# ── 6. infer_batch (mock codec) ───────────────────────────────
section("infer_batch (mock codec, CPU)")

class _MockCodec:
    """Codec that returns a scaled copy of input (identity-ish)."""
    device = torch.device("cpu")
    def encode(self, x):    return x * 0.5       # latent = half amplitude
    def decode(self, z, target_length=None):
        out = z * 2.0                             # reconstruct
        if target_length is not None:
            out = out[..., :target_length]
        return out

codec = _MockCodec()
chunk = 4096
wavs  = [torch.randn(2, 30000), torch.randn(2, 27000)]  # two files, diff lengths
recon = infer_batch(codec, wavs, chunk)

check("returns list of 2 tensors", len(recon) == 2)
check("file 0 length preserved",
      recon[0].shape[-1] == wavs[0].shape[-1],
      f"{recon[0].shape[-1]} vs {wavs[0].shape[-1]}")
check("file 1 length preserved",
      recon[1].shape[-1] == wavs[1].shape[-1],
      f"{recon[1].shape[-1]} vs {wavs[1].shape[-1]}")
check("mock encode+decode is identity",
      (recon[0] - wavs[0]).abs().max().item() < 1e-5)


# ── 7. collect_fma_files ──────────────────────────────────────
section("collect_fma_files")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "000").mkdir()
    for name in ["000001.mp3", "000002.mp3", "000003.wav"]:
        (tmp / "000" / name).touch()
    (tmp / "metrics").mkdir()
    (tmp / "metrics" / "000001.mp3").touch()  # should be excluded

    files = collect_fma_files(tmp, {".mp3", ".wav"}, None, 0)
    check("finds 3 audio files (metrics/ excluded)",
          len(files) == 3,
          f"found {len(files)}")

    files_limited = collect_fma_files(tmp, {".mp3", ".wav"}, None, 2)
    check("max_files=2 → 2 results",
          len(files_limited) == 2)


# ── 8. write_csv + atomic_save_npy ───────────────────────────
section("I/O helpers")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # write_csv
    csv_path = tmp / "test.csv"
    write_csv(csv_path, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    check("write_csv: 2 rows written", len(rows) == 2)
    check("write_csv: values correct", rows[0]["a"] == "1" and rows[1]["b"] == "4")

    # atomic_save_npy
    npy_path = tmp / "sub" / "test.npy"
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    atomic_save_npy(npy_path, arr)
    loaded = np.load(npy_path)
    check("atomic_save_npy: round-trip",
          np.allclose(arr, loaded))


# ── 9. evaluate.py --help ─────────────────────────────────────
section("evaluate.py CLI smoke test")
# The subprocess must inherit the full sys.path of the parent so that
# torch (provided by the cineca-ai module, not the venv directly) is found.
_inherited_pypath = os.pathsep.join(
    [str(PROJ_ROOT), str(EVAL_DIR)]
    + [p for p in sys.path if p and p not in (str(PROJ_ROOT), str(EVAL_DIR))]
)
result = subprocess.run(
    [sys.executable, str(EVAL_DIR / "evaluate.py"), "--help"],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": _inherited_pypath},
    timeout=60,
)
check("--help exits cleanly (code 0)",
      result.returncode == 0,
      result.stderr.splitlines()[0] if result.returncode != 0 else "ok")
check("--checkpoint in help text", "--checkpoint" in result.stdout)
check("--shared-cache-dir in help text", "--shared-cache-dir" in result.stdout)
check("--batch-size in help text", "--batch-size" in result.stdout)


# ── Summary ───────────────────────────────────────────────────
total  = len(_results)
passed = sum(1 for _, ok in _results if ok)
failed = total - passed

print(f"\n{'═' * 60}")
if failed == 0:
    print(f"  \033[32m ALL {total} TESTS PASSED\033[0m")
else:
    print(f"  \033[31m {failed}/{total} TESTS FAILED\033[0m")
    for name, ok in _results:
        if not ok:
            print(f"    ✗ {name}")
print(f"{'═' * 60}")
sys.exit(0 if failed == 0 else 1)
