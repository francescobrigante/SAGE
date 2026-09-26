#!/usr/bin/env python3
# =============================================================================
# evaluation/sota_models/build_refs_10s.py
# Precompute the CLEAN-ORIGINAL reference artefacts for a flat dir of ~10s WAVs,
# mirroring the chunks_mix_original layout so the existing *_10s evaluators
# (evaluate_{sota,swin,sao}_10s.py) can score ANY model against the dataset with
# NO code changes. Run ONCE per dataset (e.g. MusicCaps, SDD).
#
# Writes into <data-dir>:
#   embeddings/clap-laion-audio/{stem}.npy      (N_chunks, 512)  per-file ref → CLAP-audio cosine
#   embeddings/clap-laion-music/{stem}.npy      (N_chunks, 512)  per-file ref → CLAP-music cosine
#   embeddings/MERT-v1-95M-4/{stem}.npy         (N_frames, 768)  per-file      → fad_mert source
#   embeddings/clap-laion-audio-gud/{stem}.npy  (1, 512)         per-file      → fad_gudgud source
#   embeddings/pann-cnn14-16k/{stem}.npy        (1, 2048)        per-file      → fad_pann source
#   stats_ours/MERT-v1-95M-4/{mu,cov}.npy       Fréchet ref stats → fad_mert
#   stats_ours/clap-laion-audio-gud/{mu,cov}.npy  Fréchet ref stats → fad_gudgud
#   stats_ours/pann-cnn14-16k/{mu,cov}.npy      Fréchet ref stats → fad_pann
#
# Embeddings are computed with the SAME functions the evaluators apply to the
# reconstruction (embed_clap / embed_mert_framewise / embed_clap_gud / embed_pann), so
# the reference and the prediction always live in the same space.
# =============================================================================
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_SOTA_DIR  = Path(__file__).parent.resolve()
_EVAL_DIR  = _SOTA_DIR.parent
_PROJ_ROOT = _EVAL_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "src"), str(_EVAL_DIR), str(_SOTA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sage.utils.console import ok, warn, info, err
from utils import (atomic_save_npy, silence_output, ch_name,
                   CHANNEL_MID, CHANNEL_SIDE, CHANNELS)
from compute_clap_score import embed_clap, embed_clap_gud
from compute_fad import (embed_mert_framewise, embed_pann, get_pann_model,
                         compute_incremental_stats, PANN_NAME)
from fadtk.model_loader import CLAPLaionModel, MERTModel

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

_MERT_NAME       = "MERT-v1-95M-4"
_CLAP_AUDIO_NAME = "clap-laion-audio"
_CLAP_MUSIC_NAME = "clap-laion-music"
_CLAP_GUD_NAME   = "clap-laion-audio-gud"
_PANN_NAME       = PANN_NAME               # pann-cnn14-16k (from compute_fad)

# Models that need Fréchet reference stats (mu/cov) → fad_mert + fad_gudgud + fad_pann.
_STATS_MODELS = (_MERT_NAME, _CLAP_GUD_NAME, _PANN_NAME)

_SKIP_DIRS = frozenset({"embeddings", "stats", "stats_ours", "convert", "metrics", "parts"})



class _AudioDataset(Dataset):
    """Loads each clean WAV at its native rate as a [C, T] tensor."""

    def __init__(self, files: list[Path], target_sr: int = 44100, target_channels: int = 2):
        self.files = files
        self.target_sr = target_sr
        self.target_channels = target_channels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            C = wav.shape[0]
            if C > self.target_channels:
                wav = wav[:self.target_channels]
            elif C < self.target_channels:
                wav = wav.repeat((self.target_channels + C - 1) // C, 1)[:self.target_channels]
            return wav, p.stem
        except Exception as e:  # noqa: BLE001
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


def _collect_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(
        p for p in data_dir.glob("*.wav")
        if not _SKIP_DIRS.intersection(p.relative_to(data_dir).parts[:-1])
    )
    return files[:max_files] if max_files > 0 else files


_ALL_NAMES = (_CLAP_AUDIO_NAME, _CLAP_MUSIC_NAME, _MERT_NAME, _CLAP_GUD_NAME, _PANN_NAME)


def _all_refs_exist(emb_root: Path, stem: str, want: tuple[str, ...], channel: str) -> bool:
    return all((emb_root / ch_name(name, channel) / f"{stem}.npy").exists() for name in want)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build clean-original reference embeddings + FAD stats for a 10s WAV dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Flat dir of clean *.wav (refs are written into its embeddings/ + stats_ours/).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channel", default=CHANNEL_MID, choices=list(CHANNELS),
                   help="Which channel to embed: 'mid' = (L+R)/2, the historical mono downmix; "
                        "'side' = (L-R)/2. Side artefacts are written under '<model>-side/' so "
                        "they can never be crossed with the Mid reference stats. Every "
                        "preprocessing step inside the embedders is left untouched.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Skip files whose selected reference embeddings already exist.")
    p.add_argument("--stats-only", action="store_true",
                   help="Skip embedding; (re)compute stats_ours/ from existing embeddings/.")
    p.add_argument("--models", nargs="+", default=None, choices=list(_ALL_NAMES),
                   help="Restrict to a SUBSET of embedders (default: all). Use to add ONE new "
                        "embedder (e.g. --models pann-cnn14-16k) without recomputing/overwriting "
                        "the others' embeddings or stats.")
    return p.parse_args()


def _recompute_stats(emb_root: Path, stats_root: Path, want: tuple[str, ...],
                     channel: str = CHANNEL_MID) -> None:
    for model_name in _STATS_MODELS:
        if model_name not in want:
            continue
        model_name = ch_name(model_name, channel)
        npys = sorted((emb_root / model_name).glob("*.npy"))
        if not npys:
            warn(f"No embeddings for {model_name} — stats skipped.", prefix="STATS")
            continue
        mu, cov = compute_incremental_stats(npys)
        if mu is None:
            warn(f"<2 vectors for {model_name} — stats skipped.", prefix="STATS")
            continue
        out_dir = stats_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(out_dir / "mu.npy", mu.astype(np.float64))
        atomic_save_npy(out_dir / "cov.npy", cov.astype(np.float64))
        ok(f"stats_ours/{model_name}: mu{mu.shape} cov{cov.shape} ({len(npys)} files)", prefix="STATS")


def main() -> None:
    args = _parse_args()
    data_dir   = args.data_dir.expanduser().resolve()
    emb_root   = data_dir / "embeddings"
    stats_root = data_dir / "stats_ours"
    device = (torch.device("cpu") if args.device == "cpu" or not torch.cuda.is_available()
              else torch.device("cuda:0"))

    channel = args.channel
    if args.models:
        want = tuple(args.models)
    elif channel == CHANNEL_SIDE:
        # Sul Side servono solo i tre embedder che alimentano una FAD: le due
        # CLAP per-file esistono per la cosine similarity sul Mid, che non e'
        # parte di questo studio.
        want = _STATS_MODELS
    else:
        want = _ALL_NAMES
    info(f"channel = {channel}  ->  artefatti in embeddings/stats_ours/"
         f"{ch_name('<model>', channel)}", prefix="REFS")

    if args.stats_only:
        info(f"[STATS-ONLY] recomputing stats_ours/ from existing embeddings/ for {want}.", prefix="REFS")
        _recompute_stats(emb_root, stats_root, want, channel)
        return

    files = _collect_files(data_dir, args.max_files)
    if not files:
        err(f"No WAV files found in {data_dir}"); return

    if args.resume:
        todo = [f for f in files if not _all_refs_exist(emb_root, f.stem, want, channel)]
        info(f"{len(files)} files — {len(todo)} to embed (skip {len(files) - len(todo)} done).",
             prefix="REFS")
    else:
        todo = files
        info(f"{len(files)} files to embed.", prefix="REFS")

    # Load ONLY the models needed by the selected subset (CLAP-gud + PANN singletons load lazily).
    sr = 44100
    embedders: dict = {}          # name → callable(wav) → (N, D) np.ndarray
    info(f"Loading embedding models for: {want}", prefix="REFS")
    if _CLAP_AUDIO_NAME in want:
        clap_audio = CLAPLaionModel("audio")
        with silence_output(): clap_audio.load_model()
        clap_audio.model.to(device)
        embedders[_CLAP_AUDIO_NAME] = lambda wav: embed_clap(clap_audio, wav, sr, device, channel)
    if _CLAP_MUSIC_NAME in want:
        clap_music = CLAPLaionModel("music")
        with silence_output(): clap_music.load_model()
        clap_music.model.to(device)
        embedders[_CLAP_MUSIC_NAME] = lambda wav: embed_clap(clap_music, wav, sr, device, channel)
    if _MERT_NAME in want:
        mert = MERTModel(layer=4)
        with silence_output(): mert.load_model()
        mert.model.to(device)
        embedders[_MERT_NAME] = lambda wav: embed_mert_framewise(mert, wav, sr, device, channel)
    if _CLAP_GUD_NAME in want:
        embedders[_CLAP_GUD_NAME] = lambda wav: embed_clap_gud(wav, sr, device, channel)
    if _PANN_NAME in want:
        get_pann_model(device)    # fail-fast on missing checkpoint
        embedders[_PANN_NAME] = lambda wav: embed_pann(wav, sr, device, channel)
    ok(f"Embedding models ready ({len(embedders)} embedders).", prefix="REFS")

    dataset = _AudioDataset(todo, target_sr=44100, target_channels=2)
    loader  = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    processed = skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for wav, stem in tqdm(loader, desc="build-refs", dynamic_ncols=True,
                              mininterval=30.0, file=sys.stdout):
            if wav is None:
                skipped += 1; continue
            try:
                for name, fn in embedders.items():
                    out = emb_root / ch_name(name, channel) / f"{stem}.npy"
                    if args.resume and out.exists():
                        continue                       # per-model resume: don't recompute existing
                    atomic_save_npy(out, fn(wav).astype(np.float16))
            except Exception as e:  # noqa: BLE001
                err(f"Embedding error {stem}: {e}", prefix="REFS")
                skipped += 1; continue

            processed += 1
            if processed % 200 == 0:
                info(f"{processed}/{len(todo)} ({time.time()-t0:.0f}s)", prefix="REFS")

    ok(f"Embeddings done — processed: {processed}  skipped: {skipped}", prefix="REFS")

    info("Computing Fréchet reference stats (mu/cov)...", prefix="REFS")
    _recompute_stats(emb_root, stats_root, want, channel)
    ok(f"References ready under {data_dir}", prefix="REFS")


if __name__ == "__main__":
    main()
