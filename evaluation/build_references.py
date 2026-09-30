#!/usr/bin/env python3
# =============================================================================
# Reference artefacts of a clip set, built once per set (MoisesDB mixtures and
# stems, MusicCaps, Song Describer) before `evaluation.reconstruction` on them:
#
#   python -m evaluation.build_references dataset=musiccaps
#
# Settings: configs/build_references.yaml; the clip sets in configs/dataset/.
# Writes into the set's folder:
#   embeddings/clap-laion-audio/{stem}.npy      (N_chunks, 512)  → CLAP-audio cosine
#   embeddings/clap-laion-music/{stem}.npy      (N_chunks, 512)  → CLAP-music cosine
#   embeddings/MERT-v1-95M-4/{stem}.npy         (N_frames, 768)  → FAD-MERT
#   embeddings/clap-laion-audio-gud/{stem}.npy  (1, 512)         → FAD-CLAP
#   embeddings/pann-cnn14-16k/{stem}.npy        (1, 2048)        → FAD-PANN
#   stats_ours/<embedder>/{mu,cov}.npy          Fréchet reference statistics of the three FADs
#
# The embeddings come from the same functions the evaluator applies to the
# reconstructions, so references and predictions live in the same space.
# =============================================================================
from __future__ import annotations

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.common import (CLAP_AUDIO_NAME, CLAP_GUD_NAME, CLAP_MUSIC_NAME, MERT_NAME, AudioDataset,
                               atomic_save_npy, collect_clip_files, file_seed, first_item, seed_everything,
                               set_metric_weights, silence_output)
from evaluation.metrics.clap import embed_clap, embed_clap_gud
from evaluation.metrics.fad import PANN_NAME, compute_incremental_stats, embed_mert_framewise, embed_pann, get_pann_model
from sage.utils.console import err, info, ok, warn

EMBEDDERS = (CLAP_AUDIO_NAME, CLAP_MUSIC_NAME, MERT_NAME, CLAP_GUD_NAME, PANN_NAME)
STATS_EMBEDDERS = (MERT_NAME, CLAP_GUD_NAME, PANN_NAME)          # the three FADs


def _write_stats(emb_root: Path, stats_root: Path) -> None:
    for name in STATS_EMBEDDERS:
        npys = sorted((emb_root / name).glob("*.npy"))
        mu, cov = compute_incremental_stats(npys) if npys else (None, None)
        if mu is None:
            warn(f"Fewer than 2 vectors for {name} — stats skipped.", prefix="STATS")
            continue
        atomic_save_npy(stats_root / name / "mu.npy", mu.astype(np.float64))
        atomic_save_npy(stats_root / name / "cov.npy", cov.astype(np.float64))
        ok(f"stats_ours/{name}: mu{mu.shape} cov{cov.shape} ({len(npys)} files)", prefix="STATS")


def run(cfg: DictConfig) -> None:
    if cfg.get("dataset") is None:
        raise SystemExit("dataset=<name> is required: fma | moisesdb_mix | moisesdb_stems | musiccaps | song_describer")
    if cfg.dataset.protocol != "clips":
        raise SystemExit(f"dataset={cfg.dataset.name}: references are built for clip sets only "
                         "(the fma protocol caches its targets during the evaluation)")
    if not cfg.dataset.data_dir:
        raise SystemExit(f"dataset={cfg.dataset.name}: its data folder is not set (configs/paths)")
    set_metric_weights(cfg.paths)
    data_dir = Path(cfg.dataset.data_dir).expanduser().resolve()
    emb_root = data_dir / "embeddings"
    device = (torch.device("cpu") if cfg.device == "cpu" or not torch.cuda.is_available()
              else torch.device("cuda:0"))

    files = collect_clip_files(data_dir, cfg.max_files)
    if not files:
        raise SystemExit(f"No WAV files found in {data_dir}")
    if cfg.resume:
        files = [f for f in files if not all((emb_root / n / f"{f.stem}.npy").exists() for n in EMBEDDERS)]
    info(f"{len(files)} files to embed.", prefix="REFS")

    from fadtk.model_loader import CLAPLaionModel, MERTModel
    sr = 44100
    clap_audio, clap_music, mert = CLAPLaionModel("audio"), CLAPLaionModel("music"), MERTModel(layer=4)
    for ml in (clap_audio, clap_music, mert):
        with silence_output():
            ml.load_model()
        ml.model.to(device)
    get_pann_model(device)                                        # fail fast on a missing checkpoint
    embedders = {                                                 # name → wav [C, T] → (N, D)
        CLAP_AUDIO_NAME: lambda wav: embed_clap(clap_audio, wav, sr, device),
        CLAP_MUSIC_NAME: lambda wav: embed_clap(clap_music, wav, sr, device),
        MERT_NAME:       lambda wav: embed_mert_framewise(mert, wav, sr, device),
        CLAP_GUD_NAME:   lambda wav: embed_clap_gud(wav, sr, device),
        PANN_NAME:       lambda wav: embed_pann(wav, sr, device),
    }
    ok("Embedding models ready.", prefix="REFS")

    loader = DataLoader(AudioDataset(files, sr, target_channels=2), batch_size=1,
                        num_workers=cfg.num_workers, collate_fn=first_item,
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
    processed = skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for wav, stem in tqdm(loader, desc="references", dynamic_ncols=True, mininterval=30.0, file=sys.stdout):
            if wav is None:
                skipped += 1
                continue
            try:
                seed_everything(file_seed(cfg.seed, stem))
                for name, embed in embedders.items():
                    out = emb_root / name / f"{stem}.npy"
                    if cfg.resume and out.exists():
                        continue
                    atomic_save_npy(out, embed(wav).astype(np.float16))
            except Exception as e:  # noqa: BLE001
                err(f"Embedding error {stem}: {e}", prefix="REFS")
                skipped += 1
                continue
            processed += 1
            if processed % 200 == 0:
                info(f"{processed}/{len(files)} ({time.time() - t0:.0f}s)", prefix="REFS")
    ok(f"Embeddings done — processed: {processed}  skipped: {skipped}", prefix="REFS")

    _write_stats(emb_root, data_dir / "stats_ours")
    ok(f"References ready under {data_dir}", prefix="REFS")


@hydra.main(config_path="../configs", config_name="build_references", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
