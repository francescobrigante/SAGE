#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate_swin_10s.py
# Reconstruction eval for Swin C-VAE checkpoints on the colleague's
# chunks_mix_original dataset (1998 × 10s WAVs). Same structure as
# evaluate_sao_10s.py but loads a Swin ckpt via EuleroEncodeDecode.
#
# Reference data layout (data_dir = chunks_mix_original/original/):
#   embeddings/clap-laion-{audio,music}/{stem}.npy — pre-computed ref CLAP
#   stats/{model}/mu.npy + cov.npy                 — pre-computed Fréchet stats
#
# MERT layer=4 for predictions (matches pre-computed MERT-v1-95M-4 ref stats).
# =============================================================================
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_EVAL_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _EVAL_DIR.parent
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / "src"), str(_EVAL_DIR),
           str(_EVAL_DIR / "stereo_diagnosis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ar_spectra.models.inference import EuleroEncodeDecode
from c_vae.swin.varlen import resolve
from ar_spectra.utils.console import ok, warn, info, err
from losses import compute_sdr_and_sisdr, stft_loss, spectral_losses, cdpam_score
from utils import (atomic_save_npy, silence_output, ch_name,
                   CHANNEL_MID, CHANNEL_SIDE, CHANNELS)
from compute_clap_score import embed_clap, cosine_sim
from compute_fad import (embed_mert_framewise, embed_pann, get_pann_model, PANN_NAME)
from fadtk.model_loader import CLAPLaionModel, MERTModel
from fadtk.fad import calc_frechet_distance
from ms_eval import MS_COLUMNS, ms_metrics_row   # torch-only; shared with the SOTA evaluator

if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

_MERT_NAME       = "MERT-v1-95M-4"
_CLAP_AUDIO_NAME = "clap-laion-audio"
_CLAP_MUSIC_NAME = "clap-laion-music"
_PANN_NAME       = PANN_NAME               # pann-cnn14-16k (from compute_fad)
_SKIP_DIRS       = frozenset({"embeddings", "stats", "stats_ours", "convert", "metrics", "parts"})

_CSV_SCHEMA = {
    "spectral":   ["file", "si_sdr", "sdr", "stft_loss", "mel_loss"],
    "cdpam":      ["file", "cdpam"],
    "clap_music": ["file", "cosine"],
    "clap_audio": ["file", "cosine"],
    # Pure inference time (encode→decode), no metric/loading cost.
    "timing":     ["file", "infer_sec", "audio_sec"],
    # Stereo-imaging metrics (--compute-ms-metrics / --ms-only): only written when
    # one of those flags is set; absent CSV → notebook reports NaN, so old runs
    # stay compatible. Schema shared with the SOTA evaluator (see ms_eval).
    "ms_metrics": MS_COLUMNS,
}

def _compute_ms_metrics(ref: torch.Tensor, pred: torch.Tensor, sr: int) -> dict:
    """Stereo-imaging metrics on an aligned (2, T) ref/pred pair.

    Thin wrapper over stereo_diagnosis/ms_eval.ms_metrics_row — the same function
    the SOTA evaluator calls, so the SAGE row and the baseline row are produced by
    identical code and are directly comparable.
    """
    return ms_metrics_row(ref, pred, sr)


def _pad_for_swin(wav: torch.Tensor, hop_length: int,
                  num_downsamples: int = 2) -> tuple[torch.Tensor, int]:
    orig = wav.shape[-1]
    mult = 2 ** num_downsamples
    W    = (orig // hop_length) + 1
    pad  = (mult - W % mult) % mult
    target = max(orig, (W + pad - 1) * hop_length)
    if target > orig:
        wav = F.pad(wav, (0, target - orig))
    return wav, orig


class _AudioDataset(Dataset):
    def __init__(self, files: list[Path], target_sr: int, target_channels: int = 2):
        self.files = files; self.target_sr = target_sr; self.target_channels = target_channels

    def __len__(self) -> int: return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            C = wav.shape[0]
            if C > self.target_channels:   wav = wav[:self.target_channels]
            elif C < self.target_channels: wav = wav.repeat((self.target_channels + C - 1) // C, 1)[:self.target_channels]
            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}"); return None, p.stem


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)


def _load_ref_emb(data_dir: Path, model_name: str, stem: str) -> Optional[np.ndarray]:
    p = data_dir / "embeddings" / model_name / f"{stem}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def _collect_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(
        p for p in data_dir.glob("*.wav")
        if not _SKIP_DIRS.intersection(p.relative_to(data_dir).parts[:-1])
    )
    return files[:max_files] if max_files > 0 else files


def _fad_from_stats(data_dir: Path, model_name: str, pred_emb_dir: Path,
                    csv_path: Path, label: str) -> None:
    # Prefer OUR self-consistent stats (recompute_ref_stats_10s.py) over the
    # colleague's pre-computed stats/ — the latter live in a different MERT
    # space and leave a spurious FAD floor (~5.5) even on perfect audio.
    if (data_dir / "stats_ours" / model_name / "mu.npy").exists():
        stats_root = data_dir / "stats_ours"
        info(f"Using OUR recomputed ref stats for {model_name}.", prefix="FAD")
    else:
        stats_root = data_dir / "stats"
    ref_mu_p  = stats_root / model_name / "mu.npy"
    ref_cov_p = stats_root / model_name / "cov.npy"
    if not ref_mu_p.exists():
        warn(f"No ref stats for {model_name} — FAD skipped.", prefix="FAD"); return
    npys = sorted(pred_emb_dir.glob("*.npy")) if pred_emb_dir.is_dir() else []
    if not npys:
        warn(f"No pred embeddings for {model_name} — FAD skipped.", prefix="FAD"); return
    ref_mu  = np.load(ref_mu_p).astype(np.float64)
    ref_cov = np.load(ref_cov_p).astype(np.float64)
    from compute_fad import compute_incremental_stats
    pred_mu, pred_cov = compute_incremental_stats(npys)
    
    if pred_mu is None:
        pred_mu = np.zeros_like(ref_mu)
        pred_cov = np.zeros_like(ref_cov)
    score = calc_frechet_distance(ref_mu, ref_cov, pred_mu, pred_cov)
    ok(f"FAD {label}: {score:.6f}", prefix="FAD")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "score"])
        w.writeheader(); w.writerow({"model": label, "score": score})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",    required=True, type=Path)
    p.add_argument("--data-dir",      required=True, type=Path,
                   help="Path to chunks_mix_original/original/")
    p.add_argument("--output-dir",    required=True, type=Path)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--max-files",     type=int, default=0)
    p.add_argument("--num-downsamples", type=int, default=None,
                   help="Override Swin downsamples (auto-detected from checkpoint).")
    p.add_argument("--skip-cdpam",    action="store_true")
    p.add_argument("--sdr-only",      action="store_true")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--fad-gud-only",  action="store_true",
                   help="Compute ONLY fad_gudgud (whole-file CLAP, 48kHz, float): skip "
                        "all other metrics. Overwrites fad_gudgud.csv.")
    p.add_argument("--new-metrics-only", action="store_true",
                   help="Compute ONLY fad_pann (PANN Cnn14 whole-file FAD): skip everything "
                        "else. Overwrites fad_pann.csv, leaves every other CSV and _done untouched.")
    p.add_argument("--ms-only", action="store_true",
                   help="Compute ONLY the stereo-imaging metrics (ms_metrics.csv): one decode "
                        "pass, NO embedding model loaded. Backfill twin of --new-metrics-only; "
                        "leaves every other CSV and _done untouched.")
    p.add_argument("--channel", default=CHANNEL_MID, choices=list(CHANNELS),
                   help="Channel fed to the FAD embedders and to CDPAM. 'mid' = (L+R)/2 "
                        "(historical mono downmix, default -> results unchanged); 'side' = "
                        "(L-R)/2. In 'side' mode ONLY the three FAD sources and CDPAM are "
                        "computed: SDR/STFT/mel/CLAP-cosine/ms_metrics are channel-agnostic "
                        "or already measured on the Mid, so recomputing them is waste. "
                        "Embeddings and ref stats are read/written under '<model>-side/'. "
                        "Use a SEPARATE --output-dir: the CSV names are unchanged.")
    p.add_argument("--compute-ms-metrics", action="store_true",
                   help="ADD stereo-imaging metrics (width_bias, d_width, sisdr_s, sisdr_m) to the "
                        "default per-file metrics, computed in-memory on the (ref, recon) pair. "
                        "Opt-in: off by default → old runs unchanged. Writes ms_metrics.csv.")
    p.add_argument("--varlen", default=None,
                   help="Variable-length seam fix on the collapsed Swin stages: a preset "
                        "from config/inference/varlen.yaml (tri2, hard2, tri4, ...) or 'off' "
                        "for the original single-phase attention. Default = the project "
                        "default in that file. The mode actually applied is recorded in "
                        "metrics/varlen.json; a run without that file is a baseline run.")
    return p.parse_args()


def main() -> None:
    args        = _parse_args()
    # Modalità Side: si calcolano SOLO le tre sorgenti FAD e CDPAM. SDR/STFT/mel,
    # la cosine CLAP e ms_metrics sono già misurate sul Mid o indipendenti dal
    # canale: rifarle sarebbe spreco e sovrascriverebbe risultati validi.
    side_mode   = args.channel == CHANNEL_SIDE
    CH          = args.channel
    ckpt_path   = args.checkpoint.expanduser().absolute()
    data_dir    = args.data_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    metrics_dir = output_root / ckpt_path.stem / "metrics"
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.fad_gud_only and args.resume and (metrics_dir / "fad_gudgud.csv").exists():
        info("[RESUME] fad_gudgud.csv exists — skipping.", prefix="EVAL"); return
    elif (args.new_metrics_only and args.resume
          and (metrics_dir / "fad_pann.csv").exists()):
        info("[RESUME] fad_pann.csv exists — skipping.", prefix="EVAL"); return
    elif args.ms_only and args.resume and (metrics_dir / "ms_metrics.csv").exists():
        info("[RESUME] ms_metrics.csv exists — skipping.", prefix="EVAL"); return
    elif (not args.fad_gud_only and not args.new_metrics_only and not args.ms_only
          and args.resume and (metrics_dir / "_done").exists()):
        info("[RESUME] already done — skipping.", prefix="EVAL"); return

    audio_files = _collect_files(data_dir, args.max_files)
    if not audio_files:
        err(f"No WAV files in {data_dir}"); return
    info(f"{len(audio_files)} files", prefix="EVAL")

    parts_dir = metrics_dir / "parts"
    pred_root = parts_dir / "pred"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Per-file resume, so a run killed by a wall limit can be finished by a second
    # job instead of restarting. A stem counts as done only when its LAST artefact
    # exists — the PANN embedding, written after every CSV row and every other npy
    # for that file. Keying on anything earlier (a timing row, say) could skip a
    # file whose tail metrics were never written. Redoing one file costs nothing:
    # the merge below keys on the stem and keeps the first row, so duplicates in
    # parts/*.csv collapse. Restricted to the full run: the --*-only passes have
    # their own resume, at the granularity of a whole CSV.
    if args.resume and not (args.sdr_only or args.ms_only or args.fad_gud_only
                            or args.new_metrics_only or side_mode):
        _done_dir = pred_root / ch_name(_PANN_NAME, CH)
        _finished = {f.stem for f in _done_dir.glob("*.npy")} if _done_dir.is_dir() else set()
        if _finished:
            audio_files = [f for f in audio_files if f.stem not in _finished]
            info(f"resume: {len(_finished)} già fatti, restano {len(audio_files)}",
                 prefix="EVAL")

    info(f"Loading Swin checkpoint: {ckpt_path.name}...", prefix="EVAL")
    codec      = EuleroEncodeDecode(ckpt_path, device=device, varlen=args.varlen)
    sr: int    = codec.sample_rate or 44100
    ch: int    = codec.audio_channels or 2
    stft_cfg   = getattr(codec.autoencoder, "_stft_config", None)
    if stft_cfg is None:
        err(f"No STFT config in {ckpt_path.name}"); return
    hop_length: int = stft_cfg.hop_length
    if args.num_downsamples is not None:
        num_downsamples = args.num_downsamples
    else:
        try:
            num_downsamples = len(codec.autoencoder.encoder.depths) - 1
        except AttributeError:
            num_downsamples = 2
    ok(f"sr={sr} ch={ch} hop={hop_length} downsamples={num_downsamples}", prefix="EVAL")

    if not args.sdr_only and not args.ms_only:
        info("Loading embedding models...", prefix="EVAL")
        embed_models = []
        if not args.fad_gud_only and not args.new_metrics_only:
            # MERT (layer 4) drives fad_mert → full run only (not --new-metrics-only,
            # which computes only fad_pann, nor --fad-gud-only).
            mert_ml       = MERTModel(layer=4)
            embed_models.append(mert_ml)
            # CLAP only feeds the CLAP cosine metric → full run only.
            if not args.new_metrics_only and not side_mode:
                clap_audio_ml = CLAPLaionModel("audio")
                clap_music_ml = CLAPLaionModel("music")
                embed_models.extend([clap_audio_ml, clap_music_ml])

        for _ml in embed_models:
            with silence_output(): _ml.load_model()
            _ml.model.to(device)
        # PANN drives fad_pann → full run + --new-metrics-only. Fail-fast on missing ckpt.
        if not args.fad_gud_only:
            get_pann_model(device)
        ok("Embedding models ready.", prefix="EVAL")

    dataset = _AudioDataset(audio_files, sr, target_channels=ch)
    loader  = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                         collate_fn=lambda b: b[0],
                         persistent_workers=args.num_workers > 0,
                         prefetch_factor=4 if args.num_workers > 0 else None)

    processed = skipped = 0
    t0 = time.time()

    with torch.no_grad():
        for wav, stem in tqdm(loader, desc=ckpt_path.stem, dynamic_ncols=True,
                               mininterval=30.0, file=sys.stdout):
            if wav is None:
                skipped += 1; continue

            try:
                wav_padded, orig_len = _pad_for_swin(wav, hop_length, num_downsamples)
                wav_gpu  = wav_padded.unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                latents  = codec.encode(wav_gpu, deterministic=False)
                decoded  = codec.decode(latents, target_length=wav_padded.shape[-1])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_sec = time.perf_counter() - _t0
                n        = min(orig_len, decoded.shape[-1])
                wav_ref  = wav[..., :n]
                pred     = decoded[0, ..., :n].cpu().float()
            except Exception as e:
                err(f"Inference error {stem}: {e}", prefix="EVAL")
                skipped += 1; continue

            if (not args.fad_gud_only and not args.new_metrics_only
                    and not args.ms_only and not side_mode):
                _append_csv(parts_dir / "timing.0.csv", _CSV_SCHEMA["timing"],
                            {"file": stem, "infer_sec": infer_sec, "audio_sec": orig_len / sr})
                sdr_val, sisdr_val = compute_sdr_and_sisdr(wav_ref, pred)
                _append_csv(parts_dir / "spectral.0.csv", _CSV_SCHEMA["spectral"],
                            {"file": stem, "si_sdr": sisdr_val, "sdr": sdr_val,
                             **spectral_losses(wav_ref, pred)})

            # Stereo-imaging metrics: ADDED to a full run (--compute-ms-metrics) or the
            # sole output of a backfill pass (--ms-only). Needs a true stereo pair.
            if ((args.compute_ms_metrics or args.ms_only) and not side_mode
                    and not args.fad_gud_only and not args.new_metrics_only
                    and wav_ref.shape[0] == 2 and pred.shape[0] == 2):
                try:
                    _append_csv(parts_dir / "ms_metrics.0.csv", _CSV_SCHEMA["ms_metrics"],
                                {"file": stem, **_compute_ms_metrics(wav_ref, pred, sr)})
                except Exception as e:
                    warn(f"MS metrics {stem}: {e}", prefix="MS")

            if (not args.skip_cdpam and not args.sdr_only and not args.fad_gud_only
                    and not args.new_metrics_only and not args.ms_only):
                try:
                    _append_csv(parts_dir / "cdpam.0.csv", _CSV_SCHEMA["cdpam"],
                                {"file": stem,
                                 "cdpam": cdpam_score(wav_ref, pred, sr, device=device,
                                                      channel=CH)})
                except Exception as e:
                    warn(f"CDPAM {stem}: {e}", prefix="CDPAM")

            if not args.sdr_only and not args.ms_only:
                try:
                    if not args.fad_gud_only:
                        # CLAP cosine + CLAP pred embeddings → full run only.
                        if not args.new_metrics_only and not side_mode:
                            ref_ca = _load_ref_emb(data_dir, _CLAP_AUDIO_NAME, stem)
                            p_ca   = embed_clap(clap_audio_ml, pred, sr, device)
                            if ref_ca is not None:
                                _append_csv(parts_dir / "clap_audio.0.csv", _CSV_SCHEMA["clap_audio"],
                                            {"file": stem,
                                             "cosine": float(cosine_sim(ref_ca.mean(0), p_ca.mean(0)))})
                            atomic_save_npy(pred_root / _CLAP_AUDIO_NAME / f"{stem}.npy", p_ca.astype(np.float16))

                            ref_cm = _load_ref_emb(data_dir, _CLAP_MUSIC_NAME, stem)
                            p_cm   = embed_clap(clap_music_ml, pred, sr, device)
                            if ref_cm is not None:
                                _append_csv(parts_dir / "clap_music.0.csv", _CSV_SCHEMA["clap_music"],
                                            {"file": stem,
                                             "cosine": float(cosine_sim(ref_cm.mean(0), p_cm.mean(0)))})
                            atomic_save_npy(pred_root / _CLAP_MUSIC_NAME / f"{stem}.npy", p_cm.astype(np.float16))

                        # MERT layer-4 framewise → fad_mert source (full run only).
                        if not args.new_metrics_only:
                            p_mert = embed_mert_framewise(mert_ml, pred, sr, device, CH)
                            atomic_save_npy(pred_root / ch_name(_MERT_NAME, CH) / f"{stem}.npy", p_mert.astype(np.float16))

                    # CLAP-gud whole-file → fad_gudgud source (full + --fad-gud-only).
                    if not args.new_metrics_only:
                        from compute_clap_score import embed_clap_gud
                        p_clap_gud = embed_clap_gud(pred, sr, device, CH)
                        atomic_save_npy(pred_root / ch_name("clap-laion-audio-gud", CH) / f"{stem}.npy", p_clap_gud.astype(np.float16))

                    # PANN Cnn14 whole-file → fad_pann source (full + --new-metrics-only).
                    if not args.fad_gud_only:
                        p_pann = embed_pann(pred, sr, device, CH)
                        atomic_save_npy(pred_root / ch_name(_PANN_NAME, CH) / f"{stem}.npy", p_pann.astype(np.float16))

                except Exception as e:
                    warn(f"Embedding error {stem}: {e}", prefix="EMBED")

            processed += 1
            if processed % 200 == 0:
                info(f"{processed}/{len(audio_files)} ({time.time()-t0:.0f}s)", prefix="EVAL")

    ok(f"Done — processed: {processed}  skipped: {skipped}", prefix="EVAL")

    # --new-metrics-only writes no per-file CSV (only the scalar fad_pann below);
    # --ms-only writes exactly one and touches nothing else.
    merge_names = ([] if args.new_metrics_only
                   else ["ms_metrics"] if args.ms_only
                   else list(_CSV_SCHEMA))
    for name in merge_names:
        cols = _CSV_SCHEMA[name]
        seen: dict[str, dict] = {}
        for f in sorted(parts_dir.glob(f"{name}.*.csv")):
            with open(f, newline="") as fh:
                for r in _csv.DictReader(fh):
                    seen.setdefault(r["file"], dict(r))
        if seen:
            out = metrics_dir / f"{name}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader(); w.writerows(seen.values())
            ok(f"{name}.csv ({len(seen)} files)", prefix="MERGE")

    if not args.sdr_only and not args.ms_only:
        if not args.fad_gud_only and not args.new_metrics_only:
            _fad_from_stats(data_dir, ch_name(_MERT_NAME, CH), pred_root / ch_name(_MERT_NAME, CH),
                            metrics_dir / "fad_mert.csv",       ch_name(_MERT_NAME, CH))

        if not args.new_metrics_only:
            _fad_from_stats(data_dir, ch_name("clap-laion-audio-gud", CH),
                            pred_root / ch_name("clap-laion-audio-gud", CH),
                            metrics_dir / "fad_gudgud.csv",     ch_name("clap-laion-audio-gud", CH))

        if not args.fad_gud_only:
            _fad_from_stats(data_dir, ch_name(_PANN_NAME, CH), pred_root / ch_name(_PANN_NAME, CH),
                            metrics_dir / "fad_pann.csv", ch_name(_PANN_NAME, CH))

    if not args.fad_gud_only and not args.new_metrics_only and not args.ms_only:
        # Provenance: which attention mode produced these numbers. Needed because
        # the project default is now a multi-phase preset, so rows computed before
        # and after the fix live side by side in the same table and are otherwise
        # indistinguishable. Rule: a metrics/ directory WITHOUT this file is a
        # baseline (single-phase) run.
        _vl = resolve(codec.varlen_mode)
        (metrics_dir / "varlen.json").write_text(json.dumps(
            {"mode": codec.varlen_mode,
             "blocks": codec.varlen_blocks,
             "phases": list(_vl.phases) if _vl else [],
             "combine": _vl.combine if _vl else None}, indent=2) + "\n")
        (metrics_dir / "_done").touch()
        ok(f"_done → {metrics_dir}  (varlen={codec.varlen_mode})", prefix="EVAL")


if __name__ == "__main__":
    main()
