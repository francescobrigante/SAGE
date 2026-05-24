#!/usr/bin/env python3
# =============================================================================
# evaluation/evaluate.py
#
# In-memory evaluation pipeline for a single checkpoint.
# No temporary WAV files are written to disk.
#
# Metrics computed per file:
#   - SI-SDR              (signal-to-distortion ratio)
#   - STFT loss           (multi-resolution log-magnitude L1)
#   - CDPAM               (perceptual similarity, optional)
#   - CLAP cosine score   (music + audio flavours, optional)
#
# Dataset-level (distributional):
#   - FAD MERT            (Fréchet distance via MERT-v1-95M embeddings)
#   - FAD CLAP-audio      (Fréchet distance via CLAP-audio embeddings)
#
# Target embeddings (CLAP/MERT) are cached to --shared-cache-dir on
# $SCRATCH so subsequent runs/checkpoints skip re-embedding.
#
# Usage:
#   python evaluation/evaluate.py \
#       --checkpoint checkpoints/sa2-step=100000.ckpt \
#       --target-dir /path/to/fma_large \
#       --output-dir runs/eval_sa2_large/sa2-step=100000 \
#       --shared-cache-dir /leonardo_scratch/IscrC_AHNetBio/eval_cache \
#       --device cuda \
#       --num-workers 4 \
#       --file-batch-size 4 \
#       --skip-cdpam
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Project root on sys.path so we can import ar_spectra + config
_EVAL_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from ar_spectra.models.inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, info, err
from config import (
    DATA_PATH,
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FILES,
    FMA_METADATA,
)
from utils import (
    # I/O
    write_csv,
    silence_output,
    collect_fma_files,
    # Metrics
    si_sdr,
    stft_loss,
    cdpam_score,
    # Cache helpers
    load_or_embed,
    target_cache_path,
    # Codec
    get_expected_frames,
    infer_batch,
    # Alignment
    batch_align,
)
# Embedding functions live in their respective compute_*.py modules
from compute_clap_score import cosine_sim, embed_clap
from compute_fad import embed_mert, compute_fad_from_embeddings

# numpy 1.24+ removed np.float — patch for CDPAM internals
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True


# ── Dataset ───────────────────────────────────────────────────

class _AudioDataset(Dataset):
    """Load and resample a list of audio files; returns (waveform, stem)."""

    def __init__(self, files: list[Path], target_sr: int, target_ch: int):
        self.files     = files
        self.target_sr = target_sr
        self.target_ch = target_ch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        try:
            wav, sr = torchaudio.load(p)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            if wav.shape[0] < self.target_ch:
                wav = wav.repeat(self.target_ch, 1)
            elif wav.shape[0] > self.target_ch:
                wav = wav.mean(0, keepdim=True).expand(self.target_ch, -1).clone()
            return wav, p.stem
        except Exception as e:
            warn(f"Load error {p.name}: {e}")
            return None, p.stem


# ── Argument parsing ──────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="In-memory C-VAE evaluation — single checkpoint, all metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .ckpt file.")
    p.add_argument("--target-dir", default=str(DATA_PATH),
                   help="Reference audio directory (FMA test split).")
    p.add_argument("--output-dir", required=True,
                   help="Per-checkpoint output root; metrics/ sub-dir is created here.")
    p.add_argument("--shared-cache-dir", type=Path, default=None,
                   help="$SCRATCH path for target embedding cache (shared across checkpoints).")
    p.add_argument("--device", default=DEFAULT_DEVICE,
                   help="Inference device (cuda / cpu).")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader prefetch workers (I/O parallel, not GPU).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Files processed simultaneously during codec inference.")
    p.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                   help="Limit number of files (0 = all).")
    p.add_argument("--fma-csv-path", default=FMA_METADATA,
                   help="FMA tracks.csv for test-split filtering.")
    p.add_argument("--extensions", default=",".join(DEFAULT_AUDIO_EXTENSIONS))
    # Metric toggles
    p.add_argument("--skip-cdpam",      action="store_true", help="Skip CDPAM (slow at scale).")
    p.add_argument("--skip-clap",       action="store_true", help="Skip CLAP cosine scoring.")
    p.add_argument("--skip-fad",        action="store_true", help="Skip FAD (MERT).")
    p.add_argument("--skip-fad-gudgud", action="store_true", help="Skip FAD (CLAP-audio).")
    p.add_argument("--clap-model",      default="both", choices=["music", "audio", "both"])
    p.add_argument("--resume",          action="store_true",
                   help="Skip this checkpoint if metrics/spectral.csv already exists.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    device     = torch.device(args.device)
    target_dir = Path(args.target_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"

    # --resume: skip if spectral.csv already exists
    if args.resume and (metrics_dir / "spectral.csv").exists():
        info(f"[RESUME] {output_dir.name} already evaluated — skipping.")
        return

    if not target_dir.is_dir():
        err(f"Target directory not found: {target_dir}"); return

    if args.shared_cache_dir is None and (not args.skip_fad or not args.skip_fad_gudgud):
        warn("--shared-cache-dir not set — FAD metrics require it; skipping FAD.")
        args.skip_fad = True
        args.skip_fad_gudgud = True

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect audio files ────────────────────────────────────
    audio_exts = {(e if e.startswith(".") else f".{e}").lower()
                  for e in args.extensions.split(",")}
    audio_files = collect_fma_files(target_dir, audio_exts, args.fma_csv_path, args.max_files)
    if not audio_files:
        err(f"No audio files found in {target_dir}"); return
    ok(f"Files to evaluate: {len(audio_files)}", prefix="EVAL")

    # ── Load codec ─────────────────────────────────────────────
    codec = EuleroEncodeDecode(args.checkpoint, device=device)
    sr: int = codec.sample_rate or 44100
    ch: int = codec.audio_channels or 2
    ok(f"Codec: sr={sr} Hz, channels={ch}", prefix="EVAL")

    # Derive codec chunk size from Swin PatchEmbed temporal dimension
    expected_frames = (get_expected_frames(codec.autoencoder.encoder)
                       if hasattr(codec.autoencoder, "encoder") else None)
    stft_cfg   = getattr(codec.autoencoder, "_stft_config", None)
    hop        = stft_cfg.hop_length if (expected_frames and stft_cfg) else None
    chunk_samples: int | None = ((expected_frames - 1) * hop) if (expected_frames and hop) else None
    if chunk_samples:
        info(f"Codec chunk: {chunk_samples} samples ({chunk_samples/sr:.2f}s)", prefix="EVAL")

    # ── Load embedding models ──────────────────────────────────
    from fadtk.model_loader import CLAPLaionModel, MERTModel

    clap_flavours = (["music", "audio"] if args.clap_model == "both" else [args.clap_model]) \
                    if not args.skip_clap else []

    ml_clap_music = ml_clap_audio = ml_fad = None

    if "music" in clap_flavours:
        ml_clap_music = CLAPLaionModel("music")
        with silence_output():
            ml_clap_music.load_model()
        ok("CLAP-music loaded", prefix="EMBED")

    # CLAP-audio is shared by CLAP-audio scoring AND FAD-gudgud
    if "audio" in clap_flavours or not args.skip_fad_gudgud:
        ml_clap_audio = CLAPLaionModel("audio")
        with silence_output():
            ml_clap_audio.load_model()
        ok("CLAP-audio loaded", prefix="EMBED")

    if not args.skip_fad:
        ml_fad = MERTModel(size="v1-95M", layer=12)
        with silence_output():
            ml_fad.load_model()
        ok("MERT loaded", prefix="EMBED")
        # PyTorch ≥ 2.2 weight-norm patch for pos_conv_embed
        _patch_mert_pos_conv(ml_fad)

    # ── Accumulators ───────────────────────────────────────────
    spectral_rows:  list[dict] = []
    cdpam_rows:     list[dict] = []
    clap_rows:      list[dict] = []
    pred_embs_clap_music: list[np.ndarray] = []
    pred_embs_clap_audio: list[np.ndarray] = []
    pred_embs_mert:       list[np.ndarray] = []
    skipped = 0

    # ── DataLoader (prefetch from disk in parallel) ────────────
    dataset = _AudioDataset(audio_files, sr, ch)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda b: b,            # variable-length → list
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── Main inference + metric loop ───────────────────────────
    import time
    last_log_time = time.time()
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", unit="batch",
                                               dynamic_ncols=True, mininterval=10.0)):
            # Filter failed loads
            valid = [(wav.to(device), stem) for wav, stem in batch if wav is not None]
            if not valid:
                skipped += len(batch); continue

            wavs_gpu = [v[0] for v in valid]
            stems    = [v[1] for v in valid]

            # ── Codec inference (batched across files) ─────────
            try:
                if chunk_samples:
                    preds_cpu = infer_batch(codec, wavs_gpu, chunk_samples)
                else:
                    preds_cpu = [
                        codec.decode(codec.encode(w.unsqueeze(0)),
                                     target_length=w.shape[-1]).squeeze(0).cpu()
                        for w in wavs_gpu
                    ]
            except Exception as e:
                err(f"Inference error: {e}", prefix="EVAL")
                skipped += len(valid); continue

            # ── Per-file metrics ───────────────────────────────
            for wav_gpu, pred_cpu, stem in zip(wavs_gpu, preds_cpu, stems):
                try:
                    wav_cpu  = wav_gpu.cpu()
                    pred_cpu = pred_cpu.cpu()

                    # Trim to matching length
                    n = min(wav_cpu.shape[-1], pred_cpu.shape[-1])
                    wav_t  = wav_cpu[..., :n]
                    pred_t = pred_cpu[..., :n]

                    # Cross-correlation alignment (1-second max shift)
                    t_aln, p_aln, _ = batch_align(wav_t.unsqueeze(0), pred_t.unsqueeze(0), sr)
                    t_aln = t_aln.squeeze(0)
                    p_aln = p_aln.squeeze(0)

                    # Spectral metrics
                    spectral_rows.append({
                        "file":      stem,
                        "si_sdr":    si_sdr(t_aln, p_aln),
                        "stft_loss": stft_loss(t_aln, p_aln),
                    })

                    # CDPAM (optional, GPU if available)
                    if not args.skip_cdpam:
                        try:
                            score = cdpam_score(t_aln, p_aln, sr, device=device)
                            cdpam_rows.append({"file": stem, "cdpam": score})
                        except Exception as e:
                            warn(f"CDPAM error {stem}: {e}", prefix="CDPAM")

                    # CLAP cosine + embedding accumulation
                    clap_row: dict = {"file": stem}

                    if ml_clap_music is not None:
                        t_emb = load_or_embed(
                            ml_clap_music, embed_clap, wav_cpu, sr, device,
                            target_cache_path(args.shared_cache_dir, ml_clap_music.name, stem),
                        )
                        p_emb = embed_clap(ml_clap_music, pred_cpu, sr, device)
                        clap_row["clap_music"] = cosine_sim(t_emb.mean(0), p_emb.mean(0))
                        pred_embs_clap_music.append(p_emb)

                    if ml_clap_audio is not None:
                        t_emb = load_or_embed(
                            ml_clap_audio, embed_clap, wav_cpu, sr, device,
                            target_cache_path(args.shared_cache_dir, ml_clap_audio.name, stem),
                        )
                        p_emb = embed_clap(ml_clap_audio, pred_cpu, sr, device)
                        if "audio" in clap_flavours:
                            clap_row["clap_audio"] = cosine_sim(t_emb.mean(0), p_emb.mean(0))
                        if not args.skip_fad_gudgud:
                            pred_embs_clap_audio.append(p_emb)

                    if ml_fad is not None:
                        # Cache target embedding; accumulate pred in RAM
                        load_or_embed(
                            ml_fad, embed_mert, wav_cpu, sr, device,
                            target_cache_path(args.shared_cache_dir, ml_fad.name, stem),
                        )
                        pred_embs_mert.append(embed_mert(ml_fad, pred_cpu, sr, device))

                    if len(clap_row) > 1:
                        clap_rows.append(clap_row)

                except Exception as e:
                    warn(f"Metrics error {stem}: {e}", prefix="EVAL")
                    skipped += 1

            # Timed progress logging for SLURM stdout
            cur_time = time.time()
            if cur_time - last_log_time >= 10.0 or batch_idx + 1 == total_batches:
                print(f"[EVAL] Batch {batch_idx + 1}/{total_batches} ({ (batch_idx + 1)/total_batches * 100:.1f}%)", flush=True)
                last_log_time = cur_time

    ok(f"Processing done — skipped: {skipped}", prefix="EVAL")

    # ── Write per-file CSVs ────────────────────────────────────
    if spectral_rows:
        write_csv(metrics_dir / "spectral.csv", ["file", "si_sdr", "stft_loss"], spectral_rows)
        ok(f"spectral.csv ({len(spectral_rows)} files)", prefix="EVAL")

    if cdpam_rows:
        write_csv(metrics_dir / "cdpam.csv", ["file", "cdpam"], cdpam_rows)
        ok(f"cdpam.csv ({len(cdpam_rows)} files)", prefix="EVAL")

    if clap_rows:
        fields = ["file"] + [k for k in clap_rows[0] if k != "file"]
        write_csv(metrics_dir / "clap_score.csv", fields, clap_rows)
        ok(f"clap_score.csv ({len(clap_rows)} files)", prefix="EVAL")

    # ── FAD (distributional, needs all pred embeddings) ────────
    if ml_fad is not None and pred_embs_mert and args.shared_cache_dir:
        compute_fad_from_embeddings(
            ml_fad, audio_files, pred_embs_mert,
            args.shared_cache_dir, metrics_dir / "fad_mert.csv", "mert",
        )

    if (ml_clap_audio is not None and not args.skip_fad_gudgud
            and pred_embs_clap_audio and args.shared_cache_dir):
        compute_fad_from_embeddings(
            ml_clap_audio, audio_files, pred_embs_clap_audio,
            args.shared_cache_dir, metrics_dir / "fad_gudgud.csv", "clap-audio",
        )

    ok("Evaluation complete.", prefix="EVAL")


# ── MERT weight-norm patch ─────────────────────────────────────

def _patch_mert_pos_conv(ml) -> None:
    """Fix MERT pos_conv_embed weight-norm parametrization for PyTorch ≥ 2.2."""
    if not (hasattr(ml, "model") and hasattr(ml.model, "encoder")):
        return
    enc = ml.model.encoder
    if not hasattr(enc, "pos_conv_embed"):
        return
    m_conv = enc.pos_conv_embed.conv
    if not hasattr(m_conv, "parametrizations"):
        return
    try:
        from huggingface_hub import hf_hub_download
        ckpt_p = hf_hub_download(ml.huggingface_id, "pytorch_model.bin")
        sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if "encoder.pos_conv_embed.conv.weight_g" in sd:
            with torch.no_grad():
                m_conv.parametrizations.weight.original0.copy_(
                    sd["encoder.pos_conv_embed.conv.weight_g"])
                m_conv.parametrizations.weight.original1.copy_(
                    sd["encoder.pos_conv_embed.conv.weight_v"])
            info("MERT pos_conv_embed patched.", prefix="EMBED")
    except Exception as e:
        warn(f"MERT patch failed (non-fatal): {e}", prefix="EMBED")


if __name__ == "__main__":
    main()
