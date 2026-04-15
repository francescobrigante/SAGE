import argparse
from pathlib import Path
import torch
import torchaudio

from ar_spectra.models.inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, info
from tqdm import tqdm
from config import (
    DEFAULT_MODEL_CHECKPOINT,
    DATA_PATH,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FILES,
    DEFAULT_AUDIO_EXTENSIONS,
    FMA_METADATA
)
    
   
def collect_audio_files(root: Path, audio_exts) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in audio_exts)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_expected_frames(module: torch.nn.Module) -> int | None:
    """Recursively search for img_size in any PatchEmbed module."""
    if hasattr(module, "img_size") and isinstance(module.img_size, (tuple, list)) and len(module.img_size) == 2:
        return int(module.img_size[1])
    for child in module.children():
        frames = get_expected_frames(child)
        if frames is not None:
            return frames
    return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test predictions using a trained Eulero model")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=str(DEFAULT_MODEL_CHECKPOINT),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(DATA_PATH),
        help="Path to the test audio directory",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated predictions")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Computation device (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_AUDIO_EXTENSIONS), help="Comma-separated list of audio file extensions to process")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max number of files to process (0 = all)")
    parser.add_argument("--fma-csv-path", type=str, default=FMA_METADATA, help="Path to FMA tracks.csv to filter only 'test' split")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    info(f"Using device: {device}", prefix="DEVICE")

    codec = EuleroEncodeDecode(args.model_checkpoint, device=device)

    audio_files = collect_audio_files(target_dir, audio_exts={ext.lower() for ext in args.extensions.split(",")})
    
    # Filter by FMA test split if metadata CSV is provided
    if args.fma_csv_path and Path(args.fma_csv_path).exists():
        import pandas as pd
        csv_path = Path(args.fma_csv_path).expanduser().resolve()
        info(f"Filtering dataset to 'test' split using {csv_path.name}...", prefix="DATALOADER")
        # FMA's tracks.csv contains header on the top. 
        # Usually index 0 is track_id, and ('set', 'split') defines test/train/val
        try:
            tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
            # Filter strictly for 'test' split AND 'small' subset to get exactly 800 files
            test_condition = (tracks[('set', 'split')] == 'test') & (tracks[('set', 'subset')] == 'small')
            test_tracks = tracks[test_condition].index.tolist()
            # FMA audio files are named like 000002.mp3
            test_ids = {f"{tid:06d}" for tid in test_tracks}
            
            # Deduplicate by stem, keeping shortest path to avoid recursive conversion folders
            dedup_map = {}
            for f in audio_files:
                stem = f.stem
                if stem in test_ids:
                    if stem not in dedup_map or len(f.parts) < len(dedup_map[stem].parts):
                        dedup_map[stem] = f
            audio_files = sorted(dedup_map.values())
        except Exception as e:
            warn(f"Failed to parse FMA CSV: {e}", prefix="DATALOADER")

    if not audio_files:
        warn(f"No audio files found under {target_dir}", prefix="DATALOADER")
        return

    ok(f"Found {len(audio_files)} files to evaluate", prefix="DATALOADER")

    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]
        ok(f"Processing first {len(audio_files)} files", prefix="DATALOADER")

    skipped = 0
    with torch.inference_mode():
        for audio_path in tqdm(audio_files, desc="Processing audio files", mininterval=2.0, dynamic_ncols=True):
            rel_path = audio_path.relative_to(target_dir)
            out_path = output_dir / rel_path.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
    
            if out_path.exists():
                continue

            try:
                waveform, sample_rate = torchaudio.load(audio_path)
    
                # Resample if needed
                if codec.sample_rate and sample_rate != codec.sample_rate:
                    waveform = torchaudio.transforms.Resample(sample_rate, codec.sample_rate)(waveform)
                    sample_rate = codec.sample_rate
    
                # Match channels (mono → stereo if model expects stereo)
                if codec.audio_channels:
                    if waveform.shape[0] < codec.audio_channels:
                        waveform = waveform.repeat(codec.audio_channels, 1)
                    elif waveform.shape[0] > codec.audio_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
    
                waveform = waveform.to(device)
    
                expected_frames = get_expected_frames(codec.autoencoder.encoder)
                if expected_frames is not None:
                    hop = codec.autoencoder._stft_config.hop_length
                    chunk_samples = (expected_frames - 1) * hop
                    chunks = list(torch.split(waveform, chunk_samples, dim=-1))
    
                    # Pad last chunk if shorter, track original length
                    pad_len = 0
                    if chunks[-1].shape[-1] < chunk_samples:
                        pad_len = chunk_samples - chunks[-1].shape[-1]
                        chunks[-1] = torch.nn.functional.pad(chunks[-1], (0, pad_len))
    
                    # Process in mini-batches to avoid OOM (each chunk ~1.5s of audio)
                    MAX_BATCH = 16
                    recons_chunks: list[torch.Tensor] = []
                    for b_start in range(0, len(chunks), MAX_BATCH):
                        batch = torch.stack(chunks[b_start : b_start + MAX_BATCH], dim=0)
                        latents = codec.encode(batch)
                        recons_b = codec.decode(latents, target_length=chunk_samples)
                        recons_chunks.extend(recons_b.unbind(dim=0))
    
                    # Trim padding from last chunk
                    if pad_len > 0:
                        recons_chunks[-1] = recons_chunks[-1][..., :chunk_samples - pad_len]
    
                    recons = torch.cat(recons_chunks, dim=-1).unsqueeze(0)  # (1, C, T_total)
                else:
                    latents = codec.encode(waveform)
                    recons = codec.decode(latents, target_length=waveform.shape[-1])
    
                recons_to_save = recons.squeeze(0).cpu()
                torchaudio.save(str(out_path), recons_to_save, sample_rate)
            except KeyboardInterrupt:
                warn("Process interrupted by user. Exiting early...", prefix="EVALUATION")
                break
            except Exception as e:
                skipped += 1
                warn(f"Skipped {audio_path.name}: {e}")
                continue

    ok("Finished processing all files.", prefix="EVALUATION")


if __name__ == "__main__":
    main()