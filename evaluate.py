import os
import argparse
from pathlib import Path

# Add project root to path so we can import ar_spectra
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio

from ar_spectra.models.eulero_inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, err, info
from tqdm import tqdm
from config import (
    DEFAULT_MODEL_CHECKPOINT,
    DATA_PATH,
    DEFAULT_DEVICE,
    DEFAULT_MAX_FILES
)
    
   
def collect_audio_files(root: Path, audio_exts) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in audio_exts)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument("--extensions", type=str, default=".wav,.flac,.mp3,.ogg,.m4a", help="Comma-separated list of audio file extensions to process")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max number of files to process (0 = all)")
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
    if not audio_files:
        warn(f"No audio files found under {target_dir}", prefix="DATALOADER")
        return

    ok(f"Found {len(audio_files)} files under {target_dir}", prefix="DATALOADER")

    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]
        ok(f"Processing first {len(audio_files)} files", prefix="DATALOADER")

    skipped = 0
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        rel_path = audio_path.relative_to(target_dir)
        out_path = output_dir / rel_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

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