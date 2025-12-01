import argparse
from pathlib import Path

import torch
import torchaudio

from ar_spectra.models.eulero_inference import EuleroEncodeDecode
from rich.console import Console
from tqdm import tqdm
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    
   
def collect_audio_files(root: Path, audio_exts) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in audio_exts)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test predictions using a trained Eulero model")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/home/cerovaz/repos/ICML/Eulero_BackBone/checkpoints/norm_111_bis_epoch_024_rigenerated.ckpt",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/home/cerovaz/repos/data/jamendo_full/test_trimmed",
        help="Path to the test audio directory",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated predictions")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--extensions", type=str, default=".wav,.flac,.mp3,.ogg,.m4a", help="Comma-separated list of audio file extensions to process")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    codec = EuleroEncodeDecode(args.model_checkpoint, device=device)

    audio_files = collect_audio_files(target_dir, audio_exts={ext.lower() for ext in args.extensions.split(",")})
    if not audio_files:
        warn(f"No audio files found under {target_dir}")
        return

    ok(f"Found {len(audio_files)} files under {target_dir}")

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        rel_path = audio_path.relative_to(target_dir)
        out_path = output_dir / rel_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)

        latents, info = codec.encode_audio(waveform, pack_complex=False)
        recons = codec.decode_audio(latents, info, pack_complex=False)

        recons_to_save = recons.squeeze(0).cpu()
        torchaudio.save(str(out_path), recons_to_save, sample_rate)

    ok("Finished processing all files.")


if __name__ == "__main__":
    main()