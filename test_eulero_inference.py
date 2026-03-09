import argparse
from pathlib import Path

import torch
import torchaudio

from ar_spectra.models.eulero_inference import EuleroEncodeDecode
from ar_spectra.utils.console import ok, warn, err, info
from config import CHECKPOINT_DIR, DEFAULT_MODEL_CHECKPOINT, DATA_PATH, DEFAULT_DEVICE

def main():
    parser = argparse.ArgumentParser(description="Test Eulero Model Inference")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=str(DEFAULT_MODEL_CHECKPOINT),
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        default=str(Path(DATA_PATH) / "000/000002.mp3"),
        help="Path to an input audio file to test"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="reconstructed_test.wav",
        help="Path to save the reconstructed audio output"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=str(DEFAULT_DEVICE),
        help="Device to run inference on"
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    audio_path = Path(args.audio)
    output_path = Path(args.output)
    
    # 1. Check if checkpoint exists
    if not checkpoint_path.is_file():
        err(f"Checkpoint non trovato: {checkpoint_path}", prefix="CHECKPOINT")
        err("Assicurati di aver spostato la cartella 'checkpoint' all'interno di C-VAE e che contenga 'eulerodec.ckpt'", prefix="CHECKPOINT")
        return
        
    if not audio_path.is_file():
        err(f"File audio di test non trovato: {audio_path}", prefix="AUDIO")
        return

    device = torch.device(args.device)
    info(f"Using device: {device}", prefix="DEVICE")

    # 2. Inizializzare il codec
    info(f"Caricamento modello da: {checkpoint_path}...", prefix="CHECKPOINT")
    codec = EuleroEncodeDecode(checkpoint_path, device=device)
    ok("Modello caricato con successo!", prefix="CHECKPOINT")

    # 3. Caricare e processare l'audio
    info(f"Caricamento audio da: {audio_path}...", prefix="AUDIO")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Check sample rate - default C-VAE target sample rate is typically 44100
    if codec.sample_rate and sample_rate != codec.sample_rate:
        warn(f"Resampling from {sample_rate} to {codec.sample_rate}...", prefix="AUDIO")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=codec.sample_rate)
        waveform = resampler(waveform)
        sample_rate = codec.sample_rate
        
    # Check channels - ensure stereo/mono matches
    if codec.audio_channels:
        if waveform.shape[0] < codec.audio_channels:
            # Duplicate mono to stereo if needed
            waveform = waveform.repeat(codec.audio_channels, 1)
        elif waveform.shape[0] > codec.audio_channels:
            # Mixdown stereo to mono if needed
            waveform = waveform.mean(dim=0, keepdim=True)
            
    waveform = waveform.to(device)
    info(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}", prefix="AUDIO")

    # 4. Eseguire l'inferenza (encode & decode)
    info("Estrazione dei latenti (Encoding)...", prefix="MODEL")
    latents = codec.encode(waveform)
    info(f"Shape dei latenti: {latents.shape}", prefix="MODEL")
    info(f"Dtype dei latenti: {latents.dtype}", prefix="MODEL")

    info("Ricostruzione dell'audio (Decoding)...", prefix="MODEL")
    # Pass target_length to avoid padding frame mismatch
    recons = codec.decode(latents, target_length=waveform.shape[-1])
    info(f"Shape audio ricostruito: {recons.shape}", prefix="AUDIO")

    # 5. Salvare il risultato
    info(f"Salvataggio audio ricostruito in: {output_path}...", prefix="OUTPUT")
    recons_to_save = recons.squeeze(0).cpu()  # Rimuove la batch dimension
    torchaudio.save(str(output_path), recons_to_save, sample_rate)
    ok("Test completato con successo! Puoi ascoltare il file generato.", prefix="OUTPUT")

if __name__ == "__main__":
    main()
