from pathlib import Path
import json
from typing import Any, Dict

import torch
import torchaudio
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ar_spectra.training_utils.initialization import build_autoencoder_from_cfg
from ar_spectra.models.autoencoder import AutoEncoder
from rich.console import Console

console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")


def _extract_autoencoder_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the parameters that belong to the AutoEncoder module.

    Training checkpoints created via ``AutoencoderTrainingWrapper`` store
    weights under ``engine.autoencoder.*`` while direct exports may already
    use ``autoencoder.*`` or plain module prefixes (``encoder.``, ``decoder.``).
    This helper normalizes all those layouts so inference consistently receives
    the bare autoencoder state dict.
    """

    prefixes = ("engine.autoencoder.", "autoencoder.")
    extracted: Dict[str, Any] = {}

    for key, value in state_dict.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                matched = True
                break
        if not matched and (key.startswith("encoder.") or key.startswith("decoder.") or key.startswith("bottleneck.")):
            extracted[key] = value

    return extracted or state_dict


def load_checkpoint(autoencoder: AutoEncoder, ckpt_path: Path) -> AutoEncoder:
    """Load state dict from a Lightning checkpoint or plain state dict."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = _extract_autoencoder_state(ckpt["state_dict"])
        missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
        if missing:
            warn(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            warn(f"Unexpected keys when loading checkpoint: {unexpected}")
    else:
        autoencoder.load_state_dict(ckpt)
    return autoencoder


def save_audio(wav: torch.Tensor, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate)


@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Simple inference script for the AutoEncoder/VAE.

    It reuses the same initialization logic as training via
    ``build_autoencoder_from_cfg``. The Hydra config ``conf/inference.yaml``
    must provide at least:

    - model: full model configuration (same format as training)
    - train_dataset: kwargs with STFT parameters (n_fft, hop_length, ...)
    - checkpoint: path to a trained model checkpoint
    - input_wav: path to an input waveform file
    - output_wav: path where the reconstructed waveform is written
    """

    # Convert to a plain dict to reuse training initializers
    project_root = Path(get_original_cwd())
    model_cfg = OmegaConf.load(project_root / cfg.model_config_path)
    data_cfg = OmegaConf.load(project_root / cfg.data_config_path)

    # Unifica in un dict compatibile con build_autoencoder_from_cfg
    unified: Dict[str, Any] = OmegaConf.to_container(
        OmegaConf.merge(model_cfg, data_cfg), resolve=True
    )  # contiene chiavi: model, train_dataset, train_dataloader, eval_dataset, ...

    # Se l'utente richiede mono in inferenza ma il dataset di training era stereo
    # dobbiamo patchare le config prima di costruire il modello, altrimenti
    # l'inferenza produce uno spettrogramma con 1 canale e l'encoder attende 2.
    if bool(cfg.get("mono", False)):
        for ds_key in ("train_dataset", "eval_dataset"):
            spec = unified.get(ds_key, None)
            if isinstance(spec, dict):
                kwargs = spec.get("kwargs", {}) or {}
                if kwargs.get("stereo", True) is True:
                    kwargs["stereo"] = False  # forza mono per inferenza
                    spec["kwargs"] = kwargs
                    unified[ds_key] = spec
        # Nota: cac resta invarianti (false) -> tensore complesso C=1
        warn("Patched dataset config to mono (stereo=False) for inference; model channels will be 1.")

    # Build model and infer channels using the shared helper
    autoenc, model_channels, audio_channels = build_autoencoder_from_cfg(unified)
    ok(f"Built AutoEncoder for inference (model_channels={model_channels}, audio_channels={audio_channels})")

    # Load checkpoint
    ckpt_path = Path(get_original_cwd()) / cfg.checkpoint
    if not ckpt_path.is_file():
        err(f"Checkpoint not found: {ckpt_path}")
        return
    autoenc = load_checkpoint(autoenc, ckpt_path)
    autoenc.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc.to(device)

    # Load input waveform
    in_path = Path(get_original_cwd()) / cfg.input_wav
    if not in_path.is_file():
        err(f"Input audio not found: {in_path}")
        return
    wav, sr = torchaudio.load(str(in_path))

    target_sr = int(cfg.get("sample_rate", sr))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.to(device)
    if cfg.get("mono", False) and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.unsqueeze(0)  # [1, C, T]

    # STFT params
    stft_params = (unified.get("train_dataset", {}) or {}).get("kwargs", {}) or {}
    n_fft = int(stft_params.get("n_fft", 2048))
    hop_length = int(stft_params.get("hop_length", n_fft // 4))
    win_length = int(stft_params.get("win_length", n_fft))
    center_flag = bool(stft_params.get("center", True))

    # Safety 2: optional trimming of input waveform to max_seconds or max_frames
    max_seconds = cfg.get("max_seconds", None)
    max_frames  = cfg.get("max_frames", None)
    if max_seconds is not None and max_frames is not None:
        err("Specify only one of max_seconds or max_frames.")
        return
    if max_seconds is not None:
        target_samples = int(round(float(max_seconds) * sr))
    elif max_frames is not None:
        # Approx conversion: raw samples needed to produce max_frames STFT frames
        # F = floor((T + 2*pad - win_length)/hop_length)+1 ; con pad = win_length//2 se center=True
        pad = win_length // 2 if center_flag else 0
        target_samples = (max_frames - 1) * hop_length + win_length - 2 * pad
        if target_samples < 0:
            target_samples = 0
    else:
        target_samples = None

    if target_samples is not None and wav.shape[-1] > target_samples:
        if wav.shape[-1] - target_samples < hop_length:
            warn(f"Trimming input waveform from {wav.shape[-1]} to {target_samples} samples.")
        wav = wav[..., :target_samples]

    # Segment / chunk sizing configuration (user can specify one of seconds OR frames OR latent steps).
    segment_seconds = cfg.get("segment_seconds", None)           # float seconds
    segment_frames = cfg.get("segment_frames", None)             # STFT frames
    chunk_size_latent_cfg = cfg.get("chunk_size", None)          # latent steps (optional)
    overlap_cfg = int(cfg.get("overlap", 32))                    # overlap in latent steps (default)

    if sum(x is not None for x in (segment_seconds, segment_frames, chunk_size_latent_cfg)) > 1:
        err("Specify only one among segment_seconds, segment_frames, chunk_size.")
        return

    # Infer model temporal downsampling (samples per latent step)
    samples_per_latent = autoenc.infer_downsampling_ratio(hop_length=hop_length)
    frames_per_latent = max(1, samples_per_latent // hop_length)  # STFT frame stride implied by encoder

    if segment_seconds is not None:
        segment_samples = int(round(float(segment_seconds) * sr))
        # Convert to latent steps (ceil to cover duration)
        chunk_size_latent = max(1, int(round(segment_samples / samples_per_latent)))
    elif segment_frames is not None:
        # Convert STFT frames to latent steps
        chunk_size_latent = max(1, int(round(segment_frames / frames_per_latent)))
    elif chunk_size_latent_cfg is not None:
        chunk_size_latent = int(chunk_size_latent_cfg)
    else:
        # Fallback default latent length
        chunk_size_latent = 128

    # Translate latent steps + overlap into waveform samples for chunking utility
    chunk_size_samples = chunk_size_latent * samples_per_latent
    overlap_samples = overlap_cfg * samples_per_latent

    chunked = bool(cfg.get("chunked", True))

    # Decide packing of complex: if encoder handles complex values (flag), do not pack
    is_complex_model = bool(getattr(autoenc.encoder, "is_complex", False))
    pack_complex = not is_complex_model
    if pack_complex:
        ok("Using real/imag channel packing for complex spectrograms.")
    else:
        warn("Encoder marked complex-capable; skipping real/imag packing.")

    stft_common_kwargs = dict(n_fft=n_fft, hop_length=hop_length)
    if "win_length" in stft_params:
        stft_common_kwargs["win_length"] = int(stft_params.get("win_length", n_fft))
    if "center" in stft_params:
        stft_common_kwargs["center"] = bool(stft_params.get("center", True))
    if "normalized" in stft_params:
        stft_common_kwargs["normalized"] = bool(stft_params.get("normalized", False))
    if "onesided" in stft_params:
        stft_common_kwargs["onesided"] = bool(stft_params.get("onesided", True))

    with torch.no_grad():
        latents, encode_info = autoenc.encode_audio(
            wav,
            chunked=chunked,
            chunk_size=chunk_size_samples,
            overlap_size=overlap_samples,
            pack_complex=pack_complex,
            debug=True,
            **stft_common_kwargs,
        )
        ok(f"Encoded audio -> latents shape={tuple(latents.shape)}")
        recon = autoenc.decode_audio(
            latents,
            encode_info,
            chunked=chunked,
            pack_complex=pack_complex,
            debug=True,
            **stft_common_kwargs,
        )
        ok(f"Decoded latents -> waveform shape={tuple(recon.shape)}")

    # Remove batch dimension
    recon = recon.squeeze(0)

    out_path = Path(get_original_cwd()) / cfg.output_wav
    save_audio(recon, out_path, sr)
    ok(f"Saved reconstructed audio to {out_path}")


if __name__ == "__main__":
    main()
