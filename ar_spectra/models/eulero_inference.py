from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

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
    """Strip trainer-specific prefixes so checkpoints load into a bare ``AutoEncoder``.

    Lightning checkpoints store model weights under entries such as
    ``"engine.autoencoder.encoder.0.weight"``.  Inference code only owns the
    raw :class:`AutoEncoder`, so we project those keys back to the namespace the
    model expects.  If we fail to find any prefixed keys we fall back to the
    original dictionary, which covers plain ``state_dict`` exports.
    """
    prefixes = ("engine.autoencoder.", "autoencoder.")
    cleaned: Dict[str, Any] = {}

    for key, value in state_dict.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                cleaned[key[len(prefix):]] = value
                matched = True
                break
        if not matched and (
            key.startswith("encoder.")
            or key.startswith("decoder.")
            or key.startswith("bottleneck.")
        ):
            cleaned[key] = value

    return cleaned if cleaned else state_dict


def encode_audio(
    autoencoder: AutoEncoder,
    audio: torch.Tensor,
    debug: bool = False,
) -> torch.Tensor:
    """Encode waveform to latents.
    
    Parameters
    ----------
    autoencoder : AutoEncoder
        Model with STFT config set.
    audio : torch.Tensor
        Waveform tensor of shape (T,), (C, T), or (B, C, T).
    debug : bool
        Print debug info.
        
    Returns
    -------
    torch.Tensor
        Latent tensor.
    """
    autoencoder._require_stft_config()

    # Normalize input shape to (B, C, T)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    elif audio.dim() != 3:
        raise ValueError(f"audio must be 1D/2D/3D, got {tuple(audio.shape)}")

    if debug:
        print(f"[encode_audio] input waveform shape: {tuple(audio.shape)}")

    # STFT
    spec = autoencoder.stft(audio)
    if debug:
        print(f"[encode_audio] STFT output shape: {tuple(spec.shape)}")

    # Pack complex if model expects real input (check encoder's is_complex attribute)
    is_complex = getattr(autoencoder.encoder, 'is_complex', False)
    if not is_complex:
        spec = autoencoder._pack_complex(spec)
        if debug:
            print(f"[encode_audio] packed complex shape: {tuple(spec.shape)}")

    # Encode
    latents = autoencoder.encode(spec)
    if debug:
        print(f"[encode_audio] latents shape: {tuple(latents.shape)}")

    return latents


def decode_audio(
    autoencoder: AutoEncoder,
    latents: torch.Tensor,
    target_length: Optional[int] = None,
    debug: bool = False,
) -> torch.Tensor:
    """Decode latents to waveform.
    
    Parameters
    ----------
    autoencoder : AutoEncoder
        Model with STFT config set.
    latents : torch.Tensor
        Latent tensor from encode_audio.
    target_length : int, optional
        Desired output length in samples. If provided, trims the spectrogram
        to the expected number of STFT frames before iSTFT. This fixes the
        frame mismatch issue where the decoder produces extra frames.
    debug : bool
        Print debug info.
        
    Returns
    -------
    torch.Tensor
        Reconstructed waveform of shape (B, C, T).
    """
    autoencoder._require_stft_config()

    if latents.dim() < 3:
        raise ValueError(f"latents must have at least 3 dimensions, got {latents.dim()}")

    if debug:
        print(f"[decode_audio] latents shape: {tuple(latents.shape)}")

    # Decode latents to spectrogram (in transformed domain)
    spec = autoencoder.decode(latents, apply_inverse=False)
    if debug:
        print(f"[decode_audio] decoded spec shape: {tuple(spec.shape)}")

    # Apply inverse pre-transform to get linear spectrogram
    spec_linear = autoencoder.apply_inverse_pre_transform(spec)

    # Unpack complex if model outputs real (packed) format
    is_complex = getattr(autoencoder.encoder, 'is_complex', False)
    if not is_complex:
        spec_complex = autoencoder._unpack_complex(spec_linear)
        if debug:
            print(f"[decode_audio] unpacked complex shape: {tuple(spec_complex.shape)}")
    else:
        spec_complex = spec_linear

    # Trim spectrogram to expected STFT frames if target_length provided
    # This fixes the frame mismatch from non-integer downsampling ratios
    if target_length is not None:
        hop_length = autoencoder._stft_config.hop_length
        expected_frames = 1 + target_length // hop_length
        actual_frames = spec_complex.shape[-1]
        
        if debug:
            print(f"[decode_audio] target_length={target_length}, expected_frames={expected_frames}, actual_frames={actual_frames}")
        
        if actual_frames > expected_frames:
            spec_complex = spec_complex[..., :expected_frames]
            if debug:
                print(f"[decode_audio] trimmed to {expected_frames} frames")

    # iSTFT
    waveform = autoencoder.istft(spec_complex, target_length=target_length)
    if debug:
        print(f"[decode_audio] output waveform shape: {tuple(waveform.shape)}")

    return waveform


class EuleroEncodeDecode:
    """High-level checkpoint loader for inference.

    Loads a Lightning checkpoint, rebuilds the model architecture from the
    stored configuration, and provides simple encode/decode methods.
    
    Example
    -------
    >>> model = EuleroEncodeDecode("checkpoint.ckpt")
    >>> latents = model.encode(audio)
    >>> reconstructed = model.decode(latents, target_length=audio.shape[-1])
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        inference_cfg = ckpt.get("inference_config")
        if not inference_cfg:
            raise KeyError(
                "Checkpoint does not contain 'inference_config'.\n"
                "Regenerate the checkpoint with the updated training pipeline."
            )

        model_cfg = inference_cfg.get("model")
        if not isinstance(model_cfg, dict):
            raise ValueError("inference_config.model must be a dictionary.")

        self.autoencoder = AutoEncoder.from_config(model_cfg)

        stft_cfg = inference_cfg.get("stft_config")
        if stft_cfg is not None:
            self.autoencoder.set_stft_config(stft_cfg)

        state_dict = ckpt.get("state_dict", ckpt)
        cleaned_state = _extract_autoencoder_state(state_dict)
        missing, unexpected = self.autoencoder.load_state_dict(cleaned_state, strict=strict)
        if missing:
            warnings.warn(f"Missing keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        # Store useful metadata
        self.sample_rate: Optional[int] = inference_cfg.get("sample_rate")
        self.audio_channels: Optional[int] = inference_cfg.get("audio_channels")
        self._stft_config = stft_cfg
        
        ok(f"Loaded checkpoint: {self.checkpoint_path}")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def to(self, device: Union[str, torch.device]) -> "EuleroEncodeDecode":
        self.device = self._resolve_device(device)
        self.autoencoder.to(self.device)
        return self

    @property
    def model(self) -> AutoEncoder:
        return self.autoencoder

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Encode waveform to latents."""
        self.autoencoder.eval()
        return encode_audio(self.autoencoder, audio.to(self.device), debug=debug)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        target_length: Optional[int] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """Decode latents to waveform."""
        self.autoencoder.eval()
        return decode_audio(
            self.autoencoder,
            latents.to(self.device),
            target_length=target_length,
            debug=debug,
        )

    def __repr__(self) -> str:
        return f"EuleroEncodeDecode(device={self.device}, sr={self.sample_rate})"
