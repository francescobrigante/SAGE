from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

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
                cleaned[key[len(prefix) :]] = value
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
    *,
    stereo: bool = True,
    chunked: bool = False,
    chunk_size: int = 0,
    overlap_size: int = 0,
    pack_complex: bool = True,
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the full waveform→latent pipeline used by inference utilities.

    This helper wraps a few moving pieces that live on the ``AutoEncoder``
    itself (STFT, optional pre-transforms, chunk planning) so that deployment
    code can treat the model as a drop-in waveform codec.  The function mirrors
    the behaviour previously implemented as a method on ``AutoEncoder`` but is
    colocated with inference tooling to keep the core model lean.
    """

    autoencoder._require_stft_config()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    elif audio.dim() != 3:
        raise ValueError(f"audio must be 1D/2D/3D, got {tuple(audio.shape)}")

    batch, channels, total_len = audio.shape
    if stereo and channels == 1:
        audio = audio.repeat(1, 2, 1)
        channels = 2
    if not stereo and channels > 2:
        raise ValueError("Expected mono or stereo input when stereo=False")

    info: Dict[str, Any] = {
        "original_length": total_len,
        "chunked": bool(chunked),
        "overlap_size": int(overlap_size),
        "stft_config": autoencoder.stft_config_dict(),
        "pack_complex": bool(pack_complex),
        "stereo": bool(stereo),
    }

    if not chunked:
        chunks: List[Tuple[int, int]] = [(0, total_len)]
        padded_length = total_len
    else:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0 when chunked=True")
        chunks, padded_length = autoencoder._plan_chunks(total_len, chunk_size, overlap_size)
        if padded_length > total_len:
            pad = padded_length - total_len
            audio = F.pad(audio, (0, pad))
    info["padded_length"] = padded_length
    info["chunk_boundaries"] = chunks
    info["chunk_expected_lengths"] = [end - start for start, end in chunks]

    latent_chunks: List[torch.Tensor] = []
    latent_lengths: List[int] = []
    stft_frames: List[int] = []

    for idx, (start, end) in enumerate(chunks):
        wav_chunk = audio[:, :, start:end]
        if debug:
            print(f"[encode_audio] chunk={idx} samples=({start},{end}) waveform_shape={tuple(wav_chunk.shape)}")
        spec = autoencoder.stft(wav_chunk)
        stft_frames.append(spec.shape[-1])
        spec_ready = autoencoder._pack_complex(spec) if pack_complex else spec
        lat = autoencoder.encode(spec_ready)
        latent_chunks.append(lat)
        latent_lengths.append(lat.shape[-1])
        if debug:
            print(f"[encode_audio] chunk={idx} latents_shape={tuple(lat.shape)}")

    if not latent_chunks:
        raise RuntimeError("encode_audio produced no chunks; check input length and chunk configuration")

    latents = torch.cat(latent_chunks, dim=-1) if len(latent_chunks) > 1 else latent_chunks[0]
    info["latent_chunk_lengths"] = latent_lengths
    info["stft_frames_per_chunk"] = stft_frames
    info["latents_shape"] = tuple(latents.shape)

    if debug:
        print(f"[encode_audio] finished: latents_shape={tuple(latents.shape)} metadata={info}")

    return latents, info


def decode_audio(
    autoencoder: AutoEncoder,
    latents: torch.Tensor,
    info: Dict[str, Any],
    *,
    stereo: bool = True,
    chunked: bool = False,
    pack_complex: Optional[bool] = None,
    debug: bool = False,
    remove_padding: bool = True,
) -> torch.Tensor:
    """Reverse :func:`encode_audio`, turning latent tensors back into waveforms.

    The routine consumes the metadata produced by :func:`encode_audio` to
    faithfully undo chunking, overlap-add, STFT packing and any pre-transforms.
    Keeping it alongside inference helpers makes it easy to reproduce training
    reconstructions during validation or offline evaluation.
    """

    if latents.dim() < 3:
        raise ValueError("latents expected to have at least 3 dimensions")

    cfg_in_info = info.get("stft_config")
    if cfg_in_info is not None:
        current = autoencoder.stft_config_dict()
        if current != cfg_in_info:
            autoencoder.set_stft_config(cfg_in_info)

    pack_complex = info.get("pack_complex", True) if pack_complex is None else bool(pack_complex)

    original_length = int(info.get("original_length", 0) or 0)
    padded_length = int(info.get("padded_length", original_length) or original_length)
    overlap_size = int(info.get("overlap_size", 0) or 0)
    chunk_boundaries: List[Tuple[int, int]] = info.get("chunk_boundaries", [(0, original_length)])
    latent_lengths: List[int] = info.get("latent_chunk_lengths", [latents.shape[-1]])

    if not chunked:
        if debug:
            print(f"[decode_audio] single chunk path latents_shape={tuple(latents.shape)}")
        spec = autoencoder.decode(latents, apply_inverse=False)
        spec_linear = autoencoder.apply_inverse_pre_transform(spec)
        spec_complex = autoencoder._unpack_complex(spec_linear) if pack_complex else spec_linear
        target_len = original_length or info.get("chunk_expected_lengths", [None])[0]
        waveform = autoencoder.istft(spec_complex, target_length=target_len)
        if remove_padding and original_length:
            waveform = waveform[..., :original_length]
        return waveform

    if sum(latent_lengths) != latents.shape[-1]:
        raise ValueError("latent_chunk_lengths do not cover the latent time dimension")

    fade_in, fade_out = autoencoder._hann_crossfade_windows(overlap_size)
    if fade_in is not None:
        fade_in = fade_in.to(latents.device)
        fade_out = fade_out.to(latents.device)

    cursor = 0
    chunks: List[torch.Tensor] = []
    for length in latent_lengths:
        chunks.append(latents[..., cursor:cursor + length])
        cursor += length

    decoded_chunks: List[torch.Tensor] = []
    expected_lengths = info.get("chunk_expected_lengths", [None] * len(chunks))

    for idx, lat_chunk in enumerate(chunks):
        if debug:
            print(f"[decode_audio] chunk={idx} latent_shape={tuple(lat_chunk.shape)}")
        spec_chunk = autoencoder.decode(lat_chunk, apply_inverse=False)
        spec_chunk_linear = autoencoder.apply_inverse_pre_transform(spec_chunk)
        spec_complex = autoencoder._unpack_complex(spec_chunk_linear) if pack_complex else spec_chunk_linear
        target_len = expected_lengths[idx] if idx < len(expected_lengths) else None
        wav_chunk = autoencoder.istft(spec_complex, target_length=target_len)
        decoded_chunks.append(wav_chunk)

    if not decoded_chunks:
        raise RuntimeError("decode_audio received empty decoded chunk list")

    if len(chunk_boundaries) != len(decoded_chunks):
        raise ValueError("chunk metadata does not match decoded chunks")

    base_chunk = decoded_chunks[0]
    out = torch.zeros(
        base_chunk.shape[0],
        base_chunk.shape[1],
        padded_length,
        device=base_chunk.device,
        dtype=base_chunk.dtype,
    )

    for idx, ((start, end), wav_chunk) in enumerate(zip(chunk_boundaries, decoded_chunks)):
        expected_len = end - start
        if wav_chunk.shape[-1] != expected_len:
            diff = expected_len - wav_chunk.shape[-1]
            if diff > 0:
                wav_chunk = F.pad(wav_chunk, (0, diff))
            else:
                wav_chunk = wav_chunk[..., :expected_len]
        if overlap_size > 0 and fade_in is not None:
            if idx > 0:
                wav_chunk[..., :overlap_size] *= fade_in.view(1, 1, -1)
            if idx < len(decoded_chunks) - 1:
                wav_chunk[..., -overlap_size:] *= fade_out.view(1, 1, -1)
        out[..., start:end] += wav_chunk

    if remove_padding and original_length:
        out = out[..., :original_length]

    if debug:
        print(f"[decode_audio] reconstructed waveform shape={tuple(out.shape)}")

    return out


class EuleroEncodeDecode:
    """High-level checkpoint loader that exposes encode/decode conveniences.

    The class encapsulates three responsibilities often needed in deployment:
    loading the Lightning checkpoint, rebuilding the architecture from the
    exported configuration metadata, and providing audio-facing helpers that
    wrap the low-level spectrogram pipelines.  By centralising those steps we
    keep inference scripts compact and prevent duplication across validation
    and production code paths.
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
        self.strict = bool(strict)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        inference_cfg = ckpt.get("inference_config")
        if not inference_cfg:
            raise KeyError("Checkpoint does not contain an 'inference_config' section.\n"
                           "Regenerate the checkpoint with the updated training pipeline.")

        model_cfg = inference_cfg.get("model")
        if not isinstance(model_cfg, dict):
            raise ValueError("inference_config.model must be a dictionary with encoder/decoder specs.")

        self.autoencoder = AutoEncoder.from_config(model_cfg)

        stft_cfg = inference_cfg.get("stft_config")
        if stft_cfg is not None:
            self.autoencoder.set_stft_config(stft_cfg)

        state_dict = ckpt.get("state_dict", ckpt)
        cleaned_state = _extract_autoencoder_state(state_dict)
        missing, unexpected = self.autoencoder.load_state_dict(cleaned_state, strict=self.strict)
        if missing:
            warnings.warn(f"Missing keys while loading checkpoint: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys while loading checkpoint: {unexpected}")
        ok( f"Loaded checkpoint '{self.checkpoint_path}' into EuleroEncodeDecode model." )

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        self.metadata: Dict[str, Any] = dict(inference_cfg)
        self.sample_rate: Optional[int] = (
            int(inference_cfg.get("sample_rate")) if inference_cfg.get("sample_rate") is not None else None
        )
        self.audio_channels: Optional[int] = (
            int(inference_cfg.get("audio_channels")) if inference_cfg.get("audio_channels") is not None else None
        )
        self.model_channels: Optional[int] = (
            int(inference_cfg.get("in_channels")) if inference_cfg.get("in_channels") is not None else None
        )
        self._stft_config: Optional[Dict[str, Any]] = stft_cfg if isinstance(stft_cfg, dict) else None

        hop_length = self._stft_config.get("hop_length") if self._stft_config else None
        self.samples_per_latent: Optional[int] = None
        self.frames_per_latent: Optional[int] = None
        if hop_length is not None:
            try:
                samples_per_latent = self.autoencoder.infer_downsampling_ratio(int(hop_length))
                self.samples_per_latent = int(samples_per_latent)
                if hop_length > 0:
                    self.frames_per_latent = max(1, self.samples_per_latent // int(hop_length))
            except Exception as exc:
                warnings.warn(f"Unable to infer downsampling ratio: {exc}")

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

    def encode_audio(
        self,
        audio: torch.Tensor,
        *,
        stereo: bool = True,
        chunked: bool = False,
        chunk_size: int = 0,
        overlap_size: int = 0,
        pack_complex: bool = True,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.autoencoder.eval()
        with torch.no_grad():
            return encode_audio(
                self.autoencoder,
                audio.to(self.device),
                stereo=stereo,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                pack_complex=pack_complex,
                debug=debug,
            )

    def decode_audio(
        self,
        latents: torch.Tensor,
        info: Dict[str, Any],
        *,
        stereo: bool = True,
        chunked: bool = False,
        pack_complex: Optional[bool] = None,
        debug: bool = False,
        remove_padding: bool = True,
    ) -> torch.Tensor:
        self.autoencoder.eval()
        with torch.no_grad():
            return decode_audio(
                self.autoencoder,
                latents.to(self.device),
                info,
                stereo=stereo,
                chunked=chunked,
                pack_complex=pack_complex,
                debug=debug,
                remove_padding=remove_padding,
            )

    def __repr__(self) -> str:
        model_name = self.metadata.get("model", {}).get("encoder", {}).get("class", "unknown")
        return f"EuleroEncodeDecode(model={model_name}, device={self.device})"
