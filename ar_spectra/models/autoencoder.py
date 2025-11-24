from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ar_spectra.models.bottlenecks import SkipBottleneck, VAEBottleneck
from ar_spectra.training_utils.pre_transform import create_pre_transform
from rich.console import Console

console = Console()
def warn(msg):   console.print(msg, style="bold yellow")

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def _locate_class(class_path: Union[str, type]) -> type:
    """Supports either a string path 'pkg.mod.Class' or a class already passed."""
    if not isinstance(class_path, str):
        return class_path
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def instantiate_from_spec(spec: Dict[str, Any]) -> Any:
    """Instantiate an object from a simple spec.

    This local helper is used only inside this module to
    build encoder/decoder/bottleneck components. For the
    global training/inference initialization logic, use
    ``ar_spectra.training_utils.initialization.instantiate_from_spec``.
    """

    if "class" not in spec:
        raise ValueError("spec is missing the 'class' key.")
    module_path, class_name = spec["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    args = spec.get("args", []) or []
    kwargs = spec.get("kwargs", {}) or {}
    return cls(*args, **kwargs)


@dataclass
class STFTConfig:
    """Canonical STFT configuration shared across encode/decode helpers."""

    n_fft: int
    hop_length: int
    win_length: int
    center: bool = True
    normalized: bool = False
    onesided: bool = True
    window: Optional[Union[str, torch.Tensor]] = None
    _window_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be a positive integer")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be a positive integer")
        if self.win_length <= 0:
            raise ValueError("win_length must be a positive integer")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if not self.center:
            raise ValueError("center must be set to True to guarantee waveform length consistency")

    def to_public_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "n_fft": int(self.n_fft),
            "hop_length": int(self.hop_length),
            "win_length": int(self.win_length),
            "center": bool(self.center),
            "normalized": bool(self.normalized),
            "onesided": bool(self.onesided),
        }
        if isinstance(self.window, str):
            data["window"] = self.window
        return data

    def get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key in self._window_cache:
            return self._window_cache[key]

        if isinstance(self.window, torch.Tensor):
            win = self.window.to(device=device, dtype=dtype)
        elif self.window is None:
            win = torch.hann_window(self.win_length, dtype=dtype, device=device)
        elif isinstance(self.window, str):
            factory_name = f"{self.window.lower()}_window"
            factory = getattr(torch, factory_name, None)
            if factory is None:
                raise ValueError(f"Unsupported window factory '{self.window}'.")
            win = factory(self.win_length, dtype=dtype, device=device)
        else:
            raise TypeError("window must be None, Tensor or string identifier")

        self._window_cache[key] = win
        return win


class AutoEncoder(nn.Module):
    """
    Generic container. Manages forward pass encoder->decoder.
    If return_latent=True, forward returns (reconstruction, latent).
    """
    def __init__(
        self,
        encoder: Union[nn.Module, Dict[str, Any], str, type],
        decoder: Union[nn.Module, Dict[str, Any], str, type],
        bottleneck: Optional[Union[nn.Module, Dict[str, Any], str, type]] = None,
        return_latent: bool = False,
        pre_transform: Optional[Union[str, Dict[str, Any]]] = None,
        stft_config: Optional[Union[STFTConfig, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        # Allow passing either direct instances or specs / class names
        self.encoder = (
            instantiate_from_spec(encoder) if isinstance(encoder, dict) else
            _locate_class(encoder)() if isinstance(encoder, (str, type)) else
            encoder
        )
        self.decoder = (
            instantiate_from_spec(decoder) if isinstance(decoder, dict) else
            _locate_class(decoder)() if isinstance(decoder, (str, type)) else
            decoder
        )
        self.return_latent = return_latent
        self.bottleneck = (
            instantiate_from_spec(bottleneck) if isinstance(bottleneck, dict) else
            (_locate_class(bottleneck)() if isinstance(bottleneck, (str, type)) else bottleneck)
            if bottleneck is not None else None
        )
        # Optional spectrogram normalization applied at encoder input and
        # inverted after decoder output (before losses / ISTFT).
        try:
            self.pre_transform = create_pre_transform(pre_transform)
        except Exception:
            self.pre_transform = None

        self._stft_config: Optional[STFTConfig] = None
        if stft_config is not None:
            self.set_stft_config(stft_config)
        
        # Consistency checks encoder/decoder vs bottleneck 
        def _get_enc_dim(m):
            if hasattr(m, "output_size"):
                try: return int(m.output_size())
                except Exception: pass
            for k in ["dimension", "latent_dim", "out_channels"]:
                if hasattr(m, k):
                    try: return int(getattr(m, k))
                    except Exception: pass
            return None

        def _get_dec_in_dim(m):
            for k in ["input_size", "dimension", "in_channels"]:
                if hasattr(m, k):
                    try: return int(getattr(m, k))
                    except Exception: pass
            return None

        enc_dim = _get_enc_dim(self.encoder)
        dec_in  = _get_dec_in_dim(self.decoder)

        if isinstance(self.bottleneck, VAEBottleneck):
            if (enc_dim is not None) and (dec_in is not None):
                assert enc_dim == 2 * dec_in, (
                    f"Config mismatch with VAEBottleneck: encoder channels={enc_dim} "
                    f"must be 2× decoder input={dec_in}. "
                    f"Hint: set encoder.dimension=2*C and decoder.input_size=C."
                )
            else:
                warnings.warn("VAEBottleneck active but unable to deduce enc_dim/dec_in for check. Ensure encoder.dimension=2*C and decoder.input_size=C when using VAE bottleneck.")
        elif isinstance(self.bottleneck, SkipBottleneck):
            if (enc_dim is not None) and (dec_in is not None):
                assert enc_dim == dec_in or dec_in % enc_dim == 0, (
                    f"Config mismatch with SkipBottleneck: encoder channels={enc_dim} "
                    f"must be either {dec_in} or n*{dec_in}. "
                    f"Hint: set encoder.dimension=C and decoder.input_size=C or encoder.dimension=n*C and decoder.input_size=C."
                )
        else:
            if (enc_dim is not None) and (dec_in is not None):
                assert enc_dim == dec_in, (
                    f"Encoder/Decoder channel mismatch without bottleneck: "
                    f"{enc_dim} vs {dec_in}"
                )
            elif self.bottleneck is None:
                warnings.warn("No bottleneck selected: unable to verify encoder/decoder because dimensions are not deducible.")
    
    def _pack_complex(self, S: torch.Tensor) -> torch.Tensor:
        """
        Convert complex spectrogram into real/imag stacking along channel axis.
        Input:
            complex: [B, C, F, T] -> [B, 2C, F, T]
            real: returned unchanged.
        """
        if torch.is_complex(S):
            if S.dim() == 4:  # [B, C, F, T]
                return torch.cat([S.real, S.imag], dim=1)
            else:
                raise ValueError(f"_pack_complex: unsupported complex shape {tuple(S.shape)}")
        return S  # already real

    def _unpack_complex(self, S: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct complex tensor from real/imag stacking.
        Input:
            [B, 2C, F, T] -> complex [B, C, F, T]
        If tensor is already complex it is returned unchanged.
        """
        if torch.is_complex(S):
            return S
        if S.dim() != 4:
            raise ValueError(f"_unpack_complex: expected 4D (B, 2C, F, T), got {tuple(S.shape)}")
        B, Cx, F, T = S.shape
        if Cx % 2 != 0:
            raise ValueError(f"_unpack_complex: channel count {Cx} is not even (not real/imag).")
        C = Cx // 2
        real = S[:, :C]
        imag = S[:, C:]
        return torch.complex(real, imag)

    def infer_downsampling_ratio(self, hop_length: int) -> int:
        """
        Deduce total temporal downsampling factor of the encoder (frames -> latents)
        and compute samples_per_latent = hop_length * total_time_stride.
        Saves self.downsampling_ratio and returns it.
        Requires the encoder to expose an attribute 'ratios' (list of pairs [k, s] or dicts).
        """
        ratios = getattr(self.encoder, "ratios", None)
        if ratios is None:
            # Simple fallback
            self.downsampling_ratio = hop_length
            return self.downsampling_ratio
        strides = []
        for r in ratios:
            # Supports formats: [kernel, stride] or dict {"stride": s} / {"time": s}
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                strides.append(int(r[1]))
            elif isinstance(r, dict):
                val = r.get("stride", r.get("time", None))
                if val is not None:
                    strides.append(int(val))
        total_time_stride = 1
        for s in strides:
            total_time_stride *= s
        samples_per_latent = hop_length * total_time_stride
        self.downsampling_ratio = samples_per_latent
        return self.downsampling_ratio

    def set_stft_config(self, config: Union[STFTConfig, Dict[str, Any]]) -> None:
        """Store a normalized STFT configuration used by audio helpers.

        The configuration is typically sourced from dataset YAML files.
        Only a single canonical configuration is kept to avoid diverging
        parameters across encode/decode utilities.
        """

        if isinstance(config, STFTConfig):
            new_cfg = config
        elif isinstance(config, dict):
            cfg = dict(config)
            missing = [k for k in ("n_fft", "hop_length") if k not in cfg]
            if missing:
                raise ValueError(f"Missing STFT keys: {missing}")
            n_fft = int(cfg["n_fft"])
            hop = int(cfg["hop_length"])
            win_length = int(cfg.get("win_length", n_fft))
            center = bool(cfg.get("center", True))
            normalized = bool(cfg.get("normalized", False))
            onesided = bool(cfg.get("onesided", True))
            window = cfg.get("window", None)
            if isinstance(window, torch.Tensor):
                window = window.detach()
            new_cfg = STFTConfig(
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                center=center,
                normalized=normalized,
                onesided=onesided,
                window=window,
            )
        else:
            raise TypeError("config must be a dict or STFTConfig instance")

        self._stft_config = new_cfg

    def stft_config_dict(self) -> Optional[Dict[str, Any]]:
        return None if self._stft_config is None else self._stft_config.to_public_dict()

    def _require_stft_config(self) -> STFTConfig:
        if self._stft_config is None:
            raise RuntimeError(
                "STFT configuration is not set. Call set_stft_config() with the dataset parameters before using audio helpers."
            )
        return self._stft_config

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        return_info: bool = False,
        debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Run encoder (and optional bottleneck) on a spectrogram batch."""

        x = inputs
        if self.pre_transform is not None:
            try:
                x = self.pre_transform.transform(x)
            except Exception as exc:
                warn(f"pre_transform.transform failed ({type(exc).__name__}: {exc}); continuing without it.")

        if debug:
            print(f"[AutoEncoder.encode] encoder input shape={tuple(x.shape)}")

        latents = self.encoder(x)

        if debug:
            print(f"[AutoEncoder.encode] encoder output shape={tuple(latents.shape)}")

        info: Dict[str, Any] = {"pre_bottleneck_latents": latents}

        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True)
            info.update(bottleneck_info)

        if debug and self.bottleneck is not None:
            print(f"[AutoEncoder.encode] bottleneck output shape={tuple(latents.shape)}")

        if return_info:
            return latents, info
        return latents

    def decode(self, latents: torch.Tensor, *, debug: bool = False) -> torch.Tensor:
        """Run optional bottleneck decode followed by the decoder module."""

        z = latents
        if self.bottleneck is not None:
            z = self.bottleneck.decode(z)

        decoded = self.decoder(z)

        if self.pre_transform is not None:
            try:
                decoded = self.pre_transform.inverse(decoded)
            except Exception as exc:
                warn(f"pre_transform.inverse failed ({type(exc).__name__}: {exc}); returning raw decoder output.")

        if debug:
            print(f"[AutoEncoder.decode] output shape={tuple(decoded.shape)}")

        return decoded
          
    def _maybe_add_nyquist(self, S: torch.Tensor, n_fft: int, onesided: bool) -> torch.Tensor:
        F = S.shape[-2]
        if n_fft is not None and onesided and F == n_fft // 2:
            pad_shape = list(S.shape); pad_shape[-2] = 1
            nyq = torch.zeros(pad_shape, dtype=S.dtype, device=S.device)
            return torch.cat([S, nyq], dim=-2)
        return S
    
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute the STFT using the canonical configuration."""

        if torch.is_complex(audio):
            raise ValueError("stft expects real-valued waveforms")

        cfg = self._require_stft_config()

        if audio.dim() != 3:
            raise ValueError(f"Expected waveform shape [B, C, T], got {tuple(audio.shape)}")

        B, C, T = audio.shape
        window = cfg.get_window(audio.device, audio.dtype)

        audio_bc = audio.reshape(B * C, T)
        spec_bc = torch.stft(
            audio_bc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
            normalized=cfg.normalized,
            onesided=cfg.onesided,
            return_complex=True,
        )

        F_bins, frames = spec_bc.shape[-2], spec_bc.shape[-1]
        spec = spec_bc.reshape(B, C, F_bins, frames)
        return spec
        
    def istft(self, spec: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """Invert the STFT using the canonical configuration."""

        cfg = self._require_stft_config()

        if torch.is_complex(spec):
            S = spec
        elif spec.dim() == 4:
            S = self._unpack_complex(spec)
        elif spec.dim() == 3:
            S = self._unpack_complex(spec.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f"Unsupported spectrogram shape {tuple(spec.shape)}")

        window_dtype = torch.float32 if S.dtype == torch.complex64 else torch.float64
        window = cfg.get_window(S.device, window_dtype)

        S = self._maybe_add_nyquist(S, cfg.n_fft, cfg.onesided)

        frames = S.shape[-1]
        if target_length is None:
            target_length = max(cfg.hop_length * max(frames - 1, 0), 0)

        if S.dim() == 4:
            B, C, F_bins, T_frames = S.shape
            Sbc = S.reshape(B * C, F_bins, T_frames).contiguous()
            wav_bc = torch.istft(
                Sbc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                window=window,
                center=True,
                normalized=cfg.normalized,
                onesided=cfg.onesided,
                return_complex=False,
                length=target_length,
            )
            return wav_bc.reshape(B, C, -1)

        if S.dim() == 3:
            return torch.istft(
                S,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                window=window,
                center=True,
                normalized=cfg.normalized,
                onesided=cfg.onesided,
                return_complex=False,
                length=target_length,
            )

        raise RuntimeError(f"Unexpected complex shape {S.shape}")


    # =============================
    # Inference utility functions
    # =============================
    def _plan_chunks(self, total_len: int, chunk_size: int, overlap_size: int) -> Tuple[List[Tuple[int, int]], int]:
        """Plan (start,end) sample indices for chunked processing.
        Pads at end so last chunk has exact chunk_size.
        Returns list of tuples and the padded total length.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap_size < 0 or overlap_size >= chunk_size:
            raise ValueError("overlap_size must satisfy 0 <= overlap_size < chunk_size")
        if total_len <= 0:
            return [], 0
        step = chunk_size - overlap_size
        if step <= 0:
            raise ValueError("chunk_size must be greater than overlap_size")
        import math
        n_chunks = math.ceil((total_len - overlap_size) / step)
        padded_len = (n_chunks - 1) * step + chunk_size
        chunks: List[Tuple[int,int]] = []
        for i in range(n_chunks):
            s = i * step
            e = s + chunk_size
            chunks.append((s, e))
        return chunks, padded_len

    def _hann_crossfade_windows(self, overlap_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (fade_in, fade_out) Hann halves; None if overlap_size==0."""
        if overlap_size == 0:
            return None, None
        w = torch.hann_window(2 * overlap_size, periodic=False)
        return w[:overlap_size], w[overlap_size:]

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
        """Encode raw waveforms into latents using the canonical STFT setup."""

        _ = self._require_stft_config()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        elif audio.dim() != 3:
            raise ValueError(f"audio must be 1D/2D/3D, got {tuple(audio.shape)}")

        B, C, total_len = audio.shape
        if stereo and C == 1:
            audio = audio.repeat(1, 2, 1)
            C = 2
        if not stereo and C > 2:
            raise ValueError("Expected mono or stereo input when stereo=False")

        info: Dict[str, Any] = {
            "original_length": total_len,
            "chunked": bool(chunked),
            "overlap_size": int(overlap_size),
            "stft_config": self.stft_config_dict(),
            "pack_complex": bool(pack_complex),
            "stereo": bool(stereo),
        }

        if not chunked:
            chunks: List[Tuple[int, int]] = [(0, total_len)]
            padded_length = total_len
        else:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be > 0 when chunked=True")
            chunks, padded_length = self._plan_chunks(total_len, chunk_size, overlap_size)
            if padded_length > total_len:
                pad = padded_length - total_len
                audio = torch.nn.functional.pad(audio, (0, pad))
        info["padded_length"] = padded_length
        info["chunk_boundaries"] = chunks
        info["chunk_expected_lengths"] = [e - s for s, e in chunks]

        latent_chunks: List[torch.Tensor] = []
        latent_lengths: List[int] = []
        stft_frames: List[int] = []

        for idx, (start, end) in enumerate(chunks):
            wav_chunk = audio[:, :, start:end]
            if debug:
                print(f"[encode_audio] chunk={idx} samples=({start},{end}) waveform_shape={tuple(wav_chunk.shape)}")
            spec = self.stft(wav_chunk)
            stft_frames.append(spec.shape[-1])
            spec_ready = self._pack_complex(spec) if pack_complex else spec
            lat = self.encode(spec_ready)
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
        """Invert the encoding pipeline returning waveform batches."""

        if latents.dim() < 3:
            raise ValueError("latents expected to have at least 3 dimensions")

        cfg_in_info = info.get("stft_config")
        if cfg_in_info is not None:
            current = self.stft_config_dict()
            if current != cfg_in_info:
                self.set_stft_config(cfg_in_info)

        pack_complex = info.get("pack_complex", True) if pack_complex is None else bool(pack_complex)

        original_length = int(info.get("original_length", 0) or 0)
        padded_length = int(info.get("padded_length", original_length) or original_length)
        overlap_size = int(info.get("overlap_size", 0) or 0)
        chunk_boundaries: List[Tuple[int, int]] = info.get("chunk_boundaries", [(0, original_length)])
        latent_lengths: List[int] = info.get("latent_chunk_lengths", [latents.shape[-1]])

        if not chunked:
            if debug:
                print(f"[decode_audio] single chunk path latents_shape={tuple(latents.shape)}")
            spec = self.decode(latents)
            spec_complex = self._unpack_complex(spec) if pack_complex else spec
            target_len = original_length or info.get("chunk_expected_lengths", [None])[0]
            waveform = self.istft(spec_complex, target_length=target_len)
            if remove_padding and original_length:
                waveform = waveform[..., :original_length]
            return waveform

        if sum(latent_lengths) != latents.shape[-1]:
            raise ValueError("latent_chunk_lengths do not cover the latent time dimension")

        fade_in, fade_out = self._hann_crossfade_windows(overlap_size)
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
            spec_chunk = self.decode(lat_chunk)
            spec_complex = self._unpack_complex(spec_chunk) if pack_complex else spec_chunk
            target_len = expected_lengths[idx] if idx < len(expected_lengths) else None
            wav_chunk = self.istft(spec_complex, target_length=target_len)
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
                    wav_chunk = torch.nn.functional.pad(wav_chunk, (0, diff))
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

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AutoEncoder":
        ae_kwargs = cfg.get("autoencoder", {}) or {}
        encoder_spec = cfg["encoder"]
        decoder_spec = cfg["decoder"]
        bottleneck_spec = cfg.get("bottleneck", None)

        allowed_ae_keys = {"return_latent", "pre_transform"}
        unknown = set(ae_kwargs.keys()) - allowed_ae_keys
        if unknown:
            warn(f"AutoEncoder.from_config: ignoring unsupported keys in autoencoder: {sorted(list(unknown))}")
        ae_kwargs = {k: v for k, v in ae_kwargs.items() if k in allowed_ae_keys}

        skip_flag = False
        target_channels: Optional[int] = None
        if isinstance(bottleneck_spec, dict):
            bn_kwargs = bottleneck_spec.get("kwargs", {}) or {}
            skip_flag = bool(
                bottleneck_spec.get("skip_bottleneck", False)
                or bottleneck_spec.get("skip", False)
                or bn_kwargs.get("skip_bottleneck", False)
                or bn_kwargs.get("skip", False)
            )
        if isinstance(decoder_spec, dict):
            dkw = decoder_spec.get("kwargs", {}) or {}
            for k in ("input_size", "dimension", "in_channels"):
                if k in dkw and isinstance(dkw[k], (int, float)):
                    target_channels = int(dkw[k])
                    break

        if skip_flag:
            warn("Warning: bottleneck skipped; disable skip_bottleneck to revert.")
            bottleneck_inst = SkipBottleneck(target_channels=target_channels)
            return cls(encoder_spec, decoder_spec, bottleneck=bottleneck_inst, **ae_kwargs)

        return cls(encoder_spec, decoder_spec, bottleneck=bottleneck_spec, **ae_kwargs)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_from_json(path: str) -> AutoEncoder:
    return AutoEncoder.from_config(load_config(path))
