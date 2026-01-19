from __future__ import annotations

import importlib
import json
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate

from ar_spectra.models.bottlenecks import SkipBottleneck, VAEBottleneck
from ar_spectra.training_utils.pre_transform import resolve_pre_transform
from rich.console import Console
console = Console()

def ok(msg):     console.print(msg, style="bold green")
def warn(msg):   console.print(msg, style="bold yellow")
def err(msg):    console.print(msg, style="bold red")
def info(msg):   console.print(msg, style="cyan")


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


def _instantiate_component(spec: Dict[str, Any]) -> Any:
    """Instantiate a component from a Hydra-style config dict.
    
    Supports both new Hydra format (_target_) and converts legacy format (class/kwargs)
    for backwards compatibility with existing checkpoints.
    """
    if spec is None:
        return None
    
    # If already has _target_, use hydra directly
    if "_target_" in spec:
        return hydra_instantiate(spec, _convert_="all")
    
    # Legacy format conversion: class/kwargs -> _target_
    if "class" in spec:
        converted = {"_target_": spec["class"]}
        kwargs = spec.get("kwargs", {}) or {}
        converted.update(kwargs)
        return hydra_instantiate(converted, _convert_="all")
    
    raise ValueError("spec must contain either '_target_' (Hydra) or 'class' (legacy) key.")


def _class_path(obj: Union[str, type, nn.Module]) -> str:
    if isinstance(obj, str):
        return obj
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonicalize_module_spec(module_like: Union[nn.Module, Dict[str, Any], str, type, None]) -> Optional[Dict[str, Any]]:
    """Convert encoder/decoder references into exportable configuration dicts.

    Training code may build the autoencoder by passing instantiated modules,
    fully-qualified class names, or already-normalised spec dictionaries.  When
    we later embed the architecture inside checkpoints we need a consistent
    representation so inference can rebuild the same modules.  This helper
    performs that normalisation while also preserving lightweight metadata such
    as ``target_channels`` when it is exposed by the module instance.
    """
    if module_like is None:
        return None
    if isinstance(module_like, dict):
        return deepcopy(module_like)

    class_path = _class_path(module_like)
    spec: Dict[str, Any] = {"class": class_path}
    if isinstance(module_like, nn.Module):
        extra_kwargs: Dict[str, Any] = {}
        if hasattr(module_like, "target_channels"):
            extra_kwargs["target_channels"] = getattr(module_like, "target_channels")
        if extra_kwargs:
            spec["kwargs"] = extra_kwargs
    return spec


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
    """Generic container wiring encoder, optional bottleneck, and decoder.

    This class standardises how autoencoders are built and invoked across the
    repository. Encoders are expected to return ``(latents, encoder_info)``
    where ``encoder_info`` may expose spatial hints like ``feature_shape``.
    Bottlenecks (when present) may add a third info dict. Decoders consume the
    latents (and optionally the propagated feature_shape) to produce a
    spectrogram reconstruction. Pre/post transforms are applied automatically
    when configured.
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
        self._encoder_export_spec = _canonicalize_module_spec(encoder)
        self._decoder_export_spec = _canonicalize_module_spec(decoder)
        self._bottleneck_export_spec = _canonicalize_module_spec(bottleneck)
        self._autoencoder_export_config: Dict[str, Any] = {"return_latent": bool(return_latent)}
        self._pre_transform_spec: Optional[Union[str, Dict[str, Any]]] = None
        self._stft_config_source: Optional[Dict[str, Any]] = None
        # Allow passing either direct instances or specs / class names
        self.encoder = (
            _instantiate_component(encoder) if isinstance(encoder, dict) else
            _locate_class(encoder)() if isinstance(encoder, (str, type)) else
            encoder
        )
        self.decoder = (
            _instantiate_component(decoder) if isinstance(decoder, dict) else
            _locate_class(decoder)() if isinstance(decoder, (str, type)) else
            decoder
        )
        self.return_latent = return_latent
        self.bottleneck = (
            _instantiate_component(bottleneck) if isinstance(bottleneck, dict) else
            (_locate_class(bottleneck)() if isinstance(bottleneck, (str, type)) else bottleneck)
            if bottleneck is not None else None
        )
        # Optional spectrogram normalization applied at encoder input and
        # optionally inverted after decoder output.
        self.pre_transform: Optional[Any] = None
        self._pre_transform_apply_encoder: bool = True
        self._pre_transform_apply_inverse: bool = True
        self._pre_transform_apply_target: bool = False
        try:
            self.configure_pre_transform(pre_transform)
            info(self.pre_transform_description())
        except Exception:
            self.pre_transform = None
            self._pre_transform_apply_encoder = True
            self._pre_transform_apply_inverse = True
            self._pre_transform_apply_target = False
            self._pre_transform_spec = None
            self._autoencoder_export_config.pop("pre_transform", None)

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

    def _apply_pre_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pre_transform is None:
            return tensor
        return self.pre_transform.transform(tensor)

    def _apply_inverse_pre_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pre_transform is None:
            return tensor
        return self.pre_transform.inverse(tensor)

    def apply_pre_transform_to_target(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.pre_transform_applies_to_target:
            return tensor
        return self._apply_pre_transform(tensor)

    def apply_inverse_pre_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pre_transform is None:
            return tensor
        return self._apply_inverse_pre_transform(tensor)

    @property
    def has_pre_transform(self) -> bool:
        return self.pre_transform is not None

    @property
    def pre_transform_applies_to_target(self) -> bool:
        return self.pre_transform is not None and self._pre_transform_apply_target

    @property
    def pre_transform_applies_inverse(self) -> bool:
        return self.pre_transform is not None and self._pre_transform_apply_inverse

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
        self._stft_config_source = new_cfg.to_public_dict()

    def stft_config_dict(self) -> Optional[Dict[str, Any]]:
        if self._stft_config_source is None:
            return None
        return deepcopy(self._stft_config_source)

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
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Any]],
        Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]],
    ]:
        """Run encoder and optional bottleneck on a spectrogram batch.

        Parameters
        ----------
        inputs : torch.Tensor
            Spectrograms shaped (B, C, F, T) or (B, C, T) depending on the
            encoder implementation.
        return_info : bool
            If True, also returns encoder and bottleneck metadata.
        debug : bool
            If True, prints shapes at key stages.

        Returns
        -------
        latents : torch.Tensor
            Encoded representation passed to the decoder (and bottleneck if
            present).
        encoder_info : dict, optional
            Metadata emitted by the encoder (e.g., ``feature_shape``) always
            included when ``return_info`` is True.
        bottleneck_info : dict, optional
            Metadata emitted by the bottleneck (only when present).
        """

        x = inputs
        if self.pre_transform is not None and self._pre_transform_apply_encoder:
            try:
                x = self._apply_pre_transform(x)
            except Exception as exc:
                warn(f"pre_transform.transform failed ({type(exc).__name__}: {exc}); continuing without it.")

        if debug:
            print(f"[AutoEncoder.encode] encoder input shape={tuple(x.shape)}")
        enc_out = self.encoder(x)
        enc_info: Dict[str, Any] = {}
        latents: torch.Tensor

        if isinstance(enc_out, tuple):
            # Standard contract: encoder returns (latents, info)
            latents, enc_info = enc_out[0], enc_out[1] if len(enc_out) > 1 else {}
        else:
            latents = enc_out

        if isinstance(enc_info, dict):
            enc_info = {**enc_info, "pre_bottleneck_latents": latents}
        else:
            enc_info = {"pre_bottleneck_latents": latents}

        if debug:
            print(f"Latents shape after encoder: {tuple(latents.shape)}")
            print(f"[AutoEncoder.encode] encoder output shape={tuple(latents.shape)}")

        # Store encoder_info for decode() - decoder will extract what it needs
        self._last_encoder_info = enc_info

        bottleneck_info: Optional[Dict[str, Any]] = None

        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True)

        if debug and self.bottleneck is not None:
            print(f"[AutoEncoder.encode] bottleneck output shape={tuple(latents.shape)}")

        if return_info:
            if bottleneck_info is not None:
                return latents, enc_info, bottleneck_info
            return latents, enc_info
        return latents

    def decode(
        self,
        latents: torch.Tensor,
        *,
        encoder_info: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        apply_inverse: Optional[bool] = None,
    ) -> torch.Tensor:
        """Decode latents back to spectrogram space.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representation, possibly sampled by a bottleneck.
        encoder_info : dict, optional
            Metadata from encoder. Passed directly to decoder which extracts
            what it needs (e.g., feature_shape, image_size). If None, uses
            cached info from last encode() call.
        debug : bool
            If True, prints decoder output shape.
        apply_inverse : bool, optional
            Override for applying the inverse pre-transform. Defaults to the
            configuration set in ``configure_pre_transform``.
        """

        z = latents
        if self.bottleneck is not None:
            z = self.bottleneck.decode(z)

        # Use provided encoder_info or fallback to cached from last encode()
        info = encoder_info if encoder_info is not None else getattr(self, '_last_encoder_info', None)

        # Pass encoder_info to decoder - each decoder extracts what it needs
        decoded = self.decoder(z, encoder_info=info)

        if apply_inverse is None:
            apply_inverse = self._pre_transform_apply_inverse

        if self.pre_transform is not None and apply_inverse:
            try:
                decoded = self._apply_inverse_pre_transform(decoded)
            except Exception as exc:
                warn(f"pre_transform.inverse failed ({type(exc).__name__}: {exc}); returning raw decoder output.")

        if debug:
            print(f"[AutoEncoder.decode] output shape={tuple(decoded.shape)}")

        return decoded

    def configure_pre_transform(self, spec: Optional[Union[str, Dict[str, Any]]]) -> None:
        self._pre_transform_spec = deepcopy(spec) if isinstance(spec, dict) else spec
        transform, apply_encoder, apply_inverse, apply_target = resolve_pre_transform(spec)
        self.pre_transform = transform
        self._pre_transform_apply_encoder = apply_encoder
        self._pre_transform_apply_inverse = apply_inverse
        self._pre_transform_apply_target = apply_target
        if self._pre_transform_spec is None:
            self._autoencoder_export_config.pop("pre_transform", None)
        else:
            stored = deepcopy(self._pre_transform_spec) if isinstance(self._pre_transform_spec, dict) else self._pre_transform_spec
            self._autoencoder_export_config["pre_transform"] = stored

    def pre_transform_description(self) -> str:
        if self.pre_transform is None:
            return "Pre-transform disabled."
        modes: List[str] = []
        if self._pre_transform_apply_encoder:
            modes.append("encode")
        if self._pre_transform_apply_inverse:
            modes.append("decode")
        if self._pre_transform_apply_target:
            modes.append("target")
        mode_desc = "/".join(modes) if modes else "none"
        return f"Pre-transform set to: {type(self.pre_transform).__name__} (applied on {mode_desc})"
          
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

    
    def export_model_config(self) -> Dict[str, Any]:
        """Summarise the architecture for inclusion in checkpoints and exports.

        The resulting dictionary is what gets embedded in ``inference_config``
        blobs.  It contains the canonical encoder/decoder/bottleneck specs plus
        the narrow ``autoencoder`` options (e.g. ``return_latent`` and
        ``pre_transform``) required to rebuild an identical model in a new
        process.
        """
        config: Dict[str, Any] = {}

        if self._autoencoder_export_config:
            config["autoencoder"] = deepcopy(self._autoencoder_export_config)

        if self._encoder_export_spec is not None:
            config["encoder"] = deepcopy(self._encoder_export_spec)

        if self._decoder_export_spec is not None:
            config["decoder"] = deepcopy(self._decoder_export_spec)

        if self._bottleneck_export_spec is not None:
            config["bottleneck"] = deepcopy(self._bottleneck_export_spec)

        return config

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AutoEncoder":
        """Instantiate the model from the payload produced by ``export_model_config``.
        It's used in train for building the model from the cfg passed to the trainer.
        Also, this is the entry point used by ``fast_inference`` when loading
        checkpoints.  It interprets the recorded module specs, handles the
        optional ``skip_bottleneck`` convenience, and recreates the
        ``AutoEncoder`` with the same structural flags that were active during
        training.
        """
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
    """Load a JSON export produced by ``export_model_config`` or ``build_from_json``.

    Keeping this helper alongside the model makes it straightforward for
    tooling such as ``fast_inference`` to hydrate an autoencoder from a saved
    configuration file without duplicating JSON parsing code elsewhere.
    """
    with open(path, "r") as f:
        return json.load(f)


def build_from_json(path: str) -> AutoEncoder:
    """Convenience wrapper combining :func:`load_config` and ``AutoEncoder.from_config``.

    External scripts can call this in a single line to recover a fully
    initialised model from a JSON export stored alongside a checkpoint.
    """
    return AutoEncoder.from_config(load_config(path))
