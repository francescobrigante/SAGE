from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any
import json
import importlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import warnings
from .bottlenecks import VAEBottleneck, SkipBottleneck
from rich.console import Console
from ar_spectra.training_utils.pre_transform import create_pre_transform
import torchaudio

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

# For training and eval dataset instantiation
def instantiate_from_spec(spec: Dict[str, Any]) -> Any:
    """
    spec = {
      "class": "pkg.mod.Class",
      "args": [...],          # optional
      "kwargs": { ... }       # optional
    }
    """
    if "class" not in spec:
        raise ValueError("spec is missing the 'class' key.")
    cls = _locate_class(spec["class"])
    args = spec.get("args", []) or []
    kwargs = spec.get("kwargs", {}) or {}
    return cls(*args, **kwargs)


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

    def encode(self, audio, skip_bottleneck: bool = False, return_info=False, iterate_batch=False, **kwargs):
        info = {}
        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    x_i = audio[i:i+1]
                    if getattr(self, "pre_transform", None) is not None:
                        try:
                            x_i = self.pre_transform.transform(x_i)
                        except Exception:
                            pass
                    latents.append(self.encoder(x_i))
                latents = torch.cat(latents, dim=0)
            else:
                x = audio
                if getattr(self, "pre_transform", None) is not None:
                    try:
                        x = self.pre_transform.transform(x)
                    except Exception:
                        pass
                latents = self.encoder(x)
        else:
            latents = audio

        info["pre_bottleneck_latents"] = latents

        # JSON config is the ground truth: if SkipBottleneck is set we never apply VAEs or others.
        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)
        
        if return_info:
            return latents, info
        return latents

    def decode(self, latents, skip_bottleneck: bool = False, iterate_batch=False, **kwargs):
        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    dec_i = self.bottleneck.decode(latents[i:i+1])
                    decoded.append(dec_i)
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                y_i = self.decoder(latents[i:i+1], **kwargs)
                if getattr(self, "pre_transform", None) is not None:
                    try:
                        y_i = self.pre_transform.inverse(y_i)
                    except Exception:
                        pass
                decoded.append(y_i)
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)
            if getattr(self, "pre_transform", None) is not None:
                try:
                    decoded = self.pre_transform.inverse(decoded)
                except Exception:
                    pass
        
        return decoded
          
    def _maybe_add_nyquist(self, S: torch.Tensor, n_fft: int, onesided: bool) -> torch.Tensor:
        F = S.shape[-2]
        if n_fft is not None and onesided and F == n_fft // 2:
            pad_shape = list(S.shape); pad_shape[-2] = 1
            nyq = torch.zeros(pad_shape, dtype=S.dtype, device=S.device)
            return torch.cat([S, nyq], dim=-2)
        return S
    
    def stft(self, audio: torch.Tensor, **kwargs):
        """
        Compute complex STFT of a waveform batch.
        audio: [B, C, T]
        Returns: complex tensor [B, C, F, TT]
        """
        if torch.is_complex(audio):
            raise ValueError("stft expects real waveform input.")
        n_fft      = kwargs.get("n_fft", 1024)
        hop_length = kwargs.get("hop_length", n_fft // 4)
        win_length = kwargs.get("win_length", n_fft)
        window     = kwargs.get("window", torch.hann_window(win_length, device=audio.device, dtype=audio.dtype))
        center     = bool(kwargs.get("center", True))
        normalized = bool(kwargs.get("normalized", False))
        onesided   = kwargs.get("onesided", True)

        B, C, T = audio.shape
        specs = []
        for c in range(C):
            S_c = torch.stft(audio[:, c, :],
                             n_fft=n_fft,
                             hop_length=hop_length,
                             win_length=win_length,
                             window=window,
                             center=center,
                             normalized=normalized,
                             onesided=onesided,
                             return_complex=True)
            specs.append(S_c.unsqueeze(1))
        return torch.cat(specs, dim=1)  # [B, C, F, TT]
        
    def istft(self, spec: torch.Tensor, **kwargs):
        """
        Inverse STFT.
        Accepts either complex spectrogram [B, C, F, T] or stacked real/imag [B, 2C, F, T] (or [2C, F, T]).
        Returns real waveform [B, C, T].
        """
        # Inline replacement for previous to_complex()
        if torch.is_complex(spec):
            S = spec
        elif spec.dim() == 4:  # [B, 2C, F, T]
            B, Cx, F, T = spec.shape
            assert Cx % 2 == 0, "Channel count must be even (real/imag)."
            C = Cx // 2
            real = spec[:, :C]
            imag = spec[:, C:]
            S = torch.complex(real, imag)
        elif spec.dim() == 3:  # [2C, F, T] (rare path)
            Cx, F, T = spec.shape
            assert Cx % 2 == 0, "Channel count must be even (real/imag)."
            C = Cx // 2
            real = spec[:C]
            imag = spec[C:]
            S = torch.complex(real, imag)
        else:
            raise ValueError(f"Unsupported spectrogram shape {tuple(spec.shape)}")

        n_fft      = kwargs.get("n_fft")
        hop_length = kwargs.get("hop_length")
        win_length = kwargs.get("win_length")
        window     = kwargs.get("window")
        center     = bool(kwargs.get("center", True))
        target_length = kwargs.get("length", None) 
        normalized = bool(kwargs.get("normalized", False))
        onesided   = kwargs.get("onesided", None)
        
        if normalized is None:
            normalized = False
        if window is None and win_length is not None:
            real_dtype = torch.float32 if S.dtype == torch.complex64 else torch.float64
            window = torch.hann_window(win_length, dtype=real_dtype, device=S.device)

        F_bins = S.shape[-2]
        if onesided is None:
            if n_fft is not None and (F_bins == n_fft // 2 or F_bins == n_fft // 2 + 1):
                onesided = True
            else:
                onesided = (n_fft is not None and F_bins == n_fft // 2 + 1)

        S = self._maybe_add_nyquist(S, n_fft, onesided)

        T_frames = S.shape[-1]
        if target_length is None and center:
            target_length = hop_length*(T_frames-1) + (win_length or n_fft)

        if S.dim() == 4:  # [B, C, F, T]
            B, C, F_bins, T_frames = S.shape
            Sbc = S.reshape(B*C, F_bins, T_frames).contiguous()
            y = torch.istft(Sbc, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                            window=window, center=center, normalized=normalized,
                            onesided=onesided, return_complex=False, length=target_length)
            return y.reshape(B, C, -1)
        elif S.dim() == 3:  # [B, F, T]
            return torch.istft(S, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                               window=window, center=center, normalized=normalized,
                               onesided=onesided, return_complex=False, length=target_length)
        else:
            raise RuntimeError(f"Unexpected complex shape {S.shape}")

    def encode_audio(self,
                     audio: torch.Tensor,
                     chunked: bool = False,
                     overlap: int = 32,
                     chunk_size: int = 128,
                     pack_complex: bool = True,
                     **kwargs):
        """
        Waveform -> STFT -> (optional pre_transform inside encode) -> latents.
        If chunked=True performs encoding on overlapping waveform segments.
        overlap and chunk_size are expressed in 'latent steps':
        uses self.downsampling_ratio or hop_length as fallback.
        """
        if not chunked:
            S = self.stft(audio, **kwargs)  # complex [B, C, F, Tt]
            if pack_complex:
                S = self._pack_complex(S)    # [B, 2C, F, Tt]
            latents = self.encode(S, **kwargs)
            return latents

        n_fft      = kwargs.get("n_fft", 2048)
        hop_length = kwargs.get("hop_length", n_fft // 4)
        samples_per_latent = int(getattr(self, "downsampling_ratio", hop_length))
        total_size = audio.shape[-1]
        batch_size = audio.shape[0]
        chunk_size_samples = chunk_size * samples_per_latent
        overlap_samples    = overlap * samples_per_latent
        hop_samples        = chunk_size_samples - overlap_samples
        chunks = []
        for start in range(0, total_size - chunk_size_samples + 1, hop_samples):
            end = start + chunk_size_samples
            chunks.append(audio[:, :, start:end])
        if end != total_size:
            chunks.append(audio[:, :, -chunk_size_samples:])
        chunks = torch.stack(chunks)  # [N, B, C, chunk_size_samples]
        num_chunks = chunks.shape[0]

        y_size = total_size // samples_per_latent
        latent_channels = getattr(self, "latent_dim", None)
        if latent_channels is None:
            for k in ["dimension", "latent_dim", "out_channels"]:
                if hasattr(self.encoder, k):
                    latent_channels = int(getattr(self.encoder, k))
                    break
        if latent_channels is None:
            raise RuntimeError("Unable to determine latent_dim for final buffer.")
        y_final = torch.zeros((batch_size, latent_channels, y_size), dtype=audio.dtype, device=audio.device)

        for i in range(num_chunks):
            wav_chunk = chunks[i]
            S_chunk = self.stft(wav_chunk, **kwargs)
            if pack_complex:
                S_chunk = self._pack_complex(S_chunk)
            y_chunk = self.encode(S_chunk, **kwargs)
            if i == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[-1]
            else:
                t_start = i * hop_samples // samples_per_latent
                t_end = t_start + chunk_size_samples // samples_per_latent
            ol = overlap_samples // samples_per_latent // 2
            c_start = 0
            c_end = y_chunk.shape[-1]
            if i > 0:
                t_start += ol
                c_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                c_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, c_start:c_end]

        return y_final
    
    def decode_audio(self,
                     latents: torch.Tensor,
                     chunked: bool = False,
                     overlap: int = 32,
                     chunk_size: int = 128,
                     packed_input: bool = True,
                     **kwargs):
        """
        Latents -> spectrogram (self.decode applies inverse pre_transform) -> ISTFT -> waveform.
        If chunked=True decodes overlapping latent segments and reassembles waveform.
        packed_input=True means decoder returns stacked spectrogram (2C).
        """
        if not chunked:
            S_rec = self.decode(latents, **kwargs)
            if packed_input:
                S_rec = self._unpack_complex(S_rec)
            wav = self.istft(S_rec, **kwargs)
            return wav

        n_fft      = kwargs.get("n_fft", 2048)
        hop_length = kwargs.get("hop_length", n_fft // 4)
        samples_per_latent = int(getattr(self, "downsampling_ratio", hop_length))
        total_size_latent = latents.shape[-1]
        batch_size = latents.shape[0]
        hop_latent = chunk_size - overlap

        chunks = []
        for start in range(0, total_size_latent - chunk_size + 1, hop_latent):
            end = start + chunk_size
            chunks.append(latents[:, :, start:end])
        if end != total_size_latent:
            chunks.append(latents[:, :, -chunk_size:])
        chunks = torch.stack(chunks)  # [N, B, D, Lc]
        num_chunks = chunks.shape[0]

        waveform_len = total_size_latent * samples_per_latent
        out_channels = getattr(self, "out_channels", None)
        if out_channels is None:
            for k in ["out_channels", "channels", "n_channels"]:
                if hasattr(self.decoder, k):
                    out_channels = int(getattr(self.decoder, k))
                    break
        if out_channels is None:
            out_channels = 1
        y_final = torch.zeros((batch_size, out_channels, waveform_len), dtype=latents.dtype, device=latents.device)

        for i in range(num_chunks):
            y_chunk_lat = chunks[i]
            S_chunk = self.decode(y_chunk_lat, **kwargs)
            if packed_input:
                S_chunk = self._unpack_complex(S_chunk)
            wav_chunk = self.istft(S_chunk, **kwargs)
            if i == num_chunks - 1:
                t_end = waveform_len
                t_start = t_end - wav_chunk.shape[-1]
            else:
                t_start = i * hop_latent * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            ol = (overlap // 2) * samples_per_latent
            c_start = 0
            c_end = wav_chunk.shape[-1]
            if i > 0:
                t_start += ol
                c_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                c_end -= ol
            y_final[:, :, t_start:t_end] = wav_chunk[:, :, c_start:c_end]

        return y_final

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


