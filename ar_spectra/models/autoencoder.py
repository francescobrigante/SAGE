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
from ar_spectra.models.bottlenecks import VAEBottleneck, SkipBottleneck
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
        # Separate STFT/ISTFT kwargs that should NOT go to encoder/bottleneck forward.
        STFT_PARAM_KEYS = {"n_fft", "hop_length", "win_length", "window", "center", "normalized", "onesided", "length"}
        encode_kwargs = {k: v for k, v in kwargs.items() if k not in STFT_PARAM_KEYS}
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
                    latents.append(self.encoder(x_i))  # encoder forward should not receive STFT params
                latents = torch.cat(latents, dim=0)
            else:
                x = audio
                if getattr(self, "pre_transform", None) is not None:
                    try:
                        x = self.pre_transform.transform(x)
                    except Exception:
                        pass
                latents = self.encoder(x)  # avoid passing unrelated kwargs
        else:
            latents = audio

        info["pre_bottleneck_latents"] = latents

        # JSON config is the ground truth: if SkipBottleneck is set we never apply VAEs or others.
        if self.bottleneck is not None:
            # Pass only non-STFT kwargs to bottleneck.
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **encode_kwargs)
            info.update(bottleneck_info)
        
        if return_info:
            return latents, info
        return latents

    def decode(self, latents, skip_bottleneck: bool = False, iterate_batch=False, **kwargs):
        # Filter out STFT-related kwargs that belong to ISTFT only.
        STFT_PARAM_KEYS = {"n_fft", "hop_length", "win_length", "window", "center", "normalized", "onesided", "length"}
        decode_kwargs = {k: v for k, v in kwargs.items() if k not in STFT_PARAM_KEYS}
        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    dec_i = self.bottleneck.decode(latents[i:i+1])  # bottleneck decode receives no STFT params
                    decoded.append(dec_i)
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                y_i = self.decoder(latents[i:i+1], **decode_kwargs)
                if getattr(self, "pre_transform", None) is not None:
                    try:
                        y_i = self.pre_transform.inverse(y_i)
                    except Exception:
                        pass
                decoded.append(y_i)
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **decode_kwargs)
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
        **stft_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode raw waveform(s) into latents with optional chunking + overlap.

        Accepts audio [T], [C,T], or [B,C,T]; normalizes to [B,C,T].
        When chunked, splits into overlapping chunks with step chunk_size-overlap_size.
        Returns concatenated latents and info dict for decode_audio.
        """
        # Normalize shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        elif audio.dim() != 3:
            raise ValueError(f"audio must be 1D/2D/3D, got {tuple(audio.shape)}")
        B, C, T_total = audio.shape
        if stereo and C == 1:
            audio = audio.repeat(1, 2, 1)
            C = 2
        if not stereo and C > 2:
            raise ValueError("If stereo=False expected mono or stereo only.")
        if debug:
            print("[encode_audio] ---- PIPELINE START ----")
            print(f"[encode_audio] input waveform shape={audio.shape} (B={B}, C={C}, T={T_total})")
            print(f"[encode_audio] chunked={chunked}, chunk_size={chunk_size}, overlap_size={overlap_size}")
            print(f"[encode_audio] STFT kwargs={stft_kwargs}")

        info: Dict[str, Any] = {
            "original_length": T_total,
            "chunked": chunked,
            "overlap_size": overlap_size,
        }
        if not chunked:
            chunks = [(0, T_total)]
            padded_length = T_total
        else:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be > 0 when chunked=True")
            chunks, padded_length = self._plan_chunks(T_total, chunk_size, overlap_size)
            if padded_length > T_total:
                pad_amt = padded_length - T_total
                audio = torch.nn.functional.pad(audio, (0, pad_amt))
            info["padded_length"] = padded_length
        info["chunk_boundaries"] = chunks

        latent_chunks: List[torch.Tensor] = []
        latent_chunk_lengths: List[int] = []
        stft_frames_per_chunk: List[int] = []

        for i, (s, e) in enumerate(chunks):
            wav_chunk = audio[:, :, s:e]
            if debug:
                print(f"[encode_audio] ---- CHUNK {i} ----")
                print(f"[encode_audio] slice samples=({s}, {e}) -> wav_chunk shape={wav_chunk.shape}")
            spec = self.stft(wav_chunk, **stft_kwargs)  # complex [B,C,F,TT]
            if debug:
                print(f"[encode_audio] STFT -> complex spec shape={spec.shape}")
            spec_in = self._pack_complex(spec) if pack_complex else spec
            if debug:
                packing_msg = "packed real/imag (2*C channels)" if pack_complex else "kept complex tensor"
                print(f"[encode_audio] spec post-pack: shape={spec_in.shape} ({packing_msg})")
            lat = self.encode(spec_in)
            if debug:
                print(f"[encode_audio] encoder output latents shape={lat.shape}")
            latent_chunks.append(lat)
            latent_chunk_lengths.append(lat.shape[-1])
            stft_frames_per_chunk.append(spec.shape[-1])
            if debug:
                print(f"[encode_audio] chunk {i} wav {wav_chunk.shape} spec {spec_in.shape} lat {lat.shape}")

        latents = torch.cat(latent_chunks, dim=-1) if latent_chunks else torch.empty(0)
        info["latent_chunk_lengths"] = latent_chunk_lengths
        info["stft_frames_per_chunk"] = stft_frames_per_chunk
        # Store expected sample lengths per chunk (before any padding trimming at decode)
        info["chunk_expected_lengths"] = [e - s for (s, e) in chunks]
        info["latents_shape"] = tuple(latents.shape)
        if debug:
            print(f"[encode_audio] ---- PIPELINE END ----")
            print(f"[encode_audio] concatenated latents shape={latents.shape}")
            print(f"[encode_audio] metadata={info}")
        return latents, info

    def decode_audio(
        self,
        latents: torch.Tensor,
        info: Dict[str, Any],
        *,
        stereo: bool = True,
        chunked: bool = False,
        pack_complex: bool = True,
        debug: bool = False,
        remove_padding: bool = True,
        **istft_kwargs,
    ) -> torch.Tensor:
        """Decode concatenated latents back to waveform with optional chunked Hann crossfade reconstruction."""
        if latents.dim() < 3:
            raise ValueError("latents expected >=3D with time axis last")
        original_length = info.get("original_length")
        padded_length = info.get("padded_length", original_length)
        chunk_boundaries: List[Tuple[int,int]] = info.get("chunk_boundaries", [(0, original_length)])
        latent_chunk_lengths: List[int] = info.get("latent_chunk_lengths", [latents.shape[-1]])
        overlap_size = info.get("overlap_size", 0)

        if not chunked:
            if debug:
                print("[decode_audio] ---- PIPELINE START (single chunk) ----")
                print(f"[decode_audio] latent tensor shape={latents.shape}")
                print(f"[decode_audio] pack_complex={pack_complex}, ISTFT kwargs={istft_kwargs}")
            spec = self.decode(latents)
            spec_complex = self._unpack_complex(spec) if pack_complex else (spec if torch.is_complex(spec) else spec)
            if debug:
                print(f"[decode_audio] decoder output spec shape={spec.shape}")
                print(f"[decode_audio] spec converted to complex shape={spec_complex.shape}")

            expected_frames = (info.get("stft_frames_per_chunk", [spec_complex.shape[-1]]) or [spec_complex.shape[-1]])[0]
            cur_frames = spec_complex.shape[-1]
            target_len = istft_kwargs.get("length")
            if target_len is None:
                target_len = original_length or (info.get("chunk_expected_lengths", [None])[0])
            istft_args = dict(istft_kwargs)
            if target_len is not None:
                istft_args["length"] = target_len
            wav = self.istft(spec_complex, **istft_args)
            if remove_padding and wav.shape[-1] >= original_length:
                wav = wav[..., :original_length]
            if debug:
                print(f"[decode_audio] ISTFT -> waveform shape={wav.shape} (trimmed_to_original={remove_padding})")
                print("[decode_audio] ---- PIPELINE END ----")
            return wav

        if sum(latent_chunk_lengths) != latents.shape[-1]:
            raise ValueError("Sum of latent_chunk_lengths does not match latents time axis.")
        fade_in, fade_out = self._hann_crossfade_windows(overlap_size)
        if fade_in is not None:
            fade_in = fade_in.to(latents.device)
            fade_out = fade_out.to(latents.device)

        # Split latents
        cursor = 0
        lat_list: List[torch.Tensor] = []
        for L in latent_chunk_lengths:
            lat_list.append(latents[..., cursor:cursor+L])
            cursor += L

        decoded_wave_chunks: List[torch.Tensor] = []
        for i, lat_chunk in enumerate(lat_list):
            if debug:
                print(f"[decode_audio] ---- CHUNK {i} ----")
                print(f"[decode_audio] latent slice shape={lat_chunk.shape} expected_samples={chunk_boundaries[i] if i < len(chunk_boundaries) else None}")
            spec_chunk = self.decode(lat_chunk)
            spec_complex = self._unpack_complex(spec_chunk) if pack_complex else (spec_chunk if torch.is_complex(spec_chunk) else spec_chunk)
            if debug:
                print(f"[decode_audio] decoder output spec shape={spec_chunk.shape}")
                print(f"[decode_audio] spec converted to complex shape={spec_complex.shape}")
            expected_len = (chunk_boundaries[i][1] - chunk_boundaries[i][0]) if i < len(chunk_boundaries) else None
            if expected_len is not None:
                wav_chunk = self.istft(spec_complex, length=expected_len, **istft_kwargs)
            else:
                wav_chunk = self.istft(spec_complex, **istft_kwargs)
            decoded_wave_chunks.append(wav_chunk)
            if debug:
                print(f"[decode_audio] ISTFT -> wav_chunk shape={wav_chunk.shape}")

        out = torch.zeros(decoded_wave_chunks[0].shape[0], decoded_wave_chunks[0].shape[1], padded_length,
                          device=decoded_wave_chunks[0].device, dtype=decoded_wave_chunks[0].dtype)
        for i, ((s, e), wav_chunk) in enumerate(zip(chunk_boundaries, decoded_wave_chunks)):
            expected_len = e - s
            if wav_chunk.shape[-1] != expected_len:
                diff = expected_len - wav_chunk.shape[-1]
                if diff > 0:
                    wav_chunk = torch.nn.functional.pad(wav_chunk, (0, diff))
                else:
                    wav_chunk = wav_chunk[..., :expected_len]
            if overlap_size > 0 and fade_in is not None:
                if i > 0:
                    wav_chunk[..., :overlap_size] *= fade_in.view(1,1,-1)
                if i < len(decoded_wave_chunks) - 1:
                    wav_chunk[..., -overlap_size:] *= fade_out.view(1,1,-1)
            out[..., s:e] += wav_chunk
        if remove_padding and out.shape[-1] >= original_length:
            out = out[..., :original_length]
        if debug:
            print("[decode_audio] overlap-add reconstruction completed.")
            print(f"[decode_audio] final waveform shape={out.shape} (trimmed_to_original={remove_padding})")
            print("[decode_audio] ---- PIPELINE END ----")
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
