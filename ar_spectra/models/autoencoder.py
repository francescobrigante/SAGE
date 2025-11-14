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

console = Console()
def warn(msg):   console.print(msg, style="bold yellow")

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def _locate_class(class_path: Union[str, type]) -> type:
    """Supporta sia un path stringa 'pkg.mod.Class' sia una classe già passata."""
    if not isinstance(class_path, str):
        return class_path
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# for train and eval ds
def instantiate_from_spec(spec: Dict[str, Any]) -> Any:
    """
    spec = {
      "class": "pkg.mod.Class",
      "args": [...],          # opzionale
      "kwargs": { ... }       # opzionale
    }
    """
    if "class" not in spec:
        raise ValueError("spec manca la chiave 'class'.")
    cls = _locate_class(spec["class"])
    args = spec.get("args", []) or []
    kwargs = spec.get("kwargs", {}) or {}
    return cls(*args, **kwargs)


class AutoEncoder(nn.Module):
    """
    Contenitore generico. Gestisce forward encoder->decoder.
    Se return_latent=True, forward ritorna (ricostruzione, latente).
    """
    def __init__(
        self,
        encoder: Union[nn.Module, Dict[str, Any], str, type],
        decoder: Union[nn.Module, Dict[str, Any], str, type],
        bottleneck: Optional[Union[nn.Module, Dict[str, Any], str, type]] = None,

        return_latent: bool = False,
    ) -> None:
        super().__init__()
        # Permette di passare direttamente istanze oppure specifiche/nomi di classe
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
                    f"Tip: set encoder.dimension=2*C and decoder.input_size=C."
                )
            else:
                warnings.warn("VAEBottleneck active but unable to deduce enc_dim/dec_in for check. Ensure encoder.dimension=2*C and decoder.input_size=C. if using VAE bottleneck.")
        elif isinstance(self.bottleneck, SkipBottleneck):
            if (enc_dim is not None) and (dec_in is not None):
                assert enc_dim == dec_in or dec_in % enc_dim == 0, (
                    f"Config mismatch with SkipBottleneck: encoder channels={enc_dim} "
                    f"must be either {dec_in} or n*{dec_in}. "
                    f"Tip: set encoder.dimension=C and decoder.input_size=C or encoder.dimension=n*C and decoder.input_size=C."
                )
        else:
            if (enc_dim is not None) and (dec_in is not None):
                assert enc_dim == dec_in, (
                    f"Encoder/Decoder channel mismatch without bottleneck: "
                    f"{enc_dim} vs {dec_in}"
                )
            elif self.bottleneck is None:
                warnings.warn("No bottleneck selected: unable to verify enc/dec because dimensions are not deducible.")
        
    
    def encode(self, audio, skip_bottleneck: bool = False, return_info=False, iterate_batch=False, **kwargs):

        info = {}

        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1]))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio)
        else:
            latents = audio

        info["pre_bottleneck_latents"] = latents

        # JSON is the only ground truth: if a SkipBottleneck is set, we never apply VAEs or others.
        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
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
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)
        
        return decoded
          
        
    def _maybe_add_nyquist(self, S: torch.Tensor, n_fft: int, onesided: bool) -> torch.Tensor:
        F = S.shape[-2]
        if n_fft is not None and onesided and F == n_fft // 2:
            pad_shape = list(S.shape); pad_shape[-2] = 1
            nyq = torch.zeros(pad_shape, dtype=S.dtype, device=S.device)
            return torch.cat([S, nyq], dim=-2)
        return S

    def istft(self, spec: torch.Tensor, **kwargs):
        def to_complex(S: torch.Tensor) -> torch.Tensor:
            if torch.is_complex(S):
                return S
            if S.dim() == 4:  # (B, 2C, F, T) -> (B, C, F, T) complex
                B, Cx, F, T = S.shape
                assert Cx % 2 == 0
                C = Cx // 2
                S_ri = S.reshape(B, C, 2, F, T).permute(0,1,3,4,2).contiguous()
                return torch.view_as_complex(S_ri)
            if S.dim() == 3:  # (2C, F, T) -> (C, F, T) complex
                Cx, F, T = S.shape
                assert Cx % 2 == 0
                C = Cx // 2
                S_ri = S.reshape(C, 2, F, T).permute(0,2,3,1).contiguous()
                return torch.view_as_complex(S_ri)
            raise ValueError(f"Unsupported spec shape {tuple(S.shape)}")

        S = to_complex(spec)  # [B, C, F, T] or [B, F, T]

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
        # window coerente
        if window is None and win_length is not None:
            real_dtype = torch.float32 if S.dtype == torch.complex64 else torch.float64
            window = torch.hann_window(win_length, dtype=real_dtype, device=S.device)

        # deduci onesided se non passato
        F = S.shape[-2]
        if onesided is None:
            if n_fft is not None and (F == n_fft // 2 or F == n_fft // 2 + 1):
                onesided = True
            else:
                onesided = (n_fft is not None and F == n_fft // 2 + 1)
        # ripristina Nyquist solo quando serve
        S = self._maybe_add_nyquist(S, n_fft, onesided)


        # lunghezza
        T = S.shape[-1]
        if target_length is None and center:
            target_length = hop_length*(T-1) + (win_length or n_fft)

        # collassa canali nel batch
        if S.dim() == 4:  # [B, C, F, T]
            B, C, F, T = S.shape
            Sbc = S.reshape(B*C, F, T).contiguous()
            y = torch.istft(Sbc, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                            window=window, center=center, normalized=normalized,
                            onesided=onesided, return_complex=False)
            return y.reshape(B, C, -1)
        elif S.dim() == 3:  # [B, F, T]
            return torch.istft(S, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                            window=window, center=center, normalized=normalized,
                            onesided=onesided, return_complex=False)
        else:
            raise RuntimeError(f"Unexpected complex shape {S.shape}")

    
    # TODO: AR_SPECTRA: ADJUST CHUNKED ENCODE/DECODE
    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = int(self.downsampling_ratio)
            total_size = audio.shape[2] # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent # converting metric in latents to samples
            overlap *= samples_per_latent # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = audio[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size,self.latent_dim,y_size), dtype = chunks.dtype).to(audio.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap//samples_per_latent//2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = latents[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = int(self.downsampling_ratio)
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size,self.out_channels,y_size), dtype = chunks.dtype).to(latents.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap//2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final


    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AutoEncoder":
        ae_kwargs = cfg.get("autoencoder", {}) or {}
        encoder_spec = cfg["encoder"]
        decoder_spec = cfg["decoder"]
        bottleneck_spec = cfg.get("bottleneck", None)

        # It keeps only the keys supported by the AutoEncoder constructor
        allowed_ae_keys = {"return_latent"}
        unknown = set(ae_kwargs.keys()) - allowed_ae_keys
        if unknown:
            warn(f"AutoEncoder.from_config: ignoring unsupported autoencoder kwargs {sorted(unknown)}")
        ae_kwargs = {k: v for k, v in ae_kwargs.items() if k in allowed_ae_keys}

        # Decide skip from JSON only (unique source of truth)
        skip_flag = False
        target_channels: Optional[int] = None
        if isinstance(bottleneck_spec, dict):
            # Support either top-level flags inside "bottleneck" or inside its kwargs
            bn_kwargs = bottleneck_spec.get("kwargs", {}) or {}
            skip_flag = bool(
                bottleneck_spec.get("skip_bottleneck", False)
                or bottleneck_spec.get("skip", False)
                or bn_kwargs.get("skip_bottleneck", False)
                or bn_kwargs.get("skip", False)
            )
        # Infer decoder latent channels (input_size / in_channels / dimension)
        if isinstance(decoder_spec, dict):
            dkw = decoder_spec.get("kwargs", {}) or {}
            for k in ("input_size", "dimension", "in_channels"):
                if k in dkw and isinstance(dkw[k], (int, float)):
                    target_channels = int(dkw[k])
                    break

        if skip_flag:
            warn("Warning: bottleneck skipped; disable skip_bottleneck to undo this behavior.")
            bottleneck_inst = SkipBottleneck(target_channels=target_channels)
            return cls(encoder_spec, decoder_spec, bottleneck=bottleneck_inst, **ae_kwargs)

        return cls(encoder_spec, decoder_spec, bottleneck=bottleneck_spec, **ae_kwargs)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_from_json(path: str) -> AutoEncoder:
    return AutoEncoder.from_config(load_config(path))


