from __future__ import annotations

# =============================================================================
# Inference: rebuild SAGE from a checkpoint and encode / decode waveforms.
#
#   codec = SAGE.from_checkpoint("SAGE_FTe992.ckpt", device="cuda")
#   z     = codec.encode(wav)                               # (B, 2, N) -> (B, 16, T_lat)
#   rec   = codec.decode(z, target_length=wav.shape[-1])    # -> (B, 2, N)
# Inputs longer than one chunk (300 s by default) run in chunks with context margins (sage.chunked).
# =============================================================================

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn.functional as F

from sage import chunked
from sage.constants import DEFAULT_CHUNK_MARGIN_SECONDS, DEFAULT_CHUNK_SECONDS
from sage.model.autoencoder import SAGEAutoencoder
from sage.utils.console import ok, warn
from sage.compat import upgrade_model_config
from sage.model.swin.sdpa import enable_sdpa
from sage.model.swin.varlen import enable_varlen, load_config, resolve


def _normalize_ae_key_value(key: str, value: Any) -> tuple[str, Any]:
    """Strip torch.compile wrapper prefix and convert _real_view back to complex.

    make_complex_model_ddp_compatible renames complex params from ``weight``
    to ``weight_real_view`` (float32 view_as_real).  torch.compile wraps the
    module under ``_orig_mod.``.  Both must be undone at inference time so
    the state dict matches the freshly-built SAGEAutoencoder.
    """
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]
    if key.endswith("_real_view") and isinstance(value, torch.Tensor):
        orig_key = key[: -len("_real_view")]
        try:
            value = torch.view_as_complex(value.contiguous())
            key = orig_key
        except Exception:
            pass  # keep original name/value if conversion fails
    return key, value


def _extract_autoencoder_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip trainer-specific prefixes so checkpoints load into a bare ``SAGEAutoencoder``.

    Lightning checkpoints store model weights under entries such as
    ``"engine.autoencoder.encoder.0.weight"``.  Inference code only owns the
    raw :class:`SAGEAutoencoder`, so we project those keys back to the namespace the
    model expects.

    Handles two extra transformations needed for Swin checkpoints:
    - ``_orig_mod.`` prefix added by ``torch.compile``
    - ``_real_view`` suffix added by ``make_complex_model_ddp_compatible`` for
      NCCL/DDP compatibility; converted back with ``torch.view_as_complex``.

    Prioritizes Exponential Moving Average (EMA) weights if present in the
    checkpoint for maximum reconstruction fidelity.
    """
    # 1. Try to extract EMA weights (preferred for evaluation)
    ema_prefixes = (
        "engine.autoencoder_ema.ema_model.",
        "autoencoder_ema.ema_model.",
        "autoencoder_ema.online_model.",   # SAO / stable-audio-tools EMA
        "ema_autoencoder._orig_mod.",      # Swin compiled checkpoint
        "ema_autoencoder.",                # Swin standard checkpoint
    )
    cleaned: Dict[str, Any] = {}

    for key, value in state_dict.items():
        for prefix in ema_prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                norm_key, norm_value = _normalize_ae_key_value(stripped, value)
                cleaned[norm_key] = norm_value
                break

    if cleaned:
        print("[CHECKPOINT] Loaded autoencoder weights from EMA")
        return cleaned

    # 2. Fallback to standard autoencoder weights if EMA is not present
    # List longer prefixes first so torch.compile variant matches before the plain one.
    prefixes = (
        "engine.autoencoder._orig_mod.",  # compiled
        "engine.autoencoder.",
        "autoencoder.",
    )
    for key, value in state_dict.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                norm_key, norm_value = _normalize_ae_key_value(stripped, value)
                cleaned[norm_key] = norm_value
                matched = True
                break
        if not matched and (
            key.startswith("encoder.")
            or key.startswith("decoder.")
            or key.startswith("bottleneck.")
        ):
            norm_key, norm_value = _normalize_ae_key_value(key, value)
            cleaned[norm_key] = norm_value

    return cleaned if cleaned else state_dict


def pad_for_swin(wav: torch.Tensor, hop_length: int, num_downsamples: int) -> tuple[torch.Tensor, int]:
    """Right-pad a waveform so its STFT frame count is a multiple of ``2**num_downsamples``.

    The Swin encoder halves the time axis ``num_downsamples`` times, so the frame count
    ``N // hop + 1`` must divide evenly. This is the padding used by the paper's evaluators
    (``evaluation/codecs.py::SAGEAdapter``, via :meth:`SAGE.reconstruct`). Returns ``(padded, original_length)``.
    """
    orig = wav.shape[-1]
    mult = 2 ** num_downsamples
    frames = (orig // hop_length) + 1
    pad = (mult - frames % mult) % mult
    target = max(orig, (frames + pad - 1) * hop_length)
    if target > orig:
        wav = F.pad(wav, (0, target - orig))
    return wav, orig


def encode_audio(
    autoencoder: SAGEAutoencoder,
    audio: torch.Tensor,
    deterministic: bool = False,
    debug: bool = False,
) -> torch.Tensor:
    """Encode waveform to latents.

    Parameters
    ----------
    autoencoder : SAGEAutoencoder
        Model with STFT config set.
    audio : torch.Tensor
        Waveform tensor of shape (T,), (C, T), or (B, C, T).
    deterministic : bool
        If True, the VAE bottleneck returns the posterior mean μ instead of
        a reparameterized sample z.  Recommended at inference time.
    debug : bool
        Print debug info.

    Returns
    -------
    torch.Tensor
        Latent tensor.
    """
    spec = _spectrogram(autoencoder, audio, debug)

    # Encode
    latents = autoencoder.encode(spec, deterministic=deterministic)
    if debug:
        print(f"[encode_audio] latents shape: {tuple(latents.shape)}")

    return latents


def _spectrogram(autoencoder: SAGEAutoencoder, audio: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """Waveform ``(T,)``, ``(C, T)`` or ``(B, C, T)`` -> the encoder's input spectrogram."""
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
    return spec


def _encode_params(autoencoder: SAGEAutoencoder, audio: torch.Tensor) -> torch.Tensor:
    """Waveform -> the encoder output before the bottleneck (the posterior ``[mu | scale]`` of a VAE)."""
    out = autoencoder.encode(_spectrogram(autoencoder, audio), return_info=True, deterministic=True)
    return out[1]["pre_bottleneck_latents"]


def decode_audio(
    autoencoder: SAGEAutoencoder,
    latents: torch.Tensor,
    target_length: Optional[int] = None,
    debug: bool = False,
) -> torch.Tensor:
    """Decode latents to waveform.
    
    Parameters
    ----------
    autoencoder : SAGEAutoencoder
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


class SAGE:
    """High-level checkpoint loader for inference.

    Loads a Lightning checkpoint, rebuilds the model architecture from the
    stored configuration, and provides simple encode/decode methods.
    
    Example
    -------
    >>> codec = SAGE.from_checkpoint("SAGE_FTe992.ckpt")
    >>> latents = codec.encode(audio)
    >>> reconstructed = codec.decode(latents, target_length=audio.shape[-1])
    """

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
        varlen: Optional[str] = None,
        sdpa: bool = True,
        chunk_seconds: Optional[float] = DEFAULT_CHUNK_SECONDS,
        chunk_margin_seconds: float = DEFAULT_CHUNK_MARGIN_SECONDS,
    ) -> "SAGE":
        """Load a SAGE checkpoint (EMA weights when present).

        Args:
            checkpoint_path: Lightning checkpoint written by ``train.py``.
            device: Target device; defaults to CUDA when available.
            strict: Passed to ``load_state_dict``; missing/unexpected keys are warned about either way.
            varlen: Variable-length attention preset from ``sage/model/swin/varlen.yaml``
                (``"off"`` = single-phase attention); ``None`` uses the default there (``tri2``).
            sdpa: Run the window attention through ``F.scaled_dot_product_attention``: same
                formula, fused kernel, faster. On by default; ``False`` runs the original attention.
            chunk_seconds: Inputs longer than this are encoded and decoded in chunks of this length
                (rounded to the 128-frame window grid, 1.49 s), so the memory stays bounded at any
                length. ``None`` always runs one pass (memory grows with the length).
            chunk_margin_seconds: Context per side of each chunk, discarded after the pass. 3 s or
                more gives the one-pass output; less leaves audible seams.
        """
        return cls(checkpoint_path, device=device, strict=strict, varlen=varlen, sdpa=sdpa,
                   chunk_seconds=chunk_seconds, chunk_margin_seconds=chunk_margin_seconds)

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
        varlen: Optional[str] = None,
        sdpa: bool = True,
        chunk_seconds: Optional[float] = DEFAULT_CHUNK_SECONDS,
        chunk_margin_seconds: float = DEFAULT_CHUNK_MARGIN_SECONDS,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        inference_cfg = ckpt.get("inference_config")
        model_config = ckpt.get("model_config")

        if inference_cfg:
            # ── Swin / custom checkpoint (has inference_config) ──────────────
            self._model_type = "swin"
            model_cfg = inference_cfg.get("model")
            if not isinstance(model_cfg, dict):
                raise ValueError("inference_config.model must be a dictionary.")
            model_cfg = upgrade_model_config(model_cfg)
            self.autoencoder = SAGEAutoencoder.from_config(model_cfg)
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
            self.sample_rate: Optional[int] = inference_cfg.get("sample_rate")
            self.audio_channels: Optional[int] = inference_cfg.get("audio_channels")
            self._stft_config = stft_cfg

        elif model_config:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path} is a stable-audio-tools checkpoint "
                "('model_config' key), not a SAGE checkpoint."
            )

        else:
            raise KeyError(
                f"Checkpoint {self.checkpoint_path} has no 'inference_config': "
                "it was not written by the SAGE training pipeline."
            )

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        # Multi-phase attention on the stages whose window covers the whole
        # training grid. Applied here because this is the single point every
        # inference path goes through, and because training never does: the
        # Lightning modules build their SAGEAutoencoder directly. Inert at or below
        # the training length, so short-segment results are bit-identical.
        self._apply_varlen(varlen)
        self._apply_sdpa(sdpa)
        self._apply_chunking(chunk_seconds, chunk_margin_seconds)

        ok(f"Loaded checkpoint: {self.checkpoint_path}", prefix="CHECKPOINT")

    def _apply_varlen(self, varlen: Optional[str]) -> None:
        """Enable the variable-length seam fix on the collapsed Swin stages.

        Args:
            varlen: Preset name from ``sage/model/swin/varlen.yaml``, or
                ``"off"`` for the original single-phase attention. ``None``
                takes the project default from that file.

        Sets ``varlen_mode`` (what ran) and ``varlen_blocks`` (how many blocks
        it touched — 0 means the architecture has no collapsed stage and the
        fix is a no-op). Non-Swin checkpoints never reach this method.
        """

        self.varlen_mode = str(load_config().mode) if varlen is None else str(varlen)
        self.varlen_blocks = enable_varlen(self.autoencoder, resolve(self.varlen_mode))
        ok(f"varlen={self.varlen_mode} on {self.varlen_blocks} collapsed blocks",
           prefix="CHECKPOINT")

    def _apply_sdpa(self, sdpa: bool) -> None:
        """Route the window attention through ``F.scaled_dot_product_attention`` (on by default).

        Sets ``sdpa`` (what ran: False when no real-valued attention exists to route) and
        ``sdpa_blocks`` (attention modules touched). See ``sage/model/swin/sdpa.py``.
        """
        self.sdpa_blocks = enable_sdpa(self.autoencoder) if sdpa else 0
        self.sdpa = self.sdpa_blocks > 0
        if self.sdpa:
            ok(f"sdpa attention on {self.sdpa_blocks} attention modules", prefix="CHECKPOINT")
        elif sdpa:
            warn("sdpa=True ignored: the model has no real-valued window attention", prefix="CHECKPOINT")

    def _apply_chunking(self, chunk_seconds: Optional[float], margin_seconds: float) -> None:
        """Set the chunk length and margin, in frames on the window grid (see ``sage/chunked.py``).

        Chunk starts stay on multiples of the coarsest attention window of encoder and decoder, so every
        chunk sees the window and shift phases of a one-pass run. A positive margin is at least one grid
        step (it would otherwise round to no context at all).

        Args:
            chunk_seconds: Chunk length; ``None`` turns chunking off (``_chunk_frames = None``).
            margin_seconds: Context per side of each chunk, discarded after the pass.
        """
        if chunk_seconds is not None and chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be > 0 or None, got {chunk_seconds}")
        if margin_seconds < 0:
            raise ValueError(f"chunk_margin_seconds must be >= 0, got {margin_seconds}")
        self.chunk_seconds, self.chunk_margin_seconds = chunk_seconds, margin_seconds
        self._chunk_frames = self._margin_frames = None
        stft = getattr(self.autoencoder, "_stft_config", None)
        if chunk_seconds is not None and (stft is None or self.sample_rate is None):
            warn("chunking off: the checkpoint has no stft_config / sample_rate", prefix="CHECKPOINT")
            self.chunk_seconds = None
        if self.chunk_seconds is None:
            return
        enc, dec = self.autoencoder.encoder, self.autoencoder.decoder
        frames_per_step = enc.downsampling_ratio[1]
        windows = [w[1] if isinstance(w, (tuple, list)) else w                # (freq, time) or one int
                   for w in (getattr(m, "window_size", None) for m in (enc, dec)) if w is not None]
        self._geometry = g = chunked.LatentGeometry(
            hop=stft.hop_length,
            frames_per_step=frames_per_step,
            bands=1 if enc.fold_freq_to_channels else enc._final_resolution[0],
            grid=math.lcm(*windows) * frames_per_step,
        )
        self._chunk_frames = max(g.frames(chunk_seconds, self.sample_rate), g.grid)
        self._margin_frames = g.frames(margin_seconds, self.sample_rate)
        if margin_seconds > 0:
            self._margin_frames = max(self._margin_frames, g.grid)

    def _chunks(self, frames: int) -> bool:
        """Whether an input of ``frames`` STFT frames runs in chunks."""
        return self._chunk_frames is not None and frames > self._chunk_frames

    def _sample(self, params: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Posterior parameters -> latent through the bottleneck (the mean when ``deterministic``)."""
        bottleneck = self.autoencoder.bottleneck
        return params if bottleneck is None else bottleneck.encode(params, deterministic=deterministic)

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def to(self, device: Union[str, torch.device]) -> "SAGE":
        self.device = self._resolve_device(device)
        self.autoencoder.to(self.device)
        return self

    @property
    def model(self) -> SAGEAutoencoder:
        return self.autoencoder

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        deterministic: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        """Encode waveform to latents.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform of shape (T,), (C, T), or (B, C, T).
        deterministic : bool
            Return posterior mean μ instead of reparameterized sample z.
            Recommended for downstream tasks (diffusion, retrieval, probing).

        Returns
        -------
        torch.Tensor
            ``(B, 16, L)`` with ``L`` = STFT frames. The last axis is NOT time: it is the
            encoder's (4 frequency bands, L / 4 time steps) token grid flattened band-major
            (index ``f * L/4 + t``). Time is the last axis of ``z.view(B, 16, 4, L // 4)``,
            one step per 4 frames (2048 samples); cut, crop or chunk latents along it.
        """
        self.autoencoder.eval()
        audio = audio.to(self.device)
        if self._chunk_frames is not None and self._chunks(audio.shape[-1] // self._geometry.hop + 1):
            if audio.dim() > 3:
                raise ValueError(f"audio must be 1D/2D/3D, got {tuple(audio.shape)}")
            return chunked.encode(
                audio.view(-1, *audio.shape[-2:]) if audio.dim() > 1 else audio.view(1, 1, -1),  # (B, C, N)
                self._geometry, self._chunk_frames, self._margin_frames,
                encode_params=lambda a: _encode_params(self.autoencoder, a),
                sample=lambda p: self._sample(p, deterministic),
            )
        return encode_audio(self.autoencoder, audio, deterministic=deterministic, debug=debug)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        target_length: Optional[int] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """Decode latents to waveform (in chunks above ``chunk_seconds``, as :meth:`encode`)."""
        self.autoencoder.eval()
        latents = latents.to(self.device)
        if self._chunk_frames is not None and self._chunks(self._geometry.frames_of(latents.shape[-1])):
            return chunked.decode(
                latents, self._geometry, self._chunk_frames, self._margin_frames,
                decode_fn=lambda z, n: decode_audio(self.autoencoder, z, target_length=n),
                target_length=target_length,
            )
        return decode_audio(self.autoencoder, latents, target_length=target_length, debug=debug)

    def pad(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Right-pad a ``(B, C, N)`` waveform to the frame grid that :meth:`encode` needs.

        Returns ``(padded, N)``. Decode with ``decode(z, target_length=padded.shape[-1])``
        and keep ``[..., :N]`` to get back the input length, as :meth:`reconstruct` does.
        """
        hop = self.autoencoder._stft_config.hop_length
        num_downsamples = len(self.autoencoder.encoder.depths) - 1
        return pad_for_swin(audio, hop, num_downsamples)

    @torch.no_grad()
    def reconstruct(self, audio: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Encode and decode a waveform of any length, exactly as the paper's evaluators do.

        The input is right-padded with :meth:`pad`, encoded (sampled ``z`` by default,
        as in the reported reconstruction metrics; ``deterministic=True`` uses the posterior
        mean), decoded, and trimmed back to the input length.

        Args:
            audio: Waveform ``(C, N)`` or ``(B, C, N)`` at ``self.sample_rate``.
        Returns:
            Reconstruction with the same shape as ``audio``.
        """
        batched = audio.dim() == 3
        x = audio if batched else audio.unsqueeze(0)
        padded, n = self.pad(x)
        z = self.encode(padded, deterministic=deterministic)
        rec = self.decode(z, target_length=padded.shape[-1])[..., :n]
        return rec if batched else rec[0]

    def __repr__(self) -> str:
        return f"SAGE(device={self.device}, sr={self.sample_rate})"
