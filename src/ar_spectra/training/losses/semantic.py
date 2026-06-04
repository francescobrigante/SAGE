# ===============================================================
# semantic.py
#
#   Shared infrastructure for latent-alignment losses (LATENT_ALIGNMENT_PLAN.md
#   Fase 0): the parameter-free latent featurizer (`standardize_bottleneck`) and
#   the frozen MERT teacher wrapper. Per-phase LossModule subclasses (VF, cosine
#   distillation, chroma/ILD, contrastive, flow-matching) are added on top of this.
# ===============================================================

from __future__ import annotations

import torch
from torch import nn

from ar_spectra.utils.console import ok


def standardize_bottleneck(latents: torch.Tensor) -> torch.Tensor:
    """Fold the latent frequency axis into channels → flat ``(B, D, T)``.

    This is the tensor operation behind the MAEB ``standardize_bottleneck`` flag
    (``evaluation/maeb/swin_encoder.py``): a parameter-free, information-preserving
    relabeling that lets latent-alignment losses operate on a Swin latent the same
    way SAO's 1-D latent is consumed. Real and complex agnostic.

    Args:
        latents: Swin latent. Either ``(B, C, F, T)`` (real or complex) or an
            already-flat ``(B, D, T)``.

    Returns:
        ``(B, D, T)`` real tensor. Real: ``D = C·F``. Complex: ``D = 2·C·F``
        (Re/Im concatenated on the channel axis).
    """
    if torch.is_complex(latents):                       # (B, C, F, T) complex
        latents = torch.cat([latents.real, latents.imag], dim=1)   # (B, 2C, F, T)
    if latents.ndim == 4:                               # (B, C, F, T)
        B, C, Fl, T = latents.shape
        return latents.reshape(B, C * Fl, T)            # (B, C*F, T) parameter-free fold
    return latents                                      # already (B, D, T)


class MERTTeacher(nn.Module):
    """Frozen MERT teacher producing per-frame semantic features.

    Loads ``m-a-p/MERT-v1-95M`` (768-d, 24 kHz, ~75 Hz frame rate) offline from a
    local snapshot, averages a band of hidden layers, and returns ``(B, 768, T_mert)``
    at MERT's *native* temporal resolution. Any reconciliation with the latent's
    time axis (T_lat) is the caller's responsibility (LATENT_ALIGNMENT_PLAN.md §0.2),
    so this wrapper stays target-agnostic.
    """

    def __init__(
        self,
        model_dir: str,
        layers: tuple[int, ...] = (9, 10, 11, 12),  # which hidden states to average (music2latent)
        src_sr: int = 44100,                        # waveform sample rate fed by the dataloader
        mert_sr: int = 24000,                       # MERT's expected input sample rate
    ):
        super().__init__()
        import torchaudio
        from transformers import AutoModel

        self.layers = layers                                  # hidden-state indices to average
        # NOTE: transformers may warn that `encoder.pos_conv_embed.conv.parametrizations.
        # weight.original0/1` are "newly initialized" (checkpoint stores the old weight_g/
        # weight_v). This is a COSMETIC FALSE ALARM: torch>=2.1's weight_norm parametrization
        # has a load_state_dict compat hook that remaps weight_g/v → original0/1, so the
        # weights ARE loaded correctly (verified bit-exact in tests/test_mert_teacher.py).
        self.mert = AutoModel.from_pretrained(                # the frozen backbone
            model_dir, trust_remote_code=True,
            local_files_only=True, output_hidden_states=True,
        )
        self.mert.requires_grad_(False).eval()
        self.resample = torchaudio.transforms.Resample(src_sr, mert_sr)  # 44.1k → 24k
        n = sum(p.numel() for p in self.mert.parameters())
        ok(f"MERT teacher loaded from {model_dir} ({n/1e6:.1f}M params, frozen), "
           f"averaging hidden layers {layers}.", prefix="MERT")

    def train(self, mode: bool = True):  # keep frozen backbone in eval regardless of parent mode
        super().train(mode)
        self.mert.eval()
        return self

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav`` ``(B, C, N)`` @ src_sr → ``(B, 768, T_mert)`` @ MERT frame rate."""
        if wav.ndim == 3:                                 # (B, C, N)
            wav = wav.mean(dim=1)                          # (B, N) mono downmix
        wav = self.resample(wav)                           # (B, N') @ 24k
        hidden = self.mert(wav).hidden_states              # tuple[13] of (B, T_mert, 768)
        feat = torch.stack([hidden[i] for i in self.layers], dim=0).mean(0)  # (B, T_mert, 768)
        return feat.transpose(1, 2)                        # (B, 768, T_mert)
