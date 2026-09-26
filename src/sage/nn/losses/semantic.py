# ===============================================================
# semantic.py
#
#   Semantic distillation of the latent (paper §3): the parameter-free latent
#   featurizer (`standardize_bottleneck`), the frozen LAION-CLAP teacher
#   (`CLAPTeacher`) and the clip-level cosine distillation loss (`LatentCosineDistillLoss`).
# ===============================================================

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sage.utils.console import ok
from sage.nn.losses.base import LossModule


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


class CLAPTeacher(nn.Module):
    """Frozen LAION-CLAP teacher producing per-clip semantic audio features.

    Loads the LAION-CLAP Music checkpoint (512-d, 48 kHz).
    CLAP produces a single global embedding per audio clip: `(B, 512)`.
    """

    def __init__(
        self,
        model_dir: str,
        src_sr: int = 44100,                        # waveform sample rate fed by the dataloader
    ):
        super().__init__()
        import torchaudio
        import laion_clap

        # Initialize the CLAP module (HTSAT base architecture, without feature fusion for the music checkpoint)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        
        # Load the downloaded checkpoint from $FAST
        self.clap.load_ckpt(model_dir)
        self.clap.requires_grad_(False).eval()
        
        # CLAP natively operates at 48000 Hz
        self.resample = torchaudio.transforms.Resample(src_sr, 48000)
        n = sum(p.numel() for p in self.clap.parameters())
        ok(f"LAION-CLAP teacher loaded from {model_dir} ({n/1e6:.1f}M params, frozen).", prefix="CLAP")

    def train(self, mode: bool = True):
        super().train(mode)
        self.clap.eval()
        return self

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav`` ``(B, C, N)`` @ src_sr → ``(B, 512)`` global clip embedding."""
        if wav.ndim == 3:                                 # (B, C, N)
            wav = wav.mean(dim=1)                          # (B, N) mono downmix
        wav = self.resample(wav)                           # (B, N') @ 48k
        
        # laion_clap expects float32 tensors.
        # get_audio_embedding_from_data expects `use_tensor=True` to avoid numpy conversions
        feat = self.clap.get_audio_embedding_from_data(x=wav, use_tensor=True)  # (B, 512)
        return feat


class LatentCosineDistillLoss(LossModule):
    """SALAD-style clip-level cosine distillation (arXiv:2510.07592, eq.8).

    Time-averages the featurized latent (D=64), maps it to the teacher space with a
    single linear projection head, and minimizes 1 - cosine_similarity against the
    frozen teacher embedding (CLAP: one 512-d vector per clip; a frame sequence
    ``(B, C, T)`` is time-averaged first). Range [0, 2], bounded.

      L = (1 − cos(Linear(z_avg), t_avg)).mean()

    Operating at clip level makes the loss phase-robust and matches MAEB's
    time-averaged probing protocol. No adaptive grad-norm weighting (SALAD does not use it).

    A detached-warmup gate (``detach_warmup_steps``) lets the projector pre-align
    before encoder gradients flow, so the projection cannot satisfy the loss
    geometrically without the latent encoding semantics.
    """

    def __init__(self, distill_proj: nn.Module, teacher: nn.Module,
                 name: str = "distill_loss", weight: float = 1.0,
                 detach_warmup_steps: int = 25000):
        super().__init__(name=name, weight=weight)
        self.distill_proj = distill_proj      # Linear(latent_dim, proj_dim); shared with LossManager → opt_aux
        self.teacher = teacher                # frozen teacher (no grad via @torch.no_grad() + requires_grad=False)
        self.detach_warmup_steps = int(detach_warmup_steps)

    def forward(self, info: dict) -> torch.Tensor:
        z = info["latents"]                                           # (B, C, L)  L = F_lat·T_lat
        if info.get("global_step", 0) < self.detach_warmup_steps:
            z = z.detach()                                            # train projector only; gate encoder grad
        F_lat, T_lat = info["feature_shape"]                          # (freq, time) latent grid
        B, C, _ = z.shape
        z = z.reshape(B, C, int(F_lat), int(T_lat))                   # (B, C, F, T)  un-flatten
        z = standardize_bottleneck(z)                                 # (B, D, T)    D = C·F = 64
        z_avg = z.mean(dim=-1)                                        # (B, D)       time pool  ← SALAD P_L step 1
        z_proj = self.distill_proj(z_avg)                             # (B, proj_dim)           ← SALAD P_L step 2
        with torch.no_grad():
            # mean() must be inside no_grad: teacher may return a requires_grad leaf tensor
            t_feat = self.teacher(info["reals"])                         # (B, 512) CLAP, or (B, C, T) frames
            if t_feat.ndim == 3:
                t_avg = t_feat.mean(dim=-1)                              # (B, C) clip avg
            else:
                t_avg = t_feat                                           # (B, 512) CLAP clip embedding
        self.decay_weight()
        return self.weight * (1.0 - F.cosine_similarity(z_proj, t_avg, dim=-1)).mean()
