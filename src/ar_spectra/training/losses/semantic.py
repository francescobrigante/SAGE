# ===============================================================
# semantic.py
#
#   Shared infrastructure for latent-alignment losses: the parameter-free latent
#   featurizer (`standardize_bottleneck`), the frozen CLAP teacher wrapper
#   (`CLAPTeacher`), and per-phase LossModule subclasses — VF margin loss
#   (`LatentVFLoss`), clip-level cosine distillation (`LatentCosineDistillLoss`).
# ===============================================================

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ar_spectra.utils.console import ok
from .base import LossModule


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
    Unlike MERT which produces a sequence of frames, CLAP produces a single 
    global embedding per audio clip: `(B, 512)`.
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


class LatentVFLoss(LossModule):
    """VF-Loss (VA-VAE, arXiv:2501.01423): align the latent to a frozen teacher
    per-frame (cosine) and relationally (distance-matrix).

    A learnable linear projector maps the featurized latent to the teacher channel
    dim, then two margin losses are summed (VA-VAE eq.2-3):

      - marginal cosine (per-frame):       ``ReLU(1 - m1 - cos(z', f))``
      - marginal distance-matrix (T×T):    ``ReLU(|gram(z') - gram(f)| - m2)``

    The training latent arrives flattened as ``(B, C, L)`` with ``L = F_lat·T_lat``;
    it is un-flattened to ``(B, C, F_lat, T_lat)`` using the encoder grid shape
    (``info["feature_shape"]``, token order ``f*T_lat + t``), folded C·F→channels via
    ``standardize_bottleneck``, projected, and the teacher features are avg-pooled to
    the latent time axis (LATENT_ALIGNMENT_PLAN.md §0.2).

    The overall VF weight (``w_hyper · w_adaptive``) is applied OUTSIDE, in the engine
    (VA-VAE adaptive grad-norm weighting): this module returns the RAW term sum, so
    its ``weight`` is 1.0 when adaptive weighting is on.
    """

    def __init__(self, vf_proj: nn.Module, teacher: nn.Module,
                 m1: float = 0.5, m2: float = 0.25, w_cos: float = 1.0, w_dist: float = 1.0,
                 name: str = "vf_loss", weight: float = 1.0, detach_warmup_steps: int = 0):
        super().__init__(name=name, weight=weight)
        self.vf_proj = vf_proj                 # Linear(D, proj_dim); shared with LossManager.vf_proj → opt_aux
        self.teacher = teacher                 # frozen teacher (shared)
        self.m1 = float(m1)                    # cosine margin
        self.m2 = float(m2)                    # distance-matrix margin
        self.w_cos = float(w_cos)              # weight of the per-frame cosine term
        self.w_dist = float(w_dist)            # weight of the relational distance-matrix term
        self.detach_warmup_steps = int(detach_warmup_steps)  # 0 → no warmup (VA-VAE default)

    def forward(self, info: dict) -> torch.Tensor:
        z = info["latents"]                                       # (B, C, L)  L = F_lat*T_lat
        if info.get("global_step", 0) < self.detach_warmup_steps:
            z = z.detach()                                        # train projector, gate encoder grad
        F_lat, T_lat = info["feature_shape"]                      # (freq, time) latent grid
        B, C, _ = z.shape
        z = z.reshape(B, C, int(F_lat), int(T_lat))               # (B, C, F, T) un-flatten (token = f*T+t)
        z = standardize_bottleneck(z)                             # (B, C*F, T)  fold freq→channels
        z = self.vf_proj(z.transpose(1, 2)).transpose(1, 2)       # (B, P, T)   project D→proj_dim
        f = self.teacher(info["reals"])                           # (B, 768, T_mert) frozen, no grad
        f = F.adaptive_avg_pool1d(f, z.shape[-1])                 # (B, 768, T)  match latent time axis
        # marginal cosine (per-frame), VA-VAE eq.2
        l_mcos = F.relu(1.0 - self.m1 - F.cosine_similarity(z, f, dim=1)).mean()
        # marginal distance-matrix (relational T×T), VA-VAE eq.3
        z_n = F.normalize(z, dim=1)                               # (B, P, T)
        f_n = F.normalize(f, dim=1)                               # (B, 768, T)
        z_gram = torch.einsum("bci,bcj->bij", z_n, z_n)           # (B, T, T)
        f_gram = torch.einsum("bci,bcj->bij", f_n, f_n)           # (B, T, T)
        l_mdms = F.relu((z_gram - f_gram).abs() - self.m2).mean()
        self.decay_weight()
        return self.weight * (self.w_cos * l_mcos + self.w_dist * l_mdms)


class LatentCosineDistillLoss(LossModule):
    """SALAD-style clip-level cosine distillation (arXiv:2510.07592, eq.8).

    Time-averages both the featurized latent (D=64) and frozen MERT features
    (768-d), aligns them with a single linear projection head, and minimizes
    1 - cosine_similarity — range [0, 2], bounded, no gradient explosion risk.

      L = (1 − cos(Linear(z_avg), t_avg)).mean()

    Compared with LatentVFLoss (per-frame, margin-based, VA-VAE), this operates
    at clip level → phase-robust, directly optimises MAEB's time-averaged probing
    protocol. No adaptive grad-norm weighting (SALAD does not use it).

    A detached-warmup gate (``detach_warmup_steps``) lets the projector pre-align
    before encoder gradients flow, preventing the fat D→768 matrix from satisfying
    the loss geometrically without encoding semantics (the failure mode of VF-loss).
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
            t_feat = self.teacher(info["reals"])                         # (B, 768, T_mert) or (B, 512)
            if t_feat.ndim == 3:
                t_avg = t_feat.mean(dim=-1)                              # (B, 768) MERT clip avg
            else:
                t_avg = t_feat                                           # (B, 512) CLAP clip avg
        self.decay_weight()
        return self.weight * (1.0 - F.cosine_similarity(z_proj, t_avg, dim=-1)).mean()


# ── Fase 3 — SAME §3.3.2 semantic regression (chroma + ILD) ─────────────────

_CHROMA_N_FFT: int = 8192    # paper: "high-resolution spectrogram of size 8192"
_CHROMA_HOP: int = 2048      # N=65024, center=True → T = floor(65024/2048)+1 = 32
_C0_HZ: float = 16.3516      # MIDI C0 reference frequency


class OctaveChromaTarget(nn.Module):
    """Gaussian log-frequency filterbank: STFT(n_fft=8192, hop=2048) → (B, 128, 32).

    SAME §3.3.2 "128 chroma bins" target computed from linear STFT magnitude.
    The (128, F) Gaussian filterbank is stored as a frozen buffer; no parameters.
    """

    def __init__(self, center_octave: float, octave_width: float,
                 n_bins: int = 128, sr: int = 44100,
                 n_fft: int = _CHROMA_N_FFT, hop: int = _CHROMA_HOP):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

        freqs = torch.linspace(0.0, sr / 2.0, n_fft // 2 + 1)           # (F=4097,) Hz

        log_lo = math.log2(_C0_HZ) + center_octave - octave_width / 2.0
        log_hi = math.log2(_C0_HZ) + center_octave + octave_width / 2.0
        centers = torch.linspace(log_lo, log_hi, n_bins)                 # (128,) log2-Hz
        sigma = (log_hi - log_lo) / n_bins                               # one bin width

        log_freqs = freqs.clamp(min=1e-6).log2()                         # (F,)
        fb = torch.exp(
            -0.5 * ((log_freqs.unsqueeze(0) - centers.unsqueeze(1)) / sigma) ** 2
        )                                                                  # (128, F)
        fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-8)                   # normalize rows
        self.register_buffer("filterbank", fb)                           # frozen (128, F)
        self.register_buffer("window", torch.hann_window(n_fft))         # frozen hann

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav`` ``(B, C, N)`` → ``(B, 128, 32)`` normalized chroma ∈ [0, 1]."""
        x = wav.mean(dim=1)                                               # (B, N) mono
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop,
            window=self.window, center=True, return_complex=True,
        )                                                                  # (B, 4097, 32)
        mag = stft.abs()                                                  # (B, 4097, 32)
        out = torch.einsum("nf,bft->bnt", self.filterbank, mag)           # (B, 128, 32)
        out = out / out.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)   # → [0, 1] per sample
        return out


class ILDTarget(nn.Module):
    """32-band mel ILD: log(mel_L) − log(mel_R), hop=2048, center=True → (B, 32, 32).

    Interaural Level Difference target for stereo audio. Only call for stereo
    inputs; callers must guard on ``reals.shape[1] == 2``.
    """

    def __init__(self, n_mels: int = 32, sr: int = 44100,
                 n_fft: int = 2048, hop: int = _CHROMA_HOP):
        super().__init__()
        import torchaudio  # lazy: only imported when ILDTarget is instantiated (training)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, power=1.0, center=True,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav`` ``(B, 2, N)`` → ``(B, 32, 32)`` normalized log-mel ILD ∈ [-1, 1]."""
        mel_L = self.mel(wav[:, 0, :])                                    # (B, 32, T)
        mel_R = self.mel(wav[:, 1, :])                                    # (B, 32, T)
        out = mel_L.clamp(min=1e-8).log() - mel_R.clamp(min=1e-8).log()  # (B, 32, T)
        out = out / out.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)  # → [-1, 1]
        return out


class LatentChromaILDLoss(LossModule):
    """SAME §3.3.2 semantic regression: predict octave chroma + ILD from latent.

    Three 1×1 Conv heads regress octave-band chroma (128 bins, octaves 1/5/9)
    and one head regresses 32-band log-mel ILD. Targets are fixed (filterbank
    + mel). L = Σ L1(head_i(z), y_i) over all active heads.

    Detach warmup lets heads calibrate before encoder gradients flow,
    replicating SAME's implicit reconstruction-pretrain phase.
    """

    def __init__(
        self,
        chroma_heads: nn.ModuleList,   # 3× Conv1d(D, 128, 1) — learnable
        ild_head: nn.Conv1d,           # Conv1d(D, 32, 1) — learnable
        chroma_targets: nn.ModuleList, # 3× OctaveChromaTarget — frozen buffers
        ild_target: nn.Module,         # ILDTarget — frozen
        name: str = "chroma_ild_loss",
        weight: float = 1.0,
        detach_warmup_steps: int = 25000,
    ):
        super().__init__(name=name, weight=weight)
        self.chroma_heads = chroma_heads
        self.ild_head = ild_head
        self.chroma_targets = chroma_targets
        self.ild_target = ild_target
        self.detach_warmup_steps = int(detach_warmup_steps)

    def forward(self, info: dict) -> torch.Tensor:
        z = info["latents"]                                               # (B, C, L)
        if info.get("global_step", 0) < self.detach_warmup_steps:
            z = z.detach()
        F_lat, T_lat = info["feature_shape"]
        B, C, _ = z.shape
        z = z.reshape(B, C, int(F_lat), int(T_lat))                      # (B, C, F, T)
        z = standardize_bottleneck(z)                                     # (B, D=64, T=32)
        loss = z.new_tensor(0.0)
        for head, tgt_fn in zip(self.chroma_heads, self.chroma_targets):
            y = tgt_fn(info["reals"])                                     # (B, 128, 32)
            loss = loss + F.l1_loss(head(z), y)                           # (B, 128, 32)
        if info["reals"].shape[1] == 2:                                   # stereo only
            y_ild = self.ild_target(info["reals"])                        # (B, 32, 32)
            loss = loss + F.l1_loss(self.ild_head(z), y_ild)
        self.decay_weight()
        return self.weight * loss


# ── SALAD contrastive two-view loss ───────────────────────────

class LatentContrastiveLoss(LossModule):
    """SALAD-style InfoNCE contrastive loss on the latent space (arXiv:2510.07592, eq.7).

    Takes two augmented views of the same batch (latents and augmented_latents),
    time-averages them, projects them via MLP (Linear -> SiLU -> Linear),
    and computes the symmetric cross-entropy loss over the cosine similarity matrix.
    """

    def __init__(self, contr_proj: nn.Module, tau: float = 0.1,
                 name: str = "contrastive_loss", weight: float = 1.0,
                 detach_warmup_steps: int = 20000):
        super().__init__(name=name, weight=weight)
        self.contr_proj = contr_proj          # nn.Sequential; shared with LossManager → opt_aux
        self.tau = float(tau)                  # temperature
        self.detach_warmup_steps = int(detach_warmup_steps)

    def forward(self, info: dict) -> torch.Tensor:
        z1 = info["latents"]                                          # (B, C, L)
        z2 = info.get("augmented_latents")                            # (B, C, L)
        if z2 is None:
            raise ValueError("Contrastive loss requires 'augmented_latents' in loss info.")

        if info.get("global_step", 0) < self.detach_warmup_steps:
            z1 = z1.detach()
            z2 = z2.detach()

        F_lat, T_lat = info["feature_shape"]
        B, C, _ = z1.shape

        # Un-flatten and fold frequency axis to channels
        z1 = standardize_bottleneck(z1.reshape(B, C, int(F_lat), int(T_lat))) # (B, D, T)
        z2 = standardize_bottleneck(z2.reshape(B, C, int(F_lat), int(T_lat))) # (B, D, T)

        # Average over the time dimension (semantic level, time-invariant)
        z1_avg = z1.mean(dim=-1)                                       # (B, D)
        z2_avg = z2.mean(dim=-1)                                       # (B, D)

        # Project visual/audio embeddings to projection space
        p1 = F.normalize(self.contr_proj(z1_avg), dim=-1)             # (B, P)
        p2 = F.normalize(self.contr_proj(z2_avg), dim=-1)             # (B, P)

        # Cosine similarity matrix
        logits = (p1 @ p2.t()) / self.tau                             # (B, B)
        labels = torch.arange(B, device=p1.device)                    # positive views are on diagonal

        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)

        self.decay_weight()
        return self.weight * 0.5 * (loss_a + loss_b)

