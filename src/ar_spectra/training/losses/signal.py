# =============================================================================
# Auraloss-derived frequency-domain loss functions (modified for C-VAE).
# =============================================================================
# Copied and modified from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py under Apache License 2.0
# You can find the license at LICENSES/LICENSE_AURALOSS.txt

import torch
import numpy as np
from typing import List, Any, Optional
import scipy.signal
import librosa.filters as librosa_filters

def normalized_complex_distance_loss(x, y, eps=1e-7):
    numerator = torch.nn.functional.l1_loss(x,y, reduction = 'none').abs()
    denominator =  0.5 * (x.abs() + y.abs()) + eps  # add epsilon for numerical stability
    return numerator / denominator

def apply_reduction(losses, reduction="none", retain_batch_dim=False):
    """Apply reduction to collection of losses."""
    dim = [-1, -2] if retain_batch_dim and len(losses.shape) == 3 else None
    if reduction == "mean":
        losses = losses.mean(dim = dim)
    elif reduction == "sum":
        losses = losses.sum(dim = dim)
    return losses

def get_window(win_type: str, win_length: int):
    try:
        win = getattr(torch, win_type)(win_length)
    except (AttributeError, TypeError):
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))
    return win

class SumAndDifference(torch.nn.Module):
    def __init__(self):
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")
        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)
        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class FIRFilter(torch.nn.Module):
    """
    Psychoacoustic prefilter (hp/fd/A-weight/K-weight) as a fixed FIR, stable for AMP/DDP.
    - Fixes K-shelf gain (→ k, not k^2).
    - Proper padding derived from kernel length.
    - Kernel registered as a buffer and cast to input dtype/device at runtime.
    """
    def __init__(self, filter_type="kw", coef=0.85, fs=44100, ntaps=257, pad_mode="reflect", ref_hz=1000.0, plot=False):
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.pad_mode = pad_mode
        self.ref_hz = ref_hz
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            taps = np.zeros(2, dtype=np.float64)  # length-2 pre-emphasis [1, -a]
            taps[0] = 1.0
            taps[1] = -self.coef
        elif filter_type == "fd":
            # simple 2-sample difference y[n] = x[n] - a x[n-2]
            taps = np.zeros(3, dtype=np.float64)
            taps[0] = 1.0
            taps[2] = -self.coef
        elif filter_type in {"aw", "kw"}:
            taps = self._design_weighting_fir(filter_type)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # normalise to unity gain at ref_hz
        if filter_type in {"aw", "kw"}:
            w = 2 * np.pi * self.ref_hz / self.fs
            n = np.arange(len(taps))
            H_ref = np.abs(np.sum(taps * np.exp(-1j * w * n)))
            if H_ref > 0:
                taps = taps / H_ref

        # register as buffer, not parameter
        k = torch.from_numpy(taps.astype(np.float32))[None, None, :]
        self.register_buffer("kernel", k, persistent=False)

    def _design_weighting_fir(self, which: str) -> np.ndarray:
        fs = self.fs
        ntaps = self.ntaps

        if which == "aw":
            f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
            A1000 = 1.9997  # dB
            NUMs = [(2*np.pi*f4)**2 * 10**(A1000/20), 0, 0, 0, 0]
            DENs = np.polymul([1, 4*np.pi*f4, (2*np.pi*f4)**2],
                              [1, 4*np.pi*f1, (2*np.pi*f1)**2])
            DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi*f3]),
                              [1, 2*np.pi*f2])
        elif which == "kw":
            # Stage 1: 2nd-order HP (critical damping)
            f_hp, Q_hp = 38.135, 0.5
            w_hp = 2*np.pi*f_hp
            NUM_hp = [1, 0, 0]                  # s^2
            DEN_hp = [1, w_hp/Q_hp, w_hp**2]    # s^2 + (w/Q)s + w^2

            # Stage 2: high-shelf (→ gain k at HF, 1 at LF)
            f_shelf, Q_shelf, G_shelf = 1681.974, 1.69, 4.0
            k = 10**(G_shelf/20.0)
            w_s = 2*np.pi*f_shelf
            NUM_shelf = [k, (k*w_s)/Q_shelf, w_s**2]
            DEN_shelf = [1,    w_s /Q_shelf, w_s**2]

            NUMs = np.polymul(NUM_hp, NUM_shelf)
            DENs = np.polymul(DEN_hp, DEN_shelf)
        else:
            raise RuntimeError

        # Bilinear to digital IIR
        b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

        # Endpoint-safe grid for firwin2
        freq = np.linspace(0.0, fs/2.0, num=8193, endpoint=True)  # Hz, exact 0 and fs/2
        _, H = scipy.signal.freqz(b, a, worN=freq, fs=fs)
        Hmag = np.abs(H)

        # FIR fit
        taps = scipy.signal.firwin2(ntaps, freq, Hmag, fs=fs)
        return taps

    def forward(self, input, target=None):
        B, C, T = input.shape
        x = input.reshape(B*C, 1, T)

        # ensure kernel is on the right device/dtype
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        pad = (k.shape[-1] - 1) // 2

        if self.pad_mode in {"reflect", "replicate", "constant"}:
            mode = self.pad_mode if self.pad_mode != "constant" else "constant"
            x = torch.nn.functional.pad(x, (pad, pad), mode=mode)
            y = torch.nn.functional.conv1d(x, k, padding=0)
        else:
            y = torch.nn.functional.conv1d(x, k, padding=pad)
            
        y = y.reshape(B, C, -1)
        
        if target is not None:
            B, C, T = target.shape
            t = target.reshape(B*C, 1, T)
            if self.pad_mode in {"reflect", "replicate", "constant"}:
                mode = self.pad_mode if self.pad_mode != "constant" else "constant"
                t = torch.nn.functional.pad(t, (pad, pad), mode=mode)
                y_t = torch.nn.functional.conv1d(t, k, padding=0)
            else:
                y_t = torch.nn.functional.conv1d(t, k, padding=pad)
            y_t = y_t.reshape(B, C, -1)
            return y, y_t
        return y

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence ``‖|X|−|Y|‖_F / ‖|Y|‖_F`` with a guarded denominator.

    **L5 (STEREO_COLLAPSE_DIAGNOSIS §11.4).** Under SAO's reversed argument chain
    ``y_mag`` is the RECONSTRUCTION, so the denominator vanishes exactly when the
    prediction collapses — the case this term exists to punish. Unguarded that is
    a division by zero (inf/NaN gradients). The floor keeps the penalty steep but
    finite; for any non-degenerate pair the value is unchanged, because at
    ``eps_rel = 1e-3`` the floor only binds below a ~60 dB collapse.

    Args:
        eps_rel: denominator floor as a fraction of ``‖X‖`` — scale-free, so the
            guard behaves identically at any signal level.
        eps_abs: absolute floor, guarding the case ``‖X‖ = 0`` too.
    """

    def __init__(self, eps_rel: float = 1e-3, eps_abs: float = 1e-12):
        super().__init__()
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs

    def forward(self, x_mag, y_mag):
        num   = torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2])          # (B, ...)
        den   = torch.norm(y_mag,         p="fro", dim=[-1, -2])          # (B, ...)
        floor = torch.norm(x_mag,         p="fro", dim=[-1, -2])          # (B, ...)
        floor = (self.eps_rel * floor).clamp_min(self.eps_abs)            # (B, ...)
        return (num / torch.maximum(den, floor)).unsqueeze(-1).unsqueeze(-1)

class STFTMagnitudeLoss(torch.nn.Module):
    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()
        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        sc_eps_rel: float = 1e-3,       # L5: floor del denominatore SC, relativo a ‖X‖
        sc_eps_abs: float = 1e-12,      # L5: floor assoluto
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        retain_batch_dim: bool = False,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device
        self.retain_batch_dim = retain_batch_dim

        self.phs_used = bool(self.w_phs)
        self.spectralconv = SpectralConvergenceLoss(eps_rel=sc_eps_rel, eps_abs=sc_eps_abs)
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction if not self.retain_batch_dim else "none", distance=mag_distance, **kwargs)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction if not self.retain_batch_dim else "none", distance=mag_distance, **kwargs)

        if scale is not None:
            if self.scale == "mel":
                fb = librosa_filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)
            elif self.scale == "chroma":
                fb = librosa_filters.chroma(sr=sample_rate, n_fft=fft_size, n_chroma=n_bins)
            self.register_buffer("fb", fb)

        if self.perceptual_weighting:
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)

    def stft(self, x):
        x_stft = torch.stft(x.float(), self.fft_size, self.hop_size, self.win_length, self.window.float(), return_complex=True)
        x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps))
        x_phs = x_stft if self.phs_used else None
        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()
        if self.perceptual_weighting:
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)
            self.prefilter.to(input.device)
            input, target = self.prefilter(input, target)
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        self.window = self.window.to(input.device)
        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = normalized_complex_distance_loss(x_phs,y_phs) if self.phs_used else 0.0

        loss = (self.w_sc * sc_mag_loss) + (self.w_log_mag * log_mag_loss) + (self.w_lin_mag * lin_mag_loss) + (self.w_phs * phs_loss)
        loss = apply_reduction(loss, reduction=self.reduction, retain_batch_dim=self.retain_batch_dim)

        if self.output == "loss": return loss
        return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: List[int] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.stft_losses = torch.nn.ModuleList()
        for i, (fs, ss, wl) in enumerate(zip(fft_sizes, hop_sizes, win_lengths)):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, scale, n_bins[i] if scale == "mel" and n_bins is not None else None, perceptual_weighting, scale_invariance, **kwargs)]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
        for f in self.stft_losses:
            if f.output == "full":
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1]); log_mag_loss.append(tmp_loss[2]); lin_mag_loss.append(tmp_loss[3]); phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)
        mrstft_loss /= len(self.stft_losses)
        if f.output == "loss": return mrstft_loss
        return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """SAO-faithful stereo reconstruction loss over sum/difference + left/right.

    Reproduces Stable Audio Open's stereo recon term (auraloss
    ``SumAndDifferenceSTFTLoss`` + the two per-channel L/R terms) as a single
    module, so it registers under one ``mrstft_sd`` weight:

        loss = w_ms · ½·[ mrstft(M̂, M) + mrstft(Ŝ, S) ]        # mid/side branch
             + w_lr · ½·[ mrstft(L̂, L) + mrstft(R̂, R) ]        # left/right branch

    with ``M = L + R``, ``S = L − R`` and ``mrstft`` an A-weighted
    ``MultiResolutionSTFTLoss`` (spectral-convergence + log-magnitude L1 per
    resolution). With ``w_ms = w_lr = 1.0`` this equals SAO's
    ``1.0·sdstft + 0.5·lrstft_L + 0.5·lrstft_R`` schema character-for-character.

    The mid/side branch is what a plain complex-STFT MSE cannot see: by the
    parallelogram identity, an L2 on complex STFTs is a fixed multiple of the
    L/R L2, so M/S is a no-op there and the decoder is free to collapse the
    (low-energy) side channel. Replicating SAO's intentionally reversed
    AuralossLoss chain (``stable_audio_baseline .../losses/losses.py:111``),
    the internal STFT losses receive ``input=target, target=pred`` → spectral
    convergence normalizes by the RECONSTRUCTION, ``SC = ‖|X̂|−|X|‖/‖|X̂|‖``:
    a quasi-mono target (S≈0) yields a bounded ≈1 term (no spike, no clamp
    needed), while a collapsed predicted side is penalized hard → removes the
    shrinkage incentive that makes SAGE sound near-mono.

    Operates in the waveform domain, ``forward(decoded, reals)`` with stereo
    ``(B, 2, T)`` tensors, mirroring the ``mrmel``/``mrstft_same`` blocks.
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        sample_rate: int = 44100,
        window: str = "hann_window",
        perceptual_weighting: bool = True,  # A-weighting FIR pre-filter (SAO default)
        w_ms: float = 1.0,                  # weight of the mid/side branch (SAO: 1.0)
        w_lr: float = 1.0,                  # weight of the left/right branch (SAO: 1.0 → 0.5 each)
        w_mid: Optional[float] = None,      # L1: overrides the Mid half of the M/S branch
        w_side: Optional[float] = None,     # L1: overrides the Side half of the M/S branch
        side_gate_db: Optional[float] = None,   # L2: skip the Side term below this S/M ratio
        gate_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__()
        self.sd = SumAndDifference()            # (B,2,T) → sum (B,1,T), diff (B,1,T)
        # ONE shared A-weighted MR-STFT reused for M, S, L, R — identical config to SAO;
        # numerically equivalent to SAO's separate sdstft/lrstft instances (same params).
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            sample_rate=sample_rate,
            perceptual_weighting=perceptual_weighting,
            **kwargs,
        )
        self.w_ms = w_ms                        # mid/side branch weight (legacy knob)
        self.w_lr = w_lr                        # left/right branch weight
        # L1 — the M/S branch is `w_ms · ½(mid + side)`, so each half defaults to
        # `w_ms/2`: leaving w_mid/w_side unset reproduces SAO bit-for-bit, while
        # setting them addresses Mid and Side independently. Side-only is
        # `w_lr=0, w_mid=0, w_side=1` — the only configuration whose gradient on
        # the Mid is exactly zero (antisymmetric on L/R, cancels in M=L+R), hence
        # the only one that cannot move FAD/CLAP/CDPAM, which see the Mid alone.
        self.w_mid = 0.5 * w_ms if w_mid is None else w_mid
        self.w_side = 0.5 * w_ms if w_side is None else w_side
        # L2 — a target whose Side sits this far below its Mid carries no stereo
        # information (mono duplicated to 2 channels: 7.0 % of the training
        # corpus exactly, 10.4 % below −40 dB). There the log-magnitude term
        # actively teaches |Ŝ| → 0, i.e. it teaches the collapse. Skip, don't
        # clamp: a clamped term still has a gradient pointing the wrong way.
        self.side_gate_db = side_gate_db
        self.gate_eps = gate_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        m_p, s_p = self.sd(pred)                                       # (B, 1, T) each
        m_t, s_t = self.sd(target)                                     # (B, 1, T) each
        # SAO reversed-order chain (AuralossLoss "wrong order"): STFTLoss gets
        # input=target, target=pred → SC normalizes by the RECONSTRUCTION.
        total = pred.new_zeros(())                                     # scalar accumulator
        if self.w_mid:
            total = total + self.w_mid * self.mrstft(m_t, m_p)         # mid term
        if self.w_side:
            total = total + self.w_side * self._side_term(s_t, s_p, m_t)
        if self.w_lr:
            lr = 0.5 * (self.mrstft(target[:, 0:1], pred[:, 0:1]) +     # left  channel
                        self.mrstft(target[:, 1:2], pred[:, 1:2]))      # right channel
            total = total + self.w_lr * lr
        return total                                                   # scalar

    def _side_term(self, s_t: torch.Tensor, s_p: torch.Tensor,
                   m_t: torch.Tensor) -> torch.Tensor:
        """Side MR-STFT, restricted to the items whose target actually has a Side.

        L2: items below ``side_gate_db`` are dropped from the batch rather than
        down-weighted, so the term is the mean over the items that carry stereo
        information — its magnitude stays comparable instead of being diluted by
        the mono ones.

        Args:
            s_t: target side, ``(B, 1, T)``.
            s_p: predicted side, ``(B, 1, T)``.
            m_t: target mid, ``(B, 1, T)`` — the reference the gate is relative to.

        Returns:
            Scalar loss; exactly ``0`` (still attached to the graph) when every
            item in the batch is gated out.
        """
        if self.side_gate_db is None:
            return self.mrstft(s_t, s_p)

        e_s = s_t.pow(2).sum(dim=(-2, -1))                             # (B,)
        e_m = m_t.pow(2).sum(dim=(-2, -1))                             # (B,)
        ratio_db = 10.0 * torch.log10(e_s / e_m.clamp_min(self.gate_eps)
                                      + self.gate_eps)                 # (B,)
        keep = ratio_db >= self.side_gate_db                           # (B,) bool

        if not bool(keep.any()):
            return (s_p.sum() * 0.0)          # graph-connected zero, no gradient
        if bool(keep.all()):
            return self.mrstft(s_t, s_p)
        return self.mrstft(s_t[keep], s_p[keep])                       # (B', 1, T)


class SISDRLoss(torch.nn.Module):
    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target
        losses = 10 * torch.log10((target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses

class MelSTFTLoss(STFTLoss):
    def __init__(self, sample_rate, fft_size=1024, hop_size=256, win_length=1024, window="hann_window", w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0, w_phs=0.0, n_mels=128, **kwargs):
        super(MelSTFTLoss, self).__init__(fft_size, hop_size, win_length, window, w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, "mel", n_mels, **kwargs)


class StereoCoherenceLoss(torch.nn.Module):
    r"""**L6** — inter-channel coherence, the differentiable surrogate of ``d_pan``.

    Targets the normalised Mid/Side cross-correlation per TF bin::

        γ = 2·M·S* / (|M|² + |S|²)          γ ∈ ℂ,  |γ| ≤ 1

    A single complex number that carries both perceptually-validated metrics
    (STEREO_COLLAPSE_DIAGNOSIS §11.3), verified numerically:

    * ``Re(γ) = p``, i.e. **exactly** the pan the metric measures (err 3.6e-07).
    * ``|γ| = 2r/(1+r²)`` with ``r = |S|/|M|`` — monotone in the width.
    * ``arg(γ)`` is the Mid↔Side relative phase.

    Why this and not a magnitude term. ``mrstft_sd`` is spectral-convergence +
    log-magnitude: both on ``|X|``. Rotating ``arg(S)`` by 90° at fixed magnitude
    moves ``d_width`` by 4.8e-09 (numerically nothing) and ``d_pan`` by 0.715 —
    a magnitude loss is provably blind to what pan measures, so it restores the
    Side's energy with whatever phase is cheapest. Measured consequence: the M/S
    arms fix the level (−0.08 dB) and leave ``d_pan`` pinned at the mono null.

    Why not the existing ``w_phs``. ``normalized_complex_distance_loss`` divides
    per bin by ``½(|x|+|y|)``, so a near-silent bin — where phase is noise — gets
    full weight. Here the denominator is ``|M|²+|S|²``: γ goes smoothly to 0 as
    the Side vanishes instead of blowing up, and the energy weight silences those
    bins a second time. Where the reference is mono (γ≈0) a large γ̂ is penalised,
    so the term also punishes *hallucinated* Side — the ``ms_replace`` failure.

    Gradient sensitivity. Near collapse the width is quadratic in ``r`` while
    ``|γ|`` is linear: their slopes differ by ``1/r``, so at the measured
    baseline (−12.4 dB → r = 0.24) this term carries ~4× the gradient of a
    width-based one, exactly in the regime where the model is stuck.

    ``pred``'s Mid is **detached**, so ∂L/∂M̂ is exactly zero and the gradient on
    L and R is antisymmetric — it cancels bit-for-bit in the M = L+R projection.
    That is what makes the term safe to graft onto a half-trained checkpoint:
    FAD/CLAP/CDPAM see the Mid alone (§4.4) and cannot be moved by it.
    """

    def __init__(
        self,
        fft_sizes: List[int] = (2048, 1024, 512),
        hop_sizes: List[int] = (512, 256, 128),
        win_lengths: List[int] = (2048, 1024, 512),
        window: str = "hann_window",
        detach_mid: bool = True,        # keeps ∂L/∂Mid exactly zero — see class docstring
        side_gate_db: Optional[float] = None,   # L2: skip items whose target Side is negligible
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        self.detach_mid = detach_mid
        self.side_gate_db = side_gate_db
        self.eps = eps
        for i, wl in enumerate(self.win_lengths):
            self.register_buffer(f"win_{i}", get_window(window, wl).float(), persistent=False)

    def _gamma(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """γ = 2·M·S*/(|M|²+|S|²) for complex STFTs, shape-preserving."""
        return 2.0 * m * s.conj() / (m.abs().square() + s.abs().square() + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        m_p = (pred[:, 0] + pred[:, 1]) / 2 ** 0.5                     # (B, T)
        s_p = (pred[:, 0] - pred[:, 1]) / 2 ** 0.5                     # (B, T)
        m_t = (target[:, 0] + target[:, 1]) / 2 ** 0.5                 # (B, T)
        s_t = (target[:, 0] - target[:, 1]) / 2 ** 0.5                 # (B, T)

        if self.side_gate_db is not None:                              # L2, item-level
            ratio_db = 10.0 * torch.log10(
                s_t.square().sum(-1) / m_t.square().sum(-1).clamp_min(self.eps) + self.eps)
            keep = ratio_db >= self.side_gate_db                       # (B,)
            if not bool(keep.any()):
                return pred.sum() * 0.0                                # graph-connected zero
            if not bool(keep.all()):
                m_p, s_p, m_t, s_t = m_p[keep], s_p[keep], m_t[keep], s_t[keep]

        total = pred.sum() * 0.0                                       # scalar accumulator
        for i, (n_fft, hop, wl) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            win = getattr(self, f"win_{i}").to(pred.device, pred.dtype)
            stft = lambda x: torch.stft(x, n_fft, hop, wl, window=win,
                                        return_complex=True, center=True)
            Mp, Sp = stft(m_p), stft(s_p)                              # (B, F, N) complex
            Mt, St = stft(m_t), stft(s_t)                              # (B, F, N) complex
            if self.detach_mid:
                Mp = Mp.detach()

            d = self._gamma(Mp, Sp) - self._gamma(Mt, St)              # (B, F, N) complex
            mod = (d.real.square() + d.imag.square() + self.eps).sqrt()  # (B, F, N), |·| smooth at 0
            w = Mt.abs().square() + St.abs().square()                  # (B, F, N) reference energy
            total = total + (w * mod).sum() / (w.sum() + self.eps)     # same weighting as d_pan

        return total / len(self.fft_sizes)                             # scalar
