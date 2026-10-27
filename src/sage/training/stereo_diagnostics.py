# =============================================================================
# Diagnostics of the stereo critics (PLAN_WAVTOKENIZER_STEREO M2): does a WavTokenizer branch tell a
# real clip from the same clip with a foreign or quieter Side, and does it score mono clips apart;
# and how large is the adversarial share of the generator gradient, on mono and stereo examples.
# Read-only: no parameter gradient, no RNG, so a run with the diagnostics on trains identically.
# =============================================================================
from typing import Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from sage.training.loggers import log_metric
from sage.utils.console import info, warn

MONO_SIDE_DB = -40.0          # an example whose Side sits this far below its Mid counts as mono
SIDE_QUIET_GAIN = 0.25        # Side -12 dB
ADV_LOSS_NAMES = ("loss_adv", "feature_matching_loss")   # names of the weighted L_adv, L_fm in the breakdown
EPS = 1e-12


def side_to_mid_db(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, T) → (B,) energy ratio 10·log10(E_S / E_M)."""
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return 10 * torch.log10((s.square().mean(-1) + EPS) / (m.square().mean(-1) + EPS))


def with_side(x: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Rebuild L, R from the Mid of x and a new Side: (B, 2, T), (B, T) → (B, 2, T)."""
    m = (x[:, 0] + x[:, 1]) / 2
    return torch.stack([m + side, m - side], dim=1)


def foreign_side(x: torch.Tensor) -> torch.Tensor:
    """Each stereo example gets the Side of another stereo example, rescaled to its own Side energy.

    Mono examples keep their (null) Side, and no stereo example borrows one: a borrowed null Side
    would turn the swap into a mono test. With fewer than two stereo examples, x is returned as is.
    """
    s = (x[:, 0] - x[:, 1]) / 2                                       # (B, T)
    stereo = torch.nonzero(side_to_mid_db(x) >= MONO_SIDE_DB).flatten()
    if len(stereo) < 2:
        return x
    other = s.clone()
    other[stereo] = s[stereo.roll(1)]                                 # donors among the stereo examples
    gain = (s.square().mean(-1, keepdim=True) / (other.square().mean(-1, keepdim=True) + EPS)).sqrt()
    return with_side(x, gain * other)


def group_scores(disc, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Mean logit of each branch (mpd, mrd, dac) per example: {group: (B,)}.

    Works on the WavTokenizerDiscriminator: folded branches have 2 channel rows per
    example and block ([L,R] then [M,S] with fold_lrms), joint branches one row.
    """
    fmaps = disc(x)
    batch, out, i = x.shape[0], {}, 0
    modes = {"mpd": disc.mpd_stereo, "mrd": disc.mrd_stereo, "dac": disc.dac_stereo}
    for group, n in disc.group_sizes.items():
        rows_per_block = disc.channels if modes[group] == "fold" else 1
        scores = []
        for fm in fmaps[i:i + n]:
            per_row = fm[-1].flatten(1).mean(1)                          # (rows,)
            scores.append(per_row.view(-1, batch, rows_per_block).mean(dim=(0, 2)))   # (B,)
        if scores:
            out[group] = torch.stack(scores).mean(0)
        i += n
    return out


@torch.no_grad()
def critic_stereo_report(disc, x: torch.Tensor) -> Dict[str, float]:
    """Gaps D(real) - D(variant) per branch; positive = the branch prefers the real clip."""
    real = group_scores(disc, x)
    foreign = group_scores(disc, foreign_side(x))
    quiet = group_scores(disc, with_side(x, SIDE_QUIET_GAIN * (x[:, 0] - x[:, 1]) / 2))
    mono = side_to_mid_db(x) < MONO_SIDE_DB
    report = {"n_mono": float(mono.sum())}
    for g, r in real.items():
        report[f"{g}/real"] = r.mean().item()
        report[f"{g}/foreign_side_gap"] = (r - foreign[g]).mean().item()
        report[f"{g}/quiet_side_gap"] = (r - quiet[g]).mean().item()
        if mono.any() and (~mono).any():
            report[f"{g}/mono_minus_stereo"] = (r[mono].mean() - r[~mono].mean()).item()
    return report


def adversarial_gradient_share(gen_total: torch.Tensor, breakdown: Dict[str, torch.Tensor],
                               decoder_out: torch.Tensor, reals: torch.Tensor) -> Dict[str, float]:
    """‖∂(L_adv + L_fm)/∂ŝ‖ / ‖∂(rest)/∂ŝ‖ per example at the decoder output ŝ, without touching .grad.

    Every reconstruction loss depends on the decoder output (the waveform ones through the iSTFT),
    so the ratio compares the two parts of the generator update where they meet.
    """
    adv = sum(breakdown[k] for k in ADV_LOSS_NAMES if k in breakdown)
    g_adv, = torch.autograd.grad(adv, decoder_out, retain_graph=True, allow_unused=True)
    g_rest, = torch.autograd.grad(gen_total - adv, decoder_out, retain_graph=True, allow_unused=True)
    if g_adv is None or g_rest is None:
        return {}
    ratio = g_adv.flatten(1).norm(dim=1) / (g_rest.flatten(1).norm(dim=1) + EPS)   # (B,)
    out = {"adv_grad_share": ratio.mean().item()}
    if reals.dim() == 3 and reals.shape[1] == 2:
        mono = side_to_mid_db(reals) < MONO_SIDE_DB
        if mono.any():
            out["adv_grad_share_mono"] = ratio[mono].mean().item()
        if (~mono).any():
            out["adv_grad_share_stereo"] = ratio[~mono].mean().item()
    return out


class StereoCriticDiagnostics(Callback):
    """Every `every_n_epochs`, the critic_stereo_report of the first validation batch, on rank 0.

    The validation set is fixed, so the same clips are scored at every report.
    """

    def __init__(self, every_n_epochs: int, max_examples: int = 16):
        super().__init__()
        self.every_n_epochs = every_n_epochs  # report cadence in epochs
        self.max_examples = max_examples      # clips of the first validation batch that are scored
        self._batch: Optional[torch.Tensor] = None

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch,
                                batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0 and self._batch is None:
            self._batch = batch[1][: self.max_examples].detach()       # (B, 2, T) reference waveforms

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or trainer.global_rank != 0 or self._batch is None:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        gan = pl_module.discriminator
        disc = getattr(gan, "discriminator", None)
        if disc is None or not hasattr(disc, "group_sizes"):
            warn("StereoCriticDiagnostics: not a WavTokenizer discriminator — skipped", prefix="DIAG")
            return
        report = critic_stereo_report(disc, self._batch.to(pl_module.device))
        for key, value in report.items():
            log_metric(trainer.logger, f"critic/{key}", value)
        info(f"epoch {trainer.current_epoch}: " + " ".join(f"{k}={v:.4g}" for k, v in report.items()), prefix="CRITIC")
