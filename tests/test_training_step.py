# ===============
# SAGELightningModule.training_step (manual optimization, per-batch G/D alternation) on a fake
# engine of linear layers: the train/* logs of an optimizer step are the means of the batches of
# its own window.
# ===============
import types

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

from sage.training.engine import select_training_phase
from sage.training.lightning_module import SAGELightningModule


class _FakeEngine(nn.Module):
    """compute() of the AutoencoderEngine on linear layers: same output dict, losses from parameters.
    The generator loss goes through the critic (adversarial term) and through the aux head."""

    def __init__(self, aux=False):
        super().__init__()
        torch.manual_seed(0)
        self.autoencoder = nn.Linear(4, 4)
        self.loss_manager = nn.Module()
        self.loss_manager.discriminator = nn.Linear(4, 1)
        self.loss_manager.aux = nn.Linear(4, 4) if aux else None
        self.loss_manager.aux_parameters = lambda: list(self.loss_manager.aux.parameters()) if aux else []
        self.seen = []                                   # (phase, {name: value}) of every batch

    def compute(self, batch, gen_step, disc_phase):
        phase = select_training_phase(True, "adv", disc_phase, True)
        out = {"phase": phase, "gen_total": None, "gen_breakdown": {}, "disc_total": None, "disc_breakdown": {},
               "loss_info": {}, "diag": {}}
        critic = self.loss_manager.discriminator
        if phase == "gen":
            y = self.autoencoder(batch)
            adv = critic(y).mean()
            out["gen_total"] = y.pow(2).mean() + adv
            if self.loss_manager.aux is not None:
                out["gen_total"] = out["gen_total"] + self.loss_manager.aux(y.detach()).pow(2).mean()
            out["gen_breakdown"] = {"a_loss": y.abs().mean(), "adv_loss": adv}
            out["stats"] = {"data_std": batch.std(), "latent_std": y.std()}
            vals = {"loss": out["gen_total"], "latent_std": y.std(), "data_std": batch.std(), **out["gen_breakdown"]}
        else:
            out["disc_total"] = critic(batch).pow(2).mean()
            out["disc_breakdown"] = {"discriminator_loss": out["disc_total"]}
            out["stats"] = {}
            vals = dict(out["disc_breakdown"])
        self.seen.append((phase, {f"train/{k}": v.detach().item() for k, v in vals.items()}))
        return out


def _module(accum=1, aux=False, clip=1.0, n_batches=1000):
    m = SAGELightningModule.__new__(SAGELightningModule)
    pl.LightningModule.__init__(m)
    m.automatic_optimization = False
    m.engine = _FakeEngine(aux=aux)
    for k, v in dict(accumulate_grad_batches=accum, clip_grad_norm=clip, _accum_steps_gen=0, _accum_steps_disc=0,
                     _disc_phase=True, gen_step=0, _log_accum_gen={}, _log_accum_disc={}, _log_accum_gen_count=0,
                     _log_accum_disc_count=0).items():
        setattr(m, k, v)
    opts = [torch.optim.SGD(m.engine.autoencoder.parameters(), lr=0.1),
            torch.optim.SGD(m.engine.loss_manager.discriminator.parameters(), lr=0.05)]
    if aux:
        opts.append(torch.optim.SGD(m.engine.loss_manager.aux_parameters(), lr=0.1))
    m.optimizers = lambda: opts
    m.lr_schedulers = lambda: []
    m.manual_backward = lambda loss: loss.backward()
    m.logged = []
    m.log_dict = lambda d, **kw: m.logged.append({k: float(v) for k, v in d.items()})
    m._trainer = types.SimpleNamespace(num_training_batches=n_batches, global_rank=0)
    return m


def _batches(n, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3, 4, generator=g) for _ in range(n)]


def _run(m, batches):
    for i, b in enumerate(batches):
        m.training_step(b, i)


# ── logging (B1) ─────────────────────────────────────────────────────────────

def test_logs_the_batch_that_closes_the_window():
    m = _module()
    _run(m, _batches(6))
    assert len(m.logged) == 6                                       # accum 1: one log per optimizer step
    for (phase, vals), log in zip(m.engine.seen, m.logged):
        assert {k: log[k] for k in vals} == pytest.approx(vals, rel=1e-7)   # this batch, not the previous one
        assert ("train/gen_lr" in log) == (phase == "gen") and ("train/disc_lr" in log) == (phase == "disc")


def test_logs_the_window_mean_with_accumulation():
    m = _module(accum=2)
    _run(m, _batches(8))
    assert len(m.logged) == 4                                       # windows of 2 G and of 2 D batches
    seen = {p: [v for q, v in m.engine.seen if q == p] for p in ("gen", "disc")}
    for log in m.logged:
        phase = "gen" if "train/gen_lr" in log else "disc"
        a, b = seen[phase].pop(0), seen[phase].pop(0)
        assert {k: log[k] for k in a} == pytest.approx({k: (a[k] + b[k]) / 2 for k in a}, rel=1e-6)
