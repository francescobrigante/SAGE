# ===============
# Speed keys of the training loop around the critic, all off by default:
# - log_every_n_batches (sums kept on the GPU, one host copy per log flush) logs the same values as the
#   per-optimizer-step logs of tests/test_training_step.py, without a host sync per value.
# ===============
import types
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

from sage.training.engine import select_training_phase
from sage.training.lightning_module import SAGELightningModule

CONFIGS = Path(__file__).parents[1] / "configs"


# ── configs ──────────────────────────────────────────────────────────────────

def test_trainer_defaults_are_todays_loop():
    from omegaconf import OmegaConf
    t = OmegaConf.load(CONFIGS / "trainer.yaml").trainer
    assert t.log_every_n_batches == 0 and t.progress_bar_refresh_rate == 1 and t.lr_monitor is True


# ── training_step logging ────────────────────────────────────────────────────

class _FakeEngine(nn.Module):
    """compute() of the AutoencoderEngine on two linear layers: same output dict, losses from parameters."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.autoencoder = nn.Linear(4, 4)
        self.loss_manager = nn.Module()
        self.loss_manager.discriminator = nn.Linear(4, 1)
        self.loss_manager.aux_parameters = lambda: []
        self.seen = []                                   # (phase, {name: value}) of every batch
        self.record = True                               # False: no .item() of its own (sync counting)

    def compute(self, batch, gen_step, disc_phase):
        phase = select_training_phase(True, "adv", disc_phase, True)
        out = {"phase": phase, "gen_total": None, "gen_breakdown": {}, "disc_total": None, "disc_breakdown": {},
               "loss_info": {}, "diag": {}}
        if phase == "gen":
            y = self.autoencoder(batch)
            out["gen_total"] = y.pow(2).mean()
            out["gen_breakdown"] = {"a_loss": y.abs().mean(), "b_loss": 2 * out["gen_total"]}
            out["stats"] = {"data_std": batch.std(), "latent_std": y.std()}
            vals = {"loss": out["gen_total"], "latent_std": y.std(), "data_std": batch.std(), **out["gen_breakdown"]}
        else:
            out["disc_total"] = self.loss_manager.discriminator(batch).pow(2).mean()
            out["disc_breakdown"] = {"discriminator_loss": out["disc_total"]}
            out["stats"] = {}
            vals = dict(out["disc_breakdown"])
        if self.record:
            self.seen.append((phase, {f"train/{k}": v.detach().item() for k, v in vals.items()}))
        return out


class _Connector:
    """Lightning's logger connector: flushes the step metrics every `every` batches."""

    def __init__(self, every):
        self.every, self.batch = every, 0

    @property
    def should_update_logs(self):
        return (self.batch + 1) % self.every == 0


def _module(log_on_device=False, every=1, accum=1, n_batches=1000):
    m = SAGELightningModule.__new__(SAGELightningModule)
    pl.LightningModule.__init__(m)
    m.automatic_optimization = False
    m.engine = _FakeEngine()
    for k, v in dict(accumulate_grad_batches=accum, clip_grad_norm=1.0, log_on_device=log_on_device,
                     _accum_steps_gen=0, _accum_steps_disc=0, _disc_phase=True, gen_step=0).items():
        setattr(m, k, v)
    m._log_accum, m._log_count = {"gen": {}, "disc": {}}, {"gen": 0, "disc": 0}
    opts = [torch.optim.SGD(m.engine.autoencoder.parameters(), lr=0.1),
            torch.optim.SGD(m.engine.loss_manager.discriminator.parameters(), lr=0.05)]
    m.optimizers = lambda: opts
    m.lr_schedulers = lambda: []
    m.manual_backward = lambda loss: loss.backward()
    m.logged = []
    m.log_dict = lambda d, **kw: m.logged.append({k: float(v) for k, v in d.items()})
    m._trainer = types.SimpleNamespace(num_training_batches=n_batches, global_rank=0,
                                       _logger_connector=_Connector(every))
    return m


def _run(m, n, seed=5):
    """n training steps; returns what each one returned."""
    g = torch.Generator().manual_seed(seed)
    returned = []
    for i in range(n):
        returned.append(m.training_step(torch.randn(3, 4, generator=g), i))
        m._trainer._logger_connector.batch += 1
    return returned


def test_log_on_device_every_batch_logs_the_same_values():
    ref, dev = _module(), _module(log_on_device=True)
    _run(ref, 6), _run(dev, 6)
    assert len(dev.logged) == 6
    f32 = lambda x: torch.tensor(x, dtype=torch.float32)            # what Lightning stores
    for a, b in zip(ref.logged, dev.logged):
        assert all(torch.equal(f32(b[k]), f32(a[k])) for k in a)    # the same numbers
    for p, q in zip(ref.engine.parameters(), dev.engine.parameters()):
        assert torch.equal(p, q)


def test_log_on_device_window_means_without_item(monkeypatch):
    ref = _module()
    _run(ref, 8)
    m = _module(log_on_device=True, every=4)
    calls = []
    real_item = torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "item", lambda self: calls.append(1) or real_item(self))
    m.engine.record = False                                         # the fake's own bookkeeping uses .item()
    _run(m, 8)
    monkeypatch.undo()
    assert calls == [] and len(m.logged) == 2                       # no .item(); one log per 4 batches
    for w, log in enumerate(m.logged):
        window = ref.engine.seen[4 * w: 4 * w + 4]
        for key in ("train/loss", "train/a_loss", "train/discriminator_loss"):
            vals = [v[key] for _, v in window if key in v]
            assert log[key] == pytest.approx(sum(vals) / len(vals), rel=1e-6)
        assert {"train/gen_lr", "train/disc_lr"} <= set(log)


def test_log_on_device_with_accumulation_and_epoch_end():
    """accumulate_grad_batches = 2 over a 7-batch epoch (the last windows are partial): same updates as
    the per-step logs, and every batch is in one flushed window."""
    ref, dev = _module(accum=2, n_batches=7), _module(log_on_device=True, every=3, accum=2, n_batches=7)
    _run(ref, 7), _run(dev, 7)
    for p, q in zip(ref.engine.parameters(), dev.engine.parameters()):
        assert torch.equal(p, q)
    assert len(dev.logged) == 2 and dev._log_count == {"gen": 1, "disc": 0}   # batches 0-2, 3-5; 6 still open
    seen = ref.engine.seen
    for w, log in enumerate(dev.logged):
        window = seen[3 * w: 3 * w + 3]
        vals = [v["train/loss"] for p, v in window if p == "gen"]
        assert log["train/loss"] == pytest.approx(sum(vals) / len(vals), rel=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
