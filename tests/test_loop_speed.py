# ===============
# Speed keys of the training loop around the critic, all off by default:
# - log_every_n_batches (sums kept on the GPU, one host copy per log flush) logs the same values as the
#   per-optimizer-step logs of tests/test_training_step.py, without a host sync per value;
# - detach_step_output: training_step returns the loss without its graph;
# - DDP (2 gloo processes on CPU): the skip-idle-buckets hook, gradient_as_bucket_view and
#   broadcast_buffers=False give the same parameters and optimizer state bit for bit as plain DDP,
#   on a G/D model laid out like SAGE (critic-only, autoencoder-only and mixed buckets).
# ===============
import types
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from sage.training.ddp_hooks import IdleBucketState, build_ddp_strategy, skip_idle_buckets_hook
from sage.training.engine import frozen_parameters, select_training_phase
from sage.training.lightning_module import SAGELightningModule

CONFIGS = Path(__file__).parents[1] / "configs"


# ── configs ──────────────────────────────────────────────────────────────────

def test_trainer_defaults_are_todays_loop():
    from omegaconf import OmegaConf
    t = OmegaConf.load(CONFIGS / "trainer.yaml").trainer
    assert t.log_every_n_batches == 0 and t.progress_bar_refresh_rate == 1 and t.lr_monitor is True
    assert t.detach_step_output is False
    assert t.ddp_skip_idle_buckets is False and t.ddp_gradient_as_bucket_view is False
    assert t.ddp_broadcast_buffers is True


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


def _module(log_on_device=False, every=1, detach=False, accum=1, n_batches=1000):
    m = SAGELightningModule.__new__(SAGELightningModule)
    pl.LightningModule.__init__(m)
    m.automatic_optimization = False
    m.engine = _FakeEngine()
    for k, v in dict(accumulate_grad_batches=accum, clip_grad_norm=1.0, log_on_device=log_on_device,
                     detach_step_output=detach, _accum_steps_gen=0, _accum_steps_disc=0, _disc_phase=True,
                     gen_step=0, _step_phase="gen").items():
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
    """n training steps; returns what each one returned and the _step_phase it left."""
    g = torch.Generator().manual_seed(seed)
    returned, phases = [], []
    for i in range(n):
        returned.append(m.training_step(torch.randn(3, 4, generator=g), i))
        phases.append(m._step_phase)
        m._trainer._logger_connector.batch += 1
    return returned, phases


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


def test_detach_step_output():
    returned, _ = _run(_module(detach=True), 4)
    assert all(t.grad_fn is None and not t.requires_grad for t in returned)
    returned, _ = _run(_module(), 2)
    assert all(t.grad_fn is not None for t in returned)


def test_step_phase_is_the_phase_of_the_batch():
    m = _module()
    _, phases = _run(m, 4)
    assert phases == ["gen", "disc", "gen", "disc"] == [p for p, _ in m.engine.seen]


# ── DDP strategy ─────────────────────────────────────────────────────────────

def test_build_ddp_strategy():
    m = _module()
    assert build_ddp_strategy("ddp_find_unused_parameters_true", m) == "ddp_find_unused_parameters_true"
    s = build_ddp_strategy("ddp_find_unused_parameters_true", m, skip_idle_buckets=True,
                           gradient_as_bucket_view=True, broadcast_buffers=False, freeze_disc_in_gen_step=True)
    assert s._ddp_kwargs == dict(find_unused_parameters=True, gradient_as_bucket_view=True, broadcast_buffers=False)
    assert s._ddp_comm_hook is skip_idle_buckets_hook and s._ddp_comm_state.module is m
    with pytest.raises(ValueError, match="freeze_disc_in_gen_step"):
        build_ddp_strategy("ddp_find_unused_parameters_true", m, skip_idle_buckets=True)
    with pytest.raises(ValueError, match="find_unused_parameters_true"):
        build_ddp_strategy("ddp", m, skip_idle_buckets=True, freeze_disc_in_gen_step=True)
    with pytest.raises(ValueError, match="DDP strategy"):
        build_ddp_strategy("auto", m, broadcast_buffers=False)


# ── DDP exactness, 2 gloo processes ──────────────────────────────────────────

class _GD(nn.Module):
    """Autoencoder, critic and aux head registered in SAGE's order. With 1 KiB buckets DDP lays them out
    as [aux], [critic + aux], [critic], [AE + critic], [AE], [AE] (launch order): every bucket kind."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.autoencoder = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 32))
        self.critic = nn.Sequential(nn.Linear(32, 4), nn.LeakyReLU(0.2), nn.Linear(4, 64), nn.LeakyReLU(0.2),
                                    nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
        self.aux = nn.Linear(32, 4)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 32))
        self._step_phase = "gen"

    @property
    def discriminator(self):
        return self.critic

    def forward(self, x, phase, ddp, opts, detach):
        """Lightning's manual-optimisation step inside DDP's forward: losses, pre-backward, step."""
        from pytorch_lightning.overrides.distributed import prepare_for_backward
        self._step_phase = phase
        x = x * self.scale
        if phase == "gen":
            fake = self.autoencoder(x)
            with frozen_parameters(self.critic):
                adv = F.softplus(-self.critic(fake)).mean()
            loss = (fake - x).pow(2).mean() + 0.1 * adv + 0.01 * self.aux(fake).pow(2).mean()
            params, steps = self.autoencoder.parameters(), (opts["gen"], opts["aux"])
        else:
            with torch.no_grad():
                fake = self.autoencoder(x)
            loss = F.softplus(self.critic(fake)).mean() + F.softplus(-self.critic(x)).mean()
            params, steps = self.critic.parameters(), (opts["disc"],)
        prepare_for_backward(ddp, loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        for opt in steps:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss.detach() if detach else loss


DDP_RUNS = {
    "plain": dict(),
    "hook": dict(hook=True),
    "loop_fast": dict(hook=True, gradient_as_bucket_view=True, broadcast_buffers=False, detach=True),
    "no_allreduce": dict(noop=True),         # control: without the all-reduce the result must differ
}


def _ddp_worker(rank, world, init_file, out_dir):
    from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    for name, run in DDP_RUNS.items():
        model = _GD()
        ddp = DDP(model, find_unused_parameters=True, bucket_cap_mb=1 / 1024,
                  gradient_as_bucket_view=run.get("gradient_as_bucket_view", False),
                  broadcast_buffers=run.get("broadcast_buffers", True))
        state = IdleBucketState(model)
        if run.get("hook"):
            ddp.register_comm_hook(state, skip_idle_buckets_hook)
        elif run.get("noop"):
            ddp.register_comm_hook(None, noop_hook)
        opts = {"gen": torch.optim.AdamW(model.autoencoder.parameters(), lr=1e-2),
                "aux": torch.optim.AdamW(model.aux.parameters(), lr=1e-2),
                "disc": torch.optim.AdamW(model.critic.parameters(), lr=1e-2, betas=(0.8, 0.99))}
        g = torch.Generator().manual_seed(100 + rank)
        losses = []
        for step in range(8):
            phase = "gen" if step % 2 == 0 else "disc"
            losses.append(ddp(torch.randn(16, 32, generator=g), phase, ddp, opts, run.get("detach", False)).item())
        torch.save({"params": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "opt": {k: o.state_dict() for k, o in opts.items()}, "losses": losses,
                    "idle": dict(state._idle)}, Path(out_dir) / f"{name}_{rank}.pt")
        del ddp
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def ddp_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("ddp")
    mp.spawn(_ddp_worker, args=(2, str(out / "init"), str(out)), nprocs=2, join=True)
    return {(name, r): torch.load(out / f"{name}_{r}.pt", weights_only=False) for name in DDP_RUNS for r in (0, 1)}


def _same(a, b):
    return all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"])


def _same_opt(a, b):
    for k in a["opt"]:
        sa, sb = a["opt"][k]["state"], b["opt"][k]["state"]
        if sa.keys() != sb.keys():
            return False
        for i in sa:
            if not all(torch.equal(torch.as_tensor(sa[i][f]), torch.as_tensor(sb[i][f])) for f in sa[i]):
                return False
    return True


def test_ddp_layout_has_every_bucket_kind(ddp_results):
    idle = ddp_results[("hook", 0)]["idle"]
    kinds = {frozenset(p) for p in idle.values()}
    assert {frozenset({"gen"}), frozenset({"disc"}), frozenset()} <= kinds   # critic-, AE/aux-only, mixed


@pytest.mark.parametrize("run", ["hook", "loop_fast"])
def test_ddp_keys_are_bit_exact(ddp_results, run):
    for r in (0, 1):
        assert _same(ddp_results[("plain", r)], ddp_results[(run, r)])
        assert _same_opt(ddp_results[("plain", r)], ddp_results[(run, r)])
        assert ddp_results[("plain", r)]["losses"] == ddp_results[(run, r)]["losses"]
    assert _same(ddp_results[(run, 0)], ddp_results[(run, 1)])            # ranks agree


def test_ddp_control_without_allreduce_differs(ddp_results):
    assert not _same(ddp_results[("plain", 0)], ddp_results[("no_allreduce", 0)])
    assert not _same(ddp_results[("no_allreduce", 0)], ddp_results[("no_allreduce", 1)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
