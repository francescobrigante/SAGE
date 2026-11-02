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


# ── critic grads of the generator batches (leak with accumulation) ──────────

def test_discriminator_step_uses_only_its_own_batches():
    m = _module(accum=2, clip=0.0)
    critic = m.engine.loss_manager.discriminator
    w0 = [p.detach().clone() for p in critic.parameters()]
    batches = _batches(4)                                           # G D G D: the D window is batches 1 and 3
    _run(m, batches)
    ref = nn.Linear(4, 1)
    with torch.no_grad():
        for p, w in zip(ref.parameters(), w0):
            p.copy_(w)
    for b in (batches[1], batches[3]):
        (ref(b).pow(2).mean() / 2).backward()
    expected = [w - 0.05 * p.grad for w, p in zip(w0, ref.parameters())]
    for p, e in zip(critic.parameters(), expected):
        torch.testing.assert_close(p.detach(), e, rtol=1e-6, atol=1e-7)


def test_generator_batch_leaves_no_critic_grads():
    m = _module()
    m.training_step(_batches(1)[0], 0)                              # first batch: generator
    assert m.engine.loss_manager.discriminator.weight.grad is None
    assert m.engine.autoencoder.weight.grad is None                 # its window closed and was zeroed


# ── gradient accumulation: partial windows, epoch end (B2) ─────────────────

def _sgd_ref(module, w0, losses, lr):
    """Parameters after one SGD step on the mean of `losses(module)` from weights w0."""
    with torch.no_grad():
        for p, w in zip(module.parameters(), w0):
            p.copy_(w)
    module.zero_grad(set_to_none=True)
    losses(module).backward()
    return [w - lr * p.grad for w, p in zip(w0, module.parameters())]


def test_partial_window_at_epoch_end_is_the_mean_of_its_batches():
    m = _module(accum=3, clip=0.0, n_batches=3)                    # G D G: both windows end partial
    ae, critic = m.engine.autoencoder, m.engine.loss_manager.discriminator
    a0 = [p.detach().clone() for p in ae.parameters()]
    c0 = [p.detach().clone() for p in critic.parameters()]
    batches = _batches(3)
    _run(m, batches)
    assert m._accum_steps_gen == 0 and m._accum_steps_disc == 0    # no window left open
    ref = _FakeEngine()
    exp_c = _sgd_ref(ref.loss_manager.discriminator, c0, lambda c: c(batches[1]).pow(2).mean(), 0.05)
    gen = lambda b: ref.autoencoder(b).pow(2).mean() + ref.loss_manager.discriminator(ref.autoencoder(b)).mean()
    with torch.no_grad():
        for p, w in zip(ref.loss_manager.discriminator.parameters(), c0):
            p.copy_(w)
    exp_a = _sgd_ref(ref.autoencoder, a0, lambda _: (gen(batches[0]) + gen(batches[2])) / 2, 0.1)
    for p, e in zip(list(critic.parameters()) + list(ae.parameters()), exp_c + exp_a):
        torch.testing.assert_close(p.detach(), e, rtol=1e-6, atol=1e-7)


# ── DDP, 2 gloo processes: accumulation equals the full-batch update ────────

class _StepModel(nn.Module):
    """What Lightning runs inside DDP's forward: the module's training_step (manual optimization)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, batch, batch_idx):
        return self.m.training_step(batch, batch_idx)


def _ddp_worker(rank, world, init_file, out_dir, accum, n):
    import torch.distributed as dist
    from pytorch_lightning.overrides.distributed import prepare_for_backward
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    m = _module(accum=accum, clip=0.0, n_batches=n)
    ddp = DDP(_StepModel(m), find_unused_parameters=True)

    def backward(loss):
        prepare_for_backward(ddp, loss)                          # Lightning's DDP pre-backward
        loss.backward()

    m.manual_backward = backward
    g = torch.Generator().manual_seed(100 + rank)
    for i in range(n):
        ddp(torch.randn(3, 4, generator=g), i)                   # Lightning runs training_step in DDP's forward
    torch.save({k: v.detach().clone() for k, v in m.engine.state_dict().items()}, f"{out_dir}/{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("accum, n", [(2, 8), (2, 7), (3, 7)])
def test_ddp_accumulation_matches_the_full_batch(tmp_path, accum, n):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_worker, args=(2, str(tmp_path / "init"), str(tmp_path), accum, n), nprocs=2, join=True)
    res = [torch.load(tmp_path / f"{r}.pt") for r in (0, 1)]
    # reference: one process on the two ranks' batches concatenated (every loss is a batch mean)
    ref = _module(accum=accum, clip=0.0, n_batches=n)
    gens = [torch.Generator().manual_seed(100 + r) for r in (0, 1)]
    for i in range(n):
        ref.training_step(torch.cat([torch.randn(3, 4, generator=g) for g in gens]), i)
    for k, v in ref.engine.state_dict().items():
        torch.testing.assert_close(res[0][k], v, rtol=1e-5, atol=1e-6)
        assert torch.equal(res[0][k], res[1][k])                        # ranks agree
