# =============================================================================
# DDP for the alternating G/D training: a communication hook that skips the all-reduce of the
# buckets holding only the idle model of the step, and the DDPStrategy that carries it.
#
# The LightningModule is ONE DistributedDataParallel with find_unused_parameters=True, so every
# backward all-reduces every bucket: in a discriminator step the autoencoder's ~415 MiB of zero
# gradients (its parameters are unused), in a generator step the critic's. The AE buckets come after
# the critic's in launch order and are never rebuilt (find_unused_parameters=True), so in the
# discriminator step they go after the critic backward has finished, on the critical path.
#
# The parameters of the idle model are unused on every rank, and DDP leaves the .grad of a globally
# unused parameter untouched whatever its bucket holds: skipping the all-reduce of a bucket that
# holds only such parameters changes nothing. The other buckets go through the same operations as
# DDP's built-in path (scale by 1/world_size, then SUM all-reduce), so the update is bit-identical.
# Precondition: freeze_disc_in_gen_step, so the critic has no gradient in the generator step.
# =============================================================================
from typing import Dict, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from sage.utils.console import ok

DDP_STRATEGIES = {"ddp": False, "ddp_find_unused_parameters_false": False, "ddp_find_unused_parameters_true": True}


class IdleBucketState:
    """State of skip_idle_buckets_hook: the module (its `_step_phase` and its critic) and, per bucket
    index, the phases in which the bucket is idle. DDP never rebuilds its buckets with
    find_unused_parameters=True, so a bucket index keeps the same parameters for the whole run."""

    def __init__(self, module: nn.Module, process_group=None):
        self.module = module                       # SAGELightningModule: _step_phase + discriminator
        self.process_group = process_group         # None = the default group
        self._idle: Dict[int, Set[str]] = {}       # bucket index -> phases in which it is skipped
        self._mib: Dict[int, float] = {}           # bucket index -> size (MiB), for the console report
        self._steps: Dict[str, int] = {}           # steps seen per phase (counted at bucket 0)

    def idle_phases(self, bucket) -> Set[str]:
        index = bucket.index()
        if index not in self._idle:
            critic = {id(p) for p in self.module.discriminator.parameters()}
            is_critic = [id(p) in critic for p in bucket.parameters()]
            phases = set()
            if all(is_critic):
                phases.add("gen")                  # critic only: frozen in the generator step
            if not any(is_critic):
                phases.add("disc")                 # autoencoder / aux head only: no grad in the critic step
            self._idle[index] = phases
            self._mib[index] = bucket.buffer().numel() * bucket.buffer().element_size() / 2**20
        return self._idle[index]

    def report(self, phase: str, bucket) -> None:
        """One console line per phase, at the start of its second step (every bucket classified by then)."""
        if bucket.index() != 0:
            return
        self._steps[phase] = self._steps.get(phase, 0) + 1
        if self._steps[phase] != 2 or (dist.is_initialized() and dist.get_rank() != 0):
            return
        idle = sorted(i for i, p in self._idle.items() if phase in p)
        ok(f"skip-idle hook, {phase} step: {len(idle)}/{len(self._idle)} buckets not all-reduced "
           f"({sum(self._mib[i] for i in idle):.1f} of {sum(self._mib.values()):.1f} MiB)", prefix="DDP")


def _allreduce_mean(bucket: dist.GradBucket, process_group) -> torch.futures.Future[torch.Tensor]:
    """DDP's built-in bucket reduction: the gradients scaled by 1/world_size (the reducer multiplies
    by the reciprocal while copying them into the bucket), then a SUM all-reduce."""
    group = process_group if process_group is not None else dist.group.WORLD
    buffer = bucket.buffer()
    buffer.mul_(1.0 / dist.get_world_size(group))
    fut = dist.all_reduce(buffer, group=group, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0])


def skip_idle_buckets_hook(state: IdleBucketState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """DDP communication hook: the bucket as it is when it holds only the idle model of the step
    (its parameters are globally unused, DDP leaves their .grad untouched), else the mean all-reduce."""
    phase = getattr(state.module, "_step_phase", "gen")
    skipped = phase in state.idle_phases(bucket)
    state.report(phase, bucket)
    if skipped:
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())
        return fut
    return _allreduce_mean(bucket, state.process_group)


def build_ddp_strategy(name: str, module: nn.Module, skip_idle_buckets: bool = False,
                       gradient_as_bucket_view: bool = False, broadcast_buffers: bool = True,
                       freeze_disc_in_gen_step: bool = False):
    """The Trainer strategy: `name` unchanged when every DDP key has its default, else a DDPStrategy
    with the same find_unused_parameters and the requested DDP arguments / communication hook.
    Lightning registers communication hooks only on CUDA devices (DDPStrategy._register_ddp_hooks)."""
    if not (skip_idle_buckets or gradient_as_bucket_view or not broadcast_buffers):
        return name
    if name not in DDP_STRATEGIES:
        raise ValueError(f"ddp_* keys need a DDP strategy ({sorted(DDP_STRATEGIES)}), got strategy={name!r}")
    from pytorch_lightning.strategies import DDPStrategy
    kwargs = dict(find_unused_parameters=DDP_STRATEGIES[name], gradient_as_bucket_view=gradient_as_bucket_view,
                  broadcast_buffers=broadcast_buffers)
    if skip_idle_buckets:
        if not freeze_disc_in_gen_step:
            raise ValueError("ddp_skip_idle_buckets=true needs freeze_disc_in_gen_step=true: without it the critic "
                             "gets gradients in the generator step, and skipping their all-reduce would change them")
        if not DDP_STRATEGIES[name]:
            raise ValueError("ddp_skip_idle_buckets=true needs strategy=ddp_find_unused_parameters_true: the idle "
                             "model's parameters are unused in every step")
        if getattr(module, "discriminator", None) is None:
            raise ValueError("ddp_skip_idle_buckets=true needs a discriminator (no idle model without G/D steps)")
        kwargs.update(ddp_comm_state=IdleBucketState(module), ddp_comm_hook=skip_idle_buckets_hook)
    return DDPStrategy(**kwargs)
