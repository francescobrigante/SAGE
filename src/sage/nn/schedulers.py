
# =============================================================================
# Learning rate schedulers.
# =============================================================================

import math
import torch


class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Inverse decay LR schedule with optional exponential warmup.

    inv_gamma is the number of steps/epochs required for the learning rate to
    decay to (1 / 2)**power of its original value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of LR decay. Default: 1.
        power (float): Exponential factor of LR decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable). Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class CosineLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Step-based cosine decay with linear warmup.

    Matches Swin Transformer's CosineLRScheduler (timm) with warmup_prefix=True,
    implemented as a standard ``_LRScheduler`` for Lightning step-based compatibility.

    LR curve:
      [0, warmup_steps)       → linear ramp from ``warmup_lr_init`` to ``base_lr``
      [warmup_steps, total_steps] → cosine decay from ``base_lr`` down to ``min_lr``

    Usage in trainer.yaml::

        scheduler:
          _target_: sage.nn.schedulers.CosineLRWithWarmup
          total_steps: 200000
          warmup_steps: 10000
          min_lr: 1.0e-6
          warmup_lr_init: 1.0e-7
          interval: step       # consumed by configure_optimizers, not passed here

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of training steps (warmup + cosine).
        warmup_steps: Number of linear warmup steps. Default: 0.
        min_lr: Minimum LR at the end of cosine decay. Default: 1e-6.
        warmup_lr_init: Starting LR at step 0 during warmup. Default: 1e-7.
        last_epoch: Step index to resume from. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps      # full training length in steps
        self.warmup_steps = warmup_steps    # linear ramp duration
        self.min_lr = min_lr                # floor LR after cosine decay
        self.warmup_lr_init = warmup_lr_init  # starting LR at step 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float]:
        t = self.last_epoch
        if t < self.warmup_steps:
            alpha = t / max(1, self.warmup_steps)
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]
        t_cos = t - self.warmup_steps
        T = max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t_cos / T))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
