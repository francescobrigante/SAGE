import typing as tp
import torch
from torch import nn
from torch.nn import functional as F

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, decay = 1.0):
        super().__init__()

        self.name = name
        self.decay = float(decay)
        self.master_weight = weight
        weight = torch.tensor(float(weight))
        self.register_buffer('weight', weight)

    def update_weight(self, new_weight: float) -> None:
        """Set weight and master_weight so decay_weight() does not undo the change."""
        self.weight.fill_(new_weight)
        self.master_weight = new_weight

    def decay_weight(self):
        if self.decay != 1.0:
            self.weight *= self.decay
        elif self.decay == 1.0 and self.weight != self.master_weight:
            self.weight = torch.tensor(self.master_weight, dtype=self.weight.dtype, device=self.weight.device)

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError

class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0, decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key = key

    def forward(self, info):
        self.decay_weight()
        return self.weight * info[self.key]

class TargetValueLoss(LossModule):
    def __init__(self, key: str, target: float, name: str, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
        self.target = target
    
    def forward(self, info):
        self.decay_weight()
        return self.weight * (info[self.key] - self.target).abs()

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b
        self.mask_key = mask_key

    def forward(self, info):
        l1_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info:
            l1_loss = l1_loss[info[self.mask_key]]

        l1_loss = l1_loss.mean()
        self.decay_weight()
        return self.weight * l1_loss

class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss',decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b
        self.mask_key = mask_key

    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()
        self.decay_weight()
        return self.weight * mse_loss

class LossWithTarget(LossModule):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(name, weight, decay)

        self.loss_module = loss_module
        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.loss_module(info[self.input_key], info[self.target_key])
        self.decay_weight()
        return self.weight * loss

class SelfLoss(LossModule):
    """Loss that operates on a single input tensor (self-supervised, no target needed)."""

    def __init__(self, loss_module: nn.Module, input_key: str, name: str, weight: float = 1.0, decay: float = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)
        self.loss_module = loss_module  # the underlying nn.Module (forward takes one arg)
        self.input_key = input_key      # key into the info dict

    def forward(self, info):
        loss = self.loss_module(info[self.input_key])
        self.decay_weight()
        return self.weight * loss


class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0
        losses = {}
        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss
        return total_loss, losses
