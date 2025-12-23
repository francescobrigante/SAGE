import torch
from torch import nn
from pytorch_lightning import Callback
from typing import Dict, List, Tuple
from complextorch.nn.modules.layernorm import LayerNorm as ComplexLayerNorm
from complextorch.nn.functional import inv_sqrtm2x2


def _percentiles(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach()
    if x.is_cuda:
        x = x.float().cpu()
    x = x.reshape(-1)
    if x.numel() == 0:
        return {}
    return {
        "min": x.min().item(),
        "median": x.quantile(0.5).item(),
        "p99": x.quantile(0.99).item(),
    }


def _compute_ln_stats(x: torch.Tensor, normalized_shape: Tuple[int, ...], eps: float) -> Dict[str, float]:
    axes = [-(i + 1) for i in range(len(normalized_shape))]
    x2 = torch.stack((x.real, x.imag), dim=0)
    mean = x2.mean(dim=axes, keepdim=True)
    xc = x2 - mean

    var = (xc * xc).mean(dim=axes) + eps
    v_rr = var[0]
    v_ii = var[1]
    v_ir = (xc[0] * xc[1]).mean(dim=axes)

    det = v_rr * v_ii - v_ir * v_ir

    # inverse sqrt params (same formulation as complextorch)
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, None, v_ii, symmetric=True)
    gain = torch.max(torch.stack([p.abs(), q.abs(), s.abs()], dim=0), dim=0).values

    stats = {}
    for name, tensor in ("v_rr", v_rr), ("v_ii", v_ii), ("det", det):
        pct = _percentiles(tensor)
        for k, v in pct.items():
            stats[f"{name}_{k}"] = v

    gain_cpu = gain.detach()
    if gain_cpu.is_cuda:
        gain_cpu = gain_cpu.float().cpu()
    stats["gain_p99"] = gain_cpu.quantile(0.99).item()
    stats["gain_max"] = gain_cpu.max().item()
    return stats


def _attach_hooks(model: nn.Module, max_modules: int) -> List[Tuple[str, torch.utils.hooks.RemovableHandle, nn.Module]]:
    attached: List[Tuple[str, torch.utils.hooks.RemovableHandle, nn.Module]] = []
    for name, module in model.named_modules():
        if len(attached) >= max_modules:
            break
        if isinstance(module, ComplexLayerNorm):
            def _hook(mod, inputs, output):
                setattr(mod, "_ln_last_stats", _compute_ln_stats(inputs[0], tuple(mod.normalized_shape), mod.eps))
            handle = module.register_forward_hook(_hook)
            attached.append((name, handle, module))
    return attached


class LayerNormStatsCallback(Callback):
    """Logs Complex LayerNorm whitening stats to the active logger (e.g., W&B).

    Logged keys per LayerNorm (prefix/name):
      - v_rr_min/median/p99
      - v_ii_min/median/p99
      - det_min/median/p99
      - gain_p99, gain_max
    """

    def __init__(self, every_n_steps: int = 200, max_modules: int = 8, prefix: str = "ln") -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.max_modules = max(1, int(max_modules))
        self.prefix = prefix
        self._handles: List[Tuple[str, torch.utils.hooks.RemovableHandle, nn.Module]] = []

    def setup(self, trainer, pl_module, stage=None):  # type: ignore[override]
        target = getattr(pl_module, "autoencoder", pl_module)
        self._handles = _attach_hooks(target, self.max_modules)

    def teardown(self, trainer, pl_module, stage=None):  # type: ignore[override]
        for _, h, _ in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore[override]
        step = int(getattr(trainer, "global_step", 0))
        if step % self.every_n_steps != 0:
            return
        if trainer.logger is None:
            return
        payload: Dict[str, float] = {}
        for name, _, module in self._handles:
            stats = getattr(module, "_ln_last_stats", None)
            if not stats:
                continue
            for k, v in stats.items():
                payload[f"{self.prefix}/{name}/{k}"] = float(v)
        if not payload:
            return
        try:
            # Use logger API to let Lightning handle step/commit to avoid out-of-order warnings.
            trainer.logger.log_metrics(payload, step=step)
        except Exception:
            pass
