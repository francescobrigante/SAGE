from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


DEFAULT_MUON_INCLUDE_PATTERNS = (
    ".attn.qkv.linear.weight",
    ".attn.proj.linear.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
    ".mlp.fc1.linear.weight",
    ".mlp.fc2.linear.weight",
)


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    """Approximate the zeroth power of a 2-D update matrix with Newton-Schulz."""
    if grad.ndim != 2:
        raise ValueError(f"Muon Newton-Schulz expects a 2-D tensor, got shape {tuple(grad.shape)}")
    if not torch.is_floating_point(grad):
        raise TypeError(f"Muon Newton-Schulz expects a real floating tensor, got {grad.dtype}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    original_dtype = grad.dtype
    x = grad.float()
    if x.norm() <= eps:
        return torch.zeros_like(grad)

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    x = x / x.norm().clamp_min(eps)

    # Coefficients used by the Muon optimizer implementation in modded-nanogpt.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def _iter_named_parameters(
    *,
    model: nn.Module | None,
    named_parameters: Iterable[tuple[str, nn.Parameter]] | None,
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]] | None,
) -> list[tuple[str, nn.Parameter]]:
    if model is not None:
        if named_parameters is not None or params is not None:
            raise ValueError("Pass only one of model, named_parameters, or params to MuonAdamW.")
        return [(name, param) for name, param in model.named_parameters()]

    if named_parameters is not None:
        if params is not None:
            raise ValueError("Pass only one of named_parameters or params to MuonAdamW.")
        return list(named_parameters)

    if params is None:
        raise ValueError("MuonAdamW requires model, named_parameters, or params.")

    named: list[tuple[str, nn.Parameter]] = []
    index = 0
    for item in params:
        if isinstance(item, dict):
            group_params = item.get("params", [])
            for param in group_params:
                named.append((f"param_{index}", param))
                index += 1
        else:
            named.append((f"param_{index}", item))
            index += 1
    return named


def _is_muon_candidate(name: str, param: nn.Parameter, include_patterns: tuple[str, ...]) -> bool:
    if not param.requires_grad:
        return False
    if param.is_complex() or not torch.is_floating_point(param):
        return False
    if param.ndim != 2:
        return False
    return any(pattern in f".{name}" for pattern in include_patterns)


class MuonAdamW(Optimizer):
    """Conservative Muon wrapper with AdamW fallback groups.

    Muon is applied only to explicitly selected 2-D transformer-block matrices.
    Everything else is updated with AdamW, including biases, normalization,
    bottleneck, patching, CPB, and projection/head parameters.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]] | None = None,
        *,
        model: nn.Module | None = None,
        named_parameters: Iterable[tuple[str, nn.Parameter]] | None = None,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        betas: tuple[float, float] | list[float] = (0.9, 0.98),
        eps: float = 1.0e-8,
        muon_lr: float | None = None,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float | None = None,
        adamw_betas: tuple[float, float] | list[float] | None = None,
        adamw_eps: float | None = None,
        matrix_lr_scale: bool = True,
        weight_decay_exclude_1d: bool = False,
        include_patterns: Iterable[str] | None = None,
        log_param_groups: bool = False,
        log_param_coverage: bool | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= muon_momentum < 1.0:
            raise ValueError(f"Invalid muon_momentum: {muon_momentum}")
        if len(betas) != 2:
            raise ValueError("betas must contain exactly two values")
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid AdamW betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid AdamW eps: {eps}")

        include = tuple(include_patterns) if include_patterns is not None else DEFAULT_MUON_INCLUDE_PATTERNS
        named = _iter_named_parameters(model=model, named_parameters=named_parameters, params=params)

        muon_params: list[nn.Parameter] = []
        adamw_decay_params: list[nn.Parameter] = []
        adamw_no_decay_params: list[nn.Parameter] = []
        self.muon_parameter_names: list[str] = []
        self.adamw_parameter_names: list[str] = []
        self.adamw_decay_parameter_names: list[str] = []
        self.adamw_no_decay_parameter_names: list[str] = []
        seen: set[int] = set()

        for name, param in named:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if _is_muon_candidate(name, param, include):
                muon_params.append(param)
                self.muon_parameter_names.append(name)
            else:
                self.adamw_parameter_names.append(name)
                local_name = name.rsplit(".", 1)[-1]
                is_complex_ddp_1d = param.ndim == 2 and local_name.endswith("_real_view")
                if weight_decay_exclude_1d and (param.ndim == 1 or name.endswith(".bias") or is_complex_ddp_1d):
                    adamw_no_decay_params.append(param)
                    self.adamw_no_decay_parameter_names.append(name)
                else:
                    adamw_decay_params.append(param)
                    self.adamw_decay_parameter_names.append(name)

        resolved_muon_lr = float(lr if muon_lr is None else muon_lr)
        resolved_adamw_lr = float(lr if adamw_lr is None else adamw_lr)
        resolved_adamw_betas = tuple(float(v) for v in (adamw_betas if adamw_betas is not None else betas))
        if len(resolved_adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly two values")
        resolved_adamw_eps = float(eps if adamw_eps is None else adamw_eps)

        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append(
                {
                    "params": muon_params,
                    "lr": resolved_muon_lr,
                    "weight_decay": float(weight_decay),
                    "use_muon": True,
                    "muon_momentum": float(muon_momentum),
                    "muon_nesterov": bool(muon_nesterov),
                    "ns_steps": int(ns_steps),
                    "matrix_lr_scale": bool(matrix_lr_scale),
                }
            )
        if adamw_decay_params:
            param_groups.append(
                {
                    "params": adamw_decay_params,
                    "lr": resolved_adamw_lr,
                    "weight_decay": float(weight_decay),
                    "use_muon": False,
                    "betas": resolved_adamw_betas,
                    "eps": resolved_adamw_eps,
                }
            )
        if adamw_no_decay_params:
            param_groups.append(
                {
                    "params": adamw_no_decay_params,
                    "lr": resolved_adamw_lr,
                    "weight_decay": 0.0,
                    "use_muon": False,
                    "betas": resolved_adamw_betas,
                    "eps": resolved_adamw_eps,
                }
            )
        if not param_groups:
            raise ValueError("MuonAdamW received no trainable parameters.")

        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "use_muon": False,
            "muon_momentum": float(muon_momentum),
            "muon_nesterov": bool(muon_nesterov),
            "ns_steps": int(ns_steps),
            "matrix_lr_scale": bool(matrix_lr_scale),
            "betas": resolved_adamw_betas,
            "eps": resolved_adamw_eps,
        }
        super().__init__(param_groups, defaults)

        if log_param_coverage is not None:
            log_param_groups = bool(log_param_coverage)
        if log_param_groups:
            print(
                "MuonAdamW parameter groups: "
                f"muon={len(self.muon_parameter_names)} "
                f"adamw_decay={len(self.adamw_decay_parameter_names)} "
                f"adamw_no_decay={len(self.adamw_no_decay_parameter_names)}"
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    @torch.no_grad()
    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["muon_momentum"]
        nesterov = group["muon_nesterov"]
        ns_steps = group["ns_steps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients.")
            if grad.is_complex() or grad.ndim != 2:
                raise RuntimeError("Muon parameter group received a non-real or non-2-D gradient.")

            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)

            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = grad.detach().clone()
            else:
                state["momentum_buffer"].mul_(momentum).add_(grad)
            update = state["momentum_buffer"]
            if nesterov:
                update = grad.add(update, alpha=momentum)

            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            if group["matrix_lr_scale"]:
                update_scale = math.sqrt(max(1.0, param.shape[0] / param.shape[1]))
            else:
                update_scale = 1.0
            param.add_(update, alpha=-lr * update_scale)

    @torch.no_grad()
    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW AdamW fallback does not support sparse gradients.")
            if grad.is_complex():
                raise RuntimeError("MuonAdamW AdamW fallback does not support complex gradients.")

            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)
