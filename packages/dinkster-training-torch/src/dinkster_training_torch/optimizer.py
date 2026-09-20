"""Memory-reduced optimizer option for LoRA masters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class FactoredAdamW(torch.optim.Optimizer):
    """AdamW with a factored second moment for matrix-shaped parameters.

    LoRA masters are matrices, so the second moment costs rows plus columns
    instead of one value per parameter. The first moment stays full-sized.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError("betas must be between zero and one")
        if eps <= 0 or weight_decay < 0:
            raise ValueError("eps must be positive and weight_decay non-negative")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def,override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("FactoredAdamW does not support sparse gradients")
                if parameter.ndim < 2:
                    raise RuntimeError("FactoredAdamW parameters must have at least two dimensions")
                grad = gradient.float()
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq_row"] = torch.zeros(
                        parameter.shape[0], device=parameter.device, dtype=torch.float32
                    )
                    state["exp_avg_sq_col"] = torch.zeros(
                        math.prod(parameter.shape[1:]),
                        device=parameter.device,
                        dtype=torch.float32,
                    )
                state["step"] += 1
                exp_avg = state["exp_avg"]
                row = state["exp_avg_sq_row"]
                col = state["exp_avg_sq_col"]
                flat_grad_sq = grad.reshape(parameter.shape[0], -1).square()
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                row.mul_(beta2).add_(flat_grad_sq.mean(dim=1), alpha=1.0 - beta2)
                col.mul_(beta2).add_(flat_grad_sq.mean(dim=0), alpha=1.0 - beta2)
                factored = row[:, None] * col[None, :] / row.mean().clamp_min(eps)
                denominator = factored.sqrt().add_(eps).reshape(parameter.shape)
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                update = exp_avg / bias1 / (denominator / math.sqrt(bias2))
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
        return loss


def factored_state_elements(shapes: Iterable[tuple[int, ...]]) -> int:
    """Number of float32 moment values allocated after the first step."""
    total = 0
    for shape in shapes:
        total += math.prod(shape)
        total += shape[0] + math.prod(shape[1:])
    return total
