from __future__ import annotations

import torch
from torch import nn


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (tokens * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


class DisabledAuxiliaryLoss:
    enabled = False
    requires_aux = False


class MatrixUnitaryOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    grad = grad.add(parameter, alpha=weight_decay)

                exp_avg.mul_(momentum).add_(grad)
                if parameter.dim() >= 2:
                    flat_update = exp_avg.view(exp_avg.shape[0], -1)
                    rank = min(flat_update.shape)
                    left, _, right = torch.svd_lowrank(flat_update, q=rank)
                    update = (left @ right.transpose(0, 1)).view_as(parameter)
                else:
                    update = exp_avg

                parameter.add_(update, alpha=-lr)

        return loss


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
    del data_config
    del model_config
    del train_config
    pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
    return MatrixUnitaryOptimizer(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )