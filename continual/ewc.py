"""Elastic Weight Consolidation for sequential sine regression.

Fisher information: for a Gaussian observation model ``y ~ N(f_theta(x), 1)`` the Fisher
is ``E_x[J(x)^T J(x)]`` with ``J = df/dtheta``; we use its diagonal, computed with
per-sample Jacobians via ``torch.func.vmap``. This is the *true* Fisher (no labels
needed), avoiding the empirical-Fisher bias of squaring mini-batch loss gradients.

Modes
  * ``separate`` - one quadratic penalty per past task (Kirkpatrick et al., 2017)
  * ``online``   - a single running Fisher/anchor with decay ``gamma`` (Schwarz et al., 2018)

Bug fixed relative to the original implementation: only the most recent task's Fisher
was kept, so every earlier task lost its protection as soon as a new one was consolidated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from continual.config import ContinualConfig
from continual.data import SineTask
from continual.models import SineMLP

METHODS = ("finetune", "ewc")


def fisher_diagonal(model: SineMLP, x: torch.Tensor, head: int = 0) -> dict[str, torch.Tensor]:
    params = {n: p.detach() for n, p in model.named_parameters()}

    def output(p, xi):
        return functional_call(model, p, (xi.unsqueeze(0),), {"head": head}).squeeze()

    per_sample = vmap(grad(output), in_dims=(None, 0))(params, x)
    return {n: g.pow(2).mean(0) for n, g in per_sample.items()}


@dataclass
class EWCRegularizer:
    lam: float
    mode: str = "separate"
    gamma: float = 1.0
    anchors: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = field(default_factory=list)

    def consolidate(self, model: nn.Module, fisher: dict[str, torch.Tensor]) -> None:
        anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
        if self.mode == "online" and self.anchors:
            _, prev_f = self.anchors[0]
            fisher = {n: self.gamma * prev_f[n] + fisher[n] for n in fisher}
            self.anchors = [(anchor, fisher)]
        elif self.mode == "online":
            self.anchors = [(anchor, fisher)]
        else:
            self.anchors.append((anchor, fisher))

    def penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.zeros((), device=next(model.parameters()).device)
        for anchor, fisher in self.anchors:
            for n, p in model.named_parameters():
                total = total + (fisher[n] * (p - anchor[n]).pow(2)).sum()
        return 0.5 * self.lam * total


# Backward-compatible alias for the original class name.
EWC = EWCRegularizer


def run_sequence(cfg: ContinualConfig, method: str, seed: int, device) -> dict[str, np.ndarray]:
    """Train on ``cfg.task_params`` in order. Returns the accuracy matrix ``R`` where
    ``R[i, j]`` is the test MSE on task ``j`` after training on task ``i`` (NaN for
    unseen heads in the task-incremental scenario) and the full training-loss trace."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tasks = [SineTask(a, p) for a, p in cfg.task_params]
    n = len(tasks)
    multi_head = cfg.cl_scenario == "task"
    model = SineMLP(
        hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers, n_heads=n if multi_head else 1
    ).to(device)
    reg = EWCRegularizer(cfg.ewc_lambda, cfg.ewc_mode, cfg.ewc_online_gamma) if method == "ewc" else None
    test_sets = [t.grid(cfg.eval_points, device, cfg.x_range) for t in tasks]

    R = np.full((n, n), np.nan, dtype=np.float64)
    trace = np.empty(n * cfg.cl_steps_per_task, dtype=np.float32)
    for i, task in enumerate(tasks):
        head = i if multi_head else 0
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for step in range(cfg.cl_steps_per_task):
            x, y = task.sample(cfg.batch_size, rng, device, cfg.x_range)
            task_loss = F.mse_loss(model(x, head=head), y)
            loss = task_loss + (reg.penalty(model) if reg is not None and reg.anchors else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            trace[i * cfg.cl_steps_per_task + step] = float(task_loss.item())

        if reg is not None:
            xf, _ = task.sample(cfg.ewc_fisher_samples, rng, device, cfg.x_range)
            reg.consolidate(model, fisher_diagonal(model, xf, head=head))

        with torch.no_grad():
            for j, (xt, yt) in enumerate(test_sets):
                if multi_head and j > i:
                    continue  # head j is untrained
                R[i, j] = float(F.mse_loss(model(xt, head=j if multi_head else 0), yt))
    return {"R": R, "trace": trace}
