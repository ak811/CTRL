"""Reptile (Nichol et al., 2018): a first-order meta-learning update.

    theta <- theta + epsilon * mean_i(phi_i - theta)

where ``phi_i`` are the parameters after ``k`` inner-loop updates on task ``i``.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from meta.reinforce import evaluate, reinforce_loss, run_episode


def clone_model(model: nn.Module) -> nn.Module:
    clone = deepcopy(model)
    for p in clone.parameters():
        p.requires_grad_(True)
    return clone


@torch.no_grad()
def reptile_update(meta_model: nn.Module, adapted: nn.Module | Sequence[nn.Module], step_size: float) -> None:
    """Move meta-parameters towards the mean of the adapted parameters (batched Reptile)."""
    adapted_models = [adapted] if isinstance(adapted, nn.Module) else list(adapted)
    adapted_params = [dict(m.named_parameters()) for m in adapted_models]
    for name, p in meta_model.named_parameters():
        delta = torch.stack([ap[name] - p for ap in adapted_params]).mean(0)
        p.add_(step_size * delta)


def linear_anneal(step: int, total: int, start: float, enabled: bool = True) -> float:
    return start * (1.0 - step / max(1, total)) if enabled else start


def make_optimizer(params, name: str, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr) if name == "adam" else torch.optim.SGD(params, lr=lr)


def adapt(
    policy: nn.Module,
    env: gym.Env,
    *,
    steps: int,
    episodes_per_step: int,
    lr: float,
    optimizer: str,
    gamma: float,
    entropy_coef: float,
    max_steps: int,
    device: torch.device,
    rng: np.random.Generator,
    eval_episodes: int = 0,
) -> tuple[list[float], list[float]]:
    """Run ``steps`` REINFORCE updates in place.

    Returns ``(train_returns, eval_curve)`` where ``train_returns[k]`` is the mean return of
    the episodes used for update ``k`` and, if ``eval_episodes > 0``, ``eval_curve`` holds the
    evaluation return before any update and after each update (length ``steps + 1``).
    """
    opt = make_optimizer(policy.parameters(), optimizer, lr)
    train_returns: list[float] = []
    eval_curve: list[float] = []
    if eval_episodes:
        eval_curve.append(evaluate(env, policy, device, eval_episodes, max_steps, rng))
    for _ in range(steps):
        episodes = [
            run_episode(env, policy, device, max_steps, seed=int(rng.integers(2**31)))
            for _ in range(episodes_per_step)
        ]
        loss = reinforce_loss(episodes, gamma, entropy_coef)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        train_returns.append(float(np.mean([e.total_reward for e in episodes])))
        if eval_episodes:
            eval_curve.append(evaluate(env, policy, device, eval_episodes, max_steps, rng))
    return train_returns, eval_curve
