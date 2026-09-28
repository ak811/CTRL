"""REINFORCE with reward-to-go and batch-normalised advantages.

Bug fixed relative to the original implementation: returns were normalised with the
unbiased standard deviation of a single episode, which is NaN for one-step episodes and
poisoned the parameters. Normalisation is now done across the whole batch with
``std(unbiased=False)`` and an epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class Episode:
    log_probs: torch.Tensor  # (T,)
    entropies: torch.Tensor  # (T,)
    rewards: np.ndarray  # (T,)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    def __len__(self) -> int:
        return int(self.rewards.size)


def reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def run_episode(
    env: gym.Env,
    policy: nn.Module,
    device: torch.device,
    max_steps: int,
    seed: int | None = None,
    greedy: bool = False,
    track_grad: bool = True,
) -> Episode:
    obs, _ = env.reset(seed=seed)
    log_probs, entropies, rewards = [], [], []
    ctx = torch.enable_grad() if track_grad else torch.no_grad()
    with ctx:
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = policy.distribution(obs_t)
            action = dist.probs.argmax(-1) if greedy else dist.sample()
            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            if terminated or truncated:
                break
    return Episode(torch.stack(log_probs), torch.stack(entropies), np.asarray(rewards, dtype=np.float32))


def reinforce_loss(episodes: list[Episode], gamma: float, entropy_coef: float) -> torch.Tensor:
    device = episodes[0].log_probs.device
    returns = torch.as_tensor(
        np.concatenate([reward_to_go(e.rewards, gamma) for e in episodes]), device=device
    )
    adv = returns - returns.mean()
    adv = adv / (returns.std(unbiased=False) + 1e-8)
    log_probs = torch.cat([e.log_probs for e in episodes])
    entropies = torch.cat([e.entropies for e in episodes])
    return -(log_probs * adv.detach()).mean() - entropy_coef * entropies.mean()


def evaluate(
    env: gym.Env,
    policy: nn.Module,
    device: torch.device,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> float:
    """Mean undiscounted return of the (stochastic) policy over ``episodes`` rollouts."""
    was_training = policy.training
    policy.eval()
    returns = [
        run_episode(
            env, policy, device, max_steps, seed=int(rng.integers(2**31)), track_grad=False
        ).total_reward
        for _ in range(episodes)
    ]
    policy.train(was_training)
    return float(np.mean(returns))
