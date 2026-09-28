"""Task distributions p(T) for meta-reinforcement learning.

Each environment family exposes a train/test split over task parameters so that
meta-learned initialisations are evaluated on *held-out* tasks:

* **snake** - action-semantics permutations (18 train / 6 test of the 24 possible)
  and a randomised food reward.
* **puckworld** - action permutations (same split) plus randomised thrust and damping.
* **pong** - ALE game ``mode`` x ``difficulty``; difficulty 3 is held out.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from envs.registry import make_env

_ALL_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
TEST_PERMUTATIONS: tuple[tuple[int, ...], ...] = _ALL_PERMUTATIONS[3::4]
TRAIN_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(
    p for p in _ALL_PERMUTATIONS if p not in TEST_PERMUTATIONS
)
SPLITS = ("train", "test")


@dataclass(frozen=True)
class TaskSpec:
    """A concrete task: an environment key plus constructor parameters."""

    env_key: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        parts = []
        for k, v in sorted(self.params.items()):
            if isinstance(v, float):
                v = f"{v:.3g}"
            elif isinstance(v, (tuple, list)):
                v = "".join(str(x) for x in v)
            parts.append(f"{k}={v}")
        return f"{self.env_key}[{','.join(parts)}]"


class TaskDistribution:
    """Base class. Subclasses implement :meth:`sample`."""

    env_key: str = ""

    def __init__(self, **env_kwargs: Any) -> None:
        self.env_kwargs = env_kwargs

    def sample(self, rng: np.random.Generator, split: str = "train") -> TaskSpec:
        raise NotImplementedError

    def make(self, task: TaskSpec) -> gym.Env:
        return make_env(task.env_key, **{**self.env_kwargs, **task.params})

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def _pick(rng: np.random.Generator, options):
    return options[int(rng.integers(len(options)))]


class SnakeTasks(TaskDistribution):
    env_key = "snake"

    def sample(self, rng: np.random.Generator, split: str = "train") -> TaskSpec:
        self._check_split(split)
        perms = TRAIN_PERMUTATIONS if split == "train" else TEST_PERMUTATIONS
        return TaskSpec(
            self.env_key,
            {"action_permutation": _pick(rng, perms), "food_reward": float(rng.uniform(0.5, 2.0))},
        )


class PuckWorldTasks(TaskDistribution):
    env_key = "puckworld"

    def sample(self, rng: np.random.Generator, split: str = "train") -> TaskSpec:
        self._check_split(split)
        perms = TRAIN_PERMUTATIONS if split == "train" else TEST_PERMUTATIONS
        return TaskSpec(
            self.env_key,
            {
                "action_permutation": _pick(rng, perms),
                "thrust": float(rng.uniform(0.08, 0.25)),
                "damping": float(rng.uniform(0.85, 0.95)),
            },
        )


class PongTasks(TaskDistribution):
    env_key = "pong_stack"
    TRAIN = tuple((m, d) for m in (0, 1) for d in (0, 1, 2))
    TEST = ((0, 3), (1, 3))

    def __init__(self, **env_kwargs: Any) -> None:
        super().__init__(**env_kwargs)
        self._cache: dict[str, gym.Env] = {}  # ALE construction is comparatively expensive

    def sample(self, rng: np.random.Generator, split: str = "train") -> TaskSpec:
        self._check_split(split)
        mode, difficulty = _pick(rng, self.TRAIN if split == "train" else self.TEST)
        return TaskSpec(self.env_key, {"mode": mode, "difficulty": difficulty})

    def make(self, task: TaskSpec) -> gym.Env:
        if task.name not in self._cache:
            self._cache[task.name] = super().make(task)
        return self._cache[task.name]


TASK_DISTRIBUTIONS: dict[str, type[TaskDistribution]] = {
    "snake": SnakeTasks,
    "puckworld": PuckWorldTasks,
    "pong": PongTasks,
}


def get_task_distribution(name: str, **env_kwargs: Any) -> TaskDistribution:
    if name not in TASK_DISTRIBUTIONS:
        raise KeyError(f"Unknown task family '{name}'. Available: {', '.join(TASK_DISTRIBUTIONS)}")
    return TASK_DISTRIBUTIONS[name](**env_kwargs)
