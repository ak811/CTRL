"""Environment registry: short keys, lazy construction and Gymnasium registration."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

ENTRY_POINTS: dict[str, str] = {
    "snake": "envs.snake:SnakeGridEnv",
    "snake_image": "envs.snake:SnakeImageEnv",
    "puckworld": "envs.puckworld:PuckWorldVectorEnv",
    "puckworld_image": "envs.puckworld:PuckWorldImageEnv",
    "pong_stack": "envs.pong:PongStackEnv",
    "pong_image": "envs.pong:PongImageEnv",
}

GYM_IDS: dict[str, str] = {
    "snake": "CTRL/Snake-v0",
    "snake_image": "CTRL/SnakeImage-v0",
    "puckworld": "CTRL/PuckWorld-v0",
    "puckworld_image": "CTRL/PuckWorldImage-v0",
    "pong_stack": "CTRL/PongStack-v0",
    "pong_image": "CTRL/PongImage-v0",
}


def _load(entry_point: str) -> type[gym.Env]:
    module_name, _, attr = entry_point.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def make_env(key: str, **kwargs: Any) -> gym.Env:
    """Construct an environment by registry key (no Gymnasium wrappers applied)."""
    if key not in ENTRY_POINTS:
        raise KeyError(f"Unknown environment '{key}'. Available: {', '.join(sorted(ENTRY_POINTS))}")
    return _load(ENTRY_POINTS[key])(**kwargs)


def register_envs() -> None:
    """Register all environments with Gymnasium, e.g. ``gym.make("CTRL/Snake-v0")``."""
    for key, env_id in GYM_IDS.items():
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=ENTRY_POINTS[key])
