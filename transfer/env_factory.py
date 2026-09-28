"""Vectorised environment construction.

All transfer environments emit ``(1, 84, 84)`` uint8 frames; ``VecFrameStack`` with
``channels_order="first"`` yields ``(frame_stack, 84, 84)``, so a single CNN encoder can be
shared across Pong, Snake and PuckWorld.

Bug fixed relative to the original implementation: ``VecTransposeImage`` was applied to
observations that are *already* channels-first, which raised an assertion at start-up
(and, in older SB3 versions, silently produced ``(84, 1, 84)`` tensors).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecMonitor

from envs.registry import make_env

ENV_KEYS = {"pong": "pong_image", "snake": "snake_image", "puckworld": "puckworld_image"}


def _init(name: str, render_mode: str | None) -> gym.Env:
    return make_env(ENV_KEYS[name], render_mode=render_mode)


def env_fn(name: str, render_mode: str | None = None) -> Callable[[], gym.Env]:
    if name not in ENV_KEYS:
        raise ValueError(f"Unknown env '{name}'. Choose from: {', '.join(ENV_KEYS)}")
    return partial(_init, name, render_mode)


def make_vec_env(
    name: str,
    n_envs: int,
    seed: int,
    frame_stack: int = 4,
    vec_env: str = "dummy",
    render_mode: str | None = None,
) -> VecEnv:
    fns = [env_fn(name, render_mode) for _ in range(n_envs)]
    venv: VecEnv = SubprocVecEnv(fns) if (vec_env == "subproc" and n_envs > 1) else DummyVecEnv(fns)
    venv.seed(seed)
    venv = VecMonitor(venv)
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="first")
    return venv


# Backward-compatible single-env constructor.
def make_env_single(env_name: str, render_mode: str | None = "rgb_array") -> gym.Env:
    from stable_baselines3.common.monitor import Monitor

    return Monitor(env_fn(env_name.lower().strip(), render_mode)())
