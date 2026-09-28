"""Atari Pong wrappers built on the Arcade Learning Environment (ALE).

ROMs ship with ``ale-py>=0.9``; no AutoROM step is required. With Gymnasium>=1.0 the ALE
namespace must be registered explicitly (``gym.register_envs(ale_py)``), which
:func:`make_ale_pong` handles.

Action space is reduced to 3 actions: NOOP, RIGHT (paddle up), LEFT (paddle down).
Pong's ``mode`` (0/1) and ``difficulty`` (0-3) flags define a small discrete family of
task variants used by the meta-learning module.
"""

from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

PONG_ACTIONS: tuple[int, ...] = (0, 2, 3)  # ALE indices: NOOP, RIGHT (up), LEFT (down)
_PLAYFIELD_ROWS = slice(34, 194)  # crop scoreboard and top/bottom walls


def make_ale_pong(
    mode: int = 0,
    difficulty: int = 0,
    frameskip: int = 4,
    repeat_action_probability: float = 0.25,
    render_mode: str | None = None,
) -> gym.Env:
    try:
        import ale_py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Pong requires the Arcade Learning Environment. Install it with "
            "`pip install -e .[atari]` (ale-py>=0.10 bundles the ROMs)."
        ) from exc
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
    return gym.make(
        "ALE/Pong-v5",
        obs_type="grayscale",
        mode=mode,
        difficulty=difficulty,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        render_mode=render_mode,
    )


def preprocess_frame(gray: np.ndarray, size: int) -> np.ndarray:
    """Crop the playfield and downsample to ``(size, size)`` uint8."""
    return cv2.resize(gray[_PLAYFIELD_ROWS], (size, size), interpolation=cv2.INTER_AREA)


class _PongBase(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mode: int = 0,
        difficulty: int = 0,
        frameskip: int = 4,
        repeat_action_probability: float = 0.25,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.mode, self.difficulty = int(mode), int(difficulty)
        self.ale_env = make_ale_pong(mode, difficulty, frameskip, repeat_action_probability, render_mode)
        self.action_space = spaces.Discrete(len(PONG_ACTIONS))

    def _ale_step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action!r}")
        return self.ale_env.step(PONG_ACTIONS[int(action)])

    def render(self):
        return self.ale_env.render()

    def close(self) -> None:
        self.ale_env.close()


class PongStackEnv(_PongBase):
    """Stack of ``frame_stack`` float32 frames in [0, 1], shape ``(frame_stack, size, size)``.

    Used by the meta-learning module's lightweight CNN policy.
    """

    def __init__(self, frame_stack: int = 4, size: int = 40, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frame_stack, self.size = int(frame_stack), int(size)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.frame_stack, self.size, self.size), dtype=np.float32
        )
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self.ale_env.reset(seed=seed, options=options)
        frame = self._process(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return np.stack(self._frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._ale_step(action)
        self._frames.append(self._process(obs))
        return np.stack(self._frames, axis=0), float(reward), terminated, truncated, info

    def _process(self, obs: np.ndarray) -> np.ndarray:
        return preprocess_frame(obs, self.size).astype(np.float32) / 255.0


class PongImageEnv(_PongBase):
    """Single ``(1, size, size)`` uint8 frame. Temporal context is added downstream with
    ``VecFrameStack`` so that all transfer environments share one observation format."""

    def __init__(self, size: int = 84, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = int(size)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self.ale_env.reset(seed=seed, options=options)
        return preprocess_frame(obs, self.size)[None], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._ale_step(action)
        return preprocess_frame(obs, self.size)[None], float(reward), terminated, truncated, info
