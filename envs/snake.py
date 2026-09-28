"""Snake: a grid-world benchmark with vector and image observation variants.

Observation encoding (vector variant, flattened ``grid_size**2``):
    0.0 empty, 0.5 body, 1.0 head, -1.0 food

The head is encoded separately from the body so that the policy can infer heading, which
is required for the task to be Markov. Action semantics can be permuted via
``action_permutation`` to create a family of related tasks for meta-learning.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# (d_row, d_col) for canonical actions: up, down, left, right
MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validate_permutation(perm: Sequence[int] | None, n: int) -> tuple[int, ...]:
    if perm is None:
        return tuple(range(n))
    perm = tuple(int(a) for a in perm)
    if sorted(perm) != list(range(n)):
        raise ValueError(f"action_permutation must be a permutation of range({n}), got {perm}")
    return perm


class SnakeGridEnv(gym.Env):
    """Grid Snake with vector observations.

    Rewards: ``food_reward`` for eating, ``death_penalty`` on collision, ``step_reward``
    otherwise. Episodes terminate on collision or when the board is full, and truncate
    after ``max_steps``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    EMPTY, BODY, HEAD, FOOD = 0.0, 0.5, 1.0, -1.0

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 200,
        food_reward: float = 1.0,
        death_penalty: float = -1.0,
        step_reward: float = 0.01,
        action_permutation: Sequence[int] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 4:
            raise ValueError("grid_size must be >= 4")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.step_reward = float(step_reward)
        self.action_permutation = _validate_permutation(action_permutation, 4)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.grid_size * self.grid_size,), dtype=np.float32
        )

        self._snake: deque[tuple[int, int]] = deque()
        self._food: tuple[int, int] | None = None
        self._steps = 0
        self._score = 0

    # ------------------------------------------------------------------ core API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c = self.grid_size // 2
        self._snake = deque([(c, c)])
        self._food = self._place_food()
        self._steps = 0
        self._score = 0
        return self._get_obs(), self._info()

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action!r}")
        self._steps += 1
        d_row, d_col = MOVES[self.action_permutation[int(action)]]
        head_row, head_col = self._snake[0]
        new_head = (head_row + d_row, head_col + d_col)

        grows = new_head == self._food
        # The tail vacates its cell this step unless the snake grows.
        body = list(self._snake) if grows else list(self._snake)[:-1]
        out_of_bounds = not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)

        terminated = False
        if out_of_bounds or new_head in set(body):
            terminated = True
            reward = self.death_penalty
        else:
            self._snake.appendleft(new_head)
            if grows:
                self._score += 1
                reward = self.food_reward
                self._food = self._place_food()
                if self._food is None:  # board filled: the game is won
                    terminated = True
            else:
                self._snake.pop()
                reward = self.step_reward

        truncated = (not terminated) and self._steps >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    # ---------------------------------------------------------------- internals
    def _place_food(self) -> tuple[int, int] | None:
        occupied = set(self._snake)
        free = [
            (r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in occupied
        ]
        if not free:
            return None
        return free[int(self.np_random.integers(len(free)))]

    def _grid(self) -> np.ndarray:
        grid = np.full((self.grid_size, self.grid_size), self.EMPTY, dtype=np.float32)
        for r, c in list(self._snake)[1:]:
            grid[r, c] = self.BODY
        if self._snake:
            r, c = self._snake[0]
            grid[r, c] = self.HEAD
        if self._food is not None:
            grid[self._food] = self.FOOD
        return grid

    def _get_obs(self) -> np.ndarray:
        return self._grid().ravel()

    def _info(self) -> dict:
        return {"score": self._score, "length": len(self._snake)}

    # ---------------------------------------------------------------- rendering
    def render_gray(self, size: int = 84) -> np.ndarray:
        """Grayscale ``(size, size)`` uint8 frame: body 128, head 255, food 64."""
        lut = {self.EMPTY: 0, self.BODY: 128, self.HEAD: 255, self.FOOD: 64}
        grid = self._grid()
        img = np.zeros(grid.shape, dtype=np.uint8)
        for value, intensity in lut.items():
            img[grid == value] = intensity
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    def render(self, cell: int = 20) -> np.ndarray:
        img = np.zeros((self.grid_size * cell, self.grid_size * cell, 3), dtype=np.uint8)
        for i, (r, c) in enumerate(self._snake):
            color = (0, 255, 0) if i == 0 else (0, 160, 0)
            img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = color
        if self._food is not None:
            r, c = self._food
            img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = (255, 60, 60)
        return img


class SnakeImageEnv(gym.Env):
    """Snake with ``(1, size, size)`` uint8 image observations (channels-first) for CNN policies."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 500,
        size: int = 84,
        render_mode: str | None = None,
        **core_kwargs,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.render_mode = render_mode
        self.core = SnakeGridEnv(
            grid_size=grid_size, max_steps=max_steps, render_mode=render_mode, **core_kwargs
        )
        self.action_space = self.core.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _, info = self.core.reset(seed=seed, options=options)
        return self._obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.core.step(action)
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return self.core.render_gray(self.size)[None, :, :]

    def render(self) -> np.ndarray:
        return self.core.render()
