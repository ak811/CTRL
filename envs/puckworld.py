"""PuckWorld: continuous-state navigation of a damped puck towards a target.

Two variants:

* :class:`PuckWorldVectorEnv` - 6-D state ``[pos(2), vel(2), target(2)]`` and 4 discrete
  thrust actions. Physics parameters (thrust, damping) and the action mapping are exposed
  so that a *distribution* of dynamics can be sampled for meta-learning.
* :class:`PuckWorldImageEnv` - ``(1, 84, 84)`` grayscale frames and 2-D continuous
  actions, used for cross-domain transfer with a shared CNN encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.snake import _validate_permutation

# Unit thrust directions for canonical actions: N, S, W, E (image coordinates, y grows down).
THRUST_DIRECTIONS = np.array([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)


class PuckWorldVectorEnv(gym.Env):
    """Damped point-mass navigation in the unit square.

    Dynamics per step: ``v <- damping * (v + dt * thrust * a)``, ``p <- clip(p + dt * v)``;
    velocity components are zeroed on wall contact. Reward is potential-based progress
    ``shaping_scale * (d_{t-1} - d_t)`` plus ``success_bonus`` on reaching the goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 300,
        dt: float = 0.1,
        thrust: float = 0.1,
        damping: float = 0.9,
        goal_radius: float = 0.05,
        success_bonus: float = 5.0,
        shaping_scale: float = 2.0,
        action_permutation: Sequence[int] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.thrust = float(thrust)
        self.damping = float(damping)
        self.goal_radius = float(goal_radius)
        self.success_bonus = float(success_bonus)
        self.shaping_scale = float(shaping_scale)
        self.action_permutation = _validate_permutation(action_permutation, 4)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.position = self.np_random.random(2).astype(np.float32)
        self.target = self.np_random.random(2).astype(np.float32)
        while np.linalg.norm(self.position - self.target) < 2 * self.goal_radius:
            self.target = self.np_random.random(2).astype(np.float32)
        self.velocity[:] = 0.0
        self._steps = 0
        return self._get_obs(), {"distance": self._distance()}

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action!r}")
        self._steps += 1
        prev_dist = self._distance()

        direction = THRUST_DIRECTIONS[self.action_permutation[int(action)]]
        self.velocity = self.damping * (self.velocity + self.dt * self.thrust * direction)
        new_pos = self.position + self.dt * self.velocity
        hit_wall = (new_pos < 0.0) | (new_pos > 1.0)
        self.position = np.clip(new_pos, 0.0, 1.0).astype(np.float32)
        self.velocity[hit_wall] = 0.0

        dist = self._distance()
        reward = self.shaping_scale * (prev_dist - dist)
        terminated = dist < self.goal_radius
        if terminated:
            reward += self.success_bonus
        truncated = (not terminated) and self._steps >= self.max_steps
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"distance": dist}

    def _distance(self) -> float:
        return float(np.linalg.norm(self.position - self.target))

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.position, self.velocity, self.target]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def render(self, size: int = 200) -> np.ndarray:
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        px, py = (int(v) for v in self.position * (size - 1))
        tx, ty = (int(v) for v in self.target * (size - 1))
        cv2.circle(img, (tx, ty), max(2, int(self.goal_radius * size)), (255, 80, 80), -1)
        cv2.circle(img, (px, py), 6, (40, 90, 255), -1)
        return img


class PuckWorldImageEnv(gym.Env):
    """Pixel-based PuckWorld with continuous 2-D velocity control.

    ``reward_mode="dense"`` (default) uses potential-based shaping
    ``(d_{t-1} - d_t) / speed - step_cost`` plus ``success_bonus``; potential-based shaping
    preserves the optimal policy (Ng et al., 1999). ``reward_mode="sparse"`` reproduces the
    original formulation: zero reward until the episode ends, then the negative mean
    distance over the episode.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 300,
        size: int = 84,
        speed: float = 2.0,
        goal_radius: float = 5.0,
        reward_mode: str = "dense",
        step_cost: float = 0.01,
        success_bonus: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if reward_mode not in ("dense", "sparse"):
            raise ValueError("reward_mode must be 'dense' or 'sparse'")
        self.max_steps = int(max_steps)
        self.size = int(size)
        self.speed = float(speed)
        self.goal_radius = float(goal_radius)
        self.reward_mode = reward_mode
        self.step_cost = float(step_cost)
        self.success_bonus = float(success_bonus)
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8)

        center = (self.size - 1) / 2.0
        self.puck = np.array([center, center], dtype=np.float32)
        self.goal = self.puck.copy()
        self._steps = 0
        self._cumulative_distance = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        center = (self.size - 1) / 2.0
        self.puck[:] = center
        margin = 0.12 * self.size
        while True:
            self.goal = self.np_random.uniform(margin, self.size - 1 - margin, size=2).astype(np.float32)
            if self._distance() > 2 * self.goal_radius:
                break
        self._steps = 0
        self._cumulative_distance = 0.0
        return self._obs(), {"distance": self._distance()}

    def step(self, action):
        self._steps += 1
        prev = self._distance()
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        self.puck = np.clip(self.puck + a * self.speed, 0.0, self.size - 1).astype(np.float32)

        dist = self._distance()
        self._cumulative_distance += dist
        terminated = dist < self.goal_radius
        truncated = (not terminated) and self._steps >= self.max_steps

        if self.reward_mode == "dense":
            reward = (prev - dist) / self.speed - self.step_cost
            if terminated:
                reward += self.success_bonus
        else:
            reward = -(self._cumulative_distance / self._steps) if (terminated or truncated) else 0.0
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"distance": dist}

    def _distance(self) -> float:
        return float(np.linalg.norm(self.puck - self.goal))

    def _obs(self) -> np.ndarray:
        frame = np.zeros((self.size, self.size), dtype=np.uint8)
        gx, gy = (int(round(v)) for v in self.goal)
        px, py = (int(round(v)) for v in self.puck)
        cv2.circle(frame, (gx, gy), 3, 128, -1)
        cv2.circle(frame, (px, py), 4, 255, -1)
        return frame[None, :, :]

    def render(self) -> np.ndarray:
        gray = self._obs()[0]
        return np.repeat(gray[:, :, None], 3, axis=2)
