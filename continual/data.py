"""Sine-wave regression tasks ``y = A sin(x + phi)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SineTask:
    amplitude: float
    phase: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.sin(x + self.phase)

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        device: torch.device | str = "cpu",
        x_range: tuple[float, float] = (-5.0, 5.0),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = rng.uniform(x_range[0], x_range[1], size=(n, 1)).astype(np.float32)
        y = self(x).astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def grid(
        self, n: int, device: torch.device | str = "cpu", x_range: tuple[float, float] = (-5.0, 5.0)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic evaluation set on an evenly spaced grid."""
        x = np.linspace(x_range[0], x_range[1], n, dtype=np.float32)[:, None]
        y = self(x).astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    @property
    def label(self) -> str:
        return f"A={self.amplitude:.2g}, φ={self.phase:.2g}"


def sample_task(
    rng: np.random.Generator,
    amplitude_range: tuple[float, float] = (0.1, 5.0),
    phase_range: tuple[float, float] = (0.0, np.pi),
) -> SineTask:
    return SineTask(float(rng.uniform(*amplitude_range)), float(rng.uniform(*phase_range)))


def generate_sine(amplitude: float, phase: float, n: int = 100, device: torch.device | None = None, rng=None):
    """Backward-compatible helper returning ``(x, y)`` tensors for a single task."""
    rng = rng if rng is not None else np.random.default_rng()
    return SineTask(amplitude, phase).sample(n, rng, device or "cpu")
