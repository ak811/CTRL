"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch RNGs and return a dedicated NumPy ``Generator``.

    Environments in this project draw randomness from their own seeded generators
    (``env.reset(seed=...)``), so the returned generator should be used to derive
    per-episode / per-task seeds rather than relying on global state.

    Note: ``PYTHONHASHSEED`` only takes effect if exported *before* the interpreter
    starts, so it is intentionally not set here.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by third-party code
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
