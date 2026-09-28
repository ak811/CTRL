"""Shared fixtures. Tests run from the repository root (see pyproject ``pythonpath``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

HAS_ALE = importlib.util.find_spec("ale_py") is not None
requires_ale = pytest.mark.skipif(not HAS_ALE, reason="ale-py not installed")


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(0)  # noqa: NPY002
    torch.manual_seed(0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
