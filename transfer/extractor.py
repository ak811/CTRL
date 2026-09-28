"""Shared convolutional encoder (Nature-DQN architecture; Mnih et al., 2015)."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NatureCNNEncoder(BaseFeaturesExtractor):
    """``(C, H, W)`` uint8/float observations -> ``features_dim`` features.

    The flatten size is inferred from the observation space, so any input resolution
    that survives the conv stack is supported (the original hard-coded 7x7x64 = 84x84).
    SB3 normalises uint8 image observations to [0, 1] before this module is called.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# Backward-compatible name used by the original project.
GeneralizedExtractor = NatureCNNEncoder
