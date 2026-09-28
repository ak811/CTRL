"""Categorical policy networks for vector (MLP) and stacked-frame (CNN) observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """``obs -> logits``. The encoder is chosen from the observation rank:
    1-D -> identity (MLP policy); 3-D ``(C, H, W)`` -> 2-layer CNN."""

    def __init__(
        self,
        obs_shape: Sequence[int],
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 64),
        conv_channels: Sequence[int] = (16, 32),
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(s) for s in obs_shape)
        self.n_actions = int(n_actions)
        if len(self.obs_shape) == 3:
            c = self.obs_shape[0]
            self.encoder: nn.Module = nn.Sequential(
                nn.Conv2d(c, conv_channels[0], kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                in_dim = int(self.encoder(torch.zeros(1, *self.obs_shape)).shape[1])
        elif len(self.obs_shape) == 1:
            self.encoder = nn.Identity()
            in_dim = self.obs_shape[0]
        else:
            raise ValueError(f"Unsupported observation shape {self.obs_shape}")

        layers: list[nn.Module] = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.n_actions)
        nn.init.orthogonal_(self.head.weight, gain=0.01)  # near-uniform initial policy
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        return self.head(self.body(self.encoder(obs)))

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self(obs))
