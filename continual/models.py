"""Regression networks for the sine benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class SineMLP(nn.Module):
    """MLP ``1 -> hidden x L -> n_heads``.

    With ``n_heads > 1`` the network is a shared trunk with one linear output per task
    (task-incremental continual learning); ``head`` selects the output column.
    """

    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 40, hidden_layers: int = 2, n_heads: int = 1
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_heads)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        out = self.head(self.trunk(x))
        return out[:, head : head + 1]
