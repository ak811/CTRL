"""Small PyTorch helpers."""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Map ``"auto" | "cpu" | "cuda" | "cuda:N" | "mps"`` to a ``torch.device``."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{spec}' but CUDA is not available.")
    return device


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
