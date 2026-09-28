"""Supervised transfer between sine tasks: scratch vs. frozen trunk vs. full fine-tuning.

Metrics (Taylor & Stone, 2009): *jumpstart* (target MSE before any target update),
*final* MSE (mean of the last 5% of steps) and normalised *AUC* of the log-MSE curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.io import ensure_dir, write_csv
from continual.config import ContinualConfig
from continual.data import SineTask
from continual.models import SineMLP

MODES = ("scratch", "frozen", "finetune")
_mse = nn.MSELoss()


@dataclass
class TransferResult:
    curve: np.ndarray  # target-task MSE per step, evaluated before each update
    grad_norms: np.ndarray


def grad_norm(model: nn.Module) -> float:
    total = sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
    return total**0.5


def _new_model(cfg: ContinualConfig, device) -> SineMLP:
    return SineMLP(hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers).to(device)


def train_regressor(
    model: nn.Module, task: SineTask, steps: int, cfg: ContinualConfig, rng, device
) -> tuple[np.ndarray, np.ndarray]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    curve = np.empty(steps, dtype=np.float32)
    gnorm = np.empty(steps, dtype=np.float32)
    for step in range(steps):
        x, y = task.sample(cfg.batch_size, rng, device, cfg.x_range)
        loss = _mse(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm[step] = grad_norm(model)
        opt.step()
        curve[step] = float(loss.item())
    return curve, gnorm


def run_transfer_pair(
    source: SineTask, target: SineTask, seed: int, cfg: ContinualConfig, device, out_dir: Path
) -> dict[str, TransferResult]:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    src_model = _new_model(cfg, device)
    train_regressor(src_model, source, cfg.transfer_steps, cfg, rng, device)

    results: dict[str, TransferResult] = {}
    for i, mode in enumerate(MODES):
        torch.manual_seed(seed * 1000 + i)
        model = _new_model(cfg, device)
        if mode != "scratch":
            model.load_state_dict(src_model.state_dict())
        if mode == "frozen":  # linear probe: only the output layer adapts
            for p in model.trunk.parameters():
                p.requires_grad_(False)
        curve, gnorm = train_regressor(model, target, cfg.transfer_steps, cfg, rng, device)
        results[mode] = TransferResult(curve, gnorm)

    tag = f"src{source.amplitude:.2g}_{source.phase:.2g}_tgt{target.amplitude:.2g}_{target.phase:.2g}_seed{seed}"
    rows = zip(range(cfg.transfer_steps), *(results[m].curve for m in MODES), strict=True)
    write_csv(out_dir / f"curves_{tag}.csv", rows, header=["step", *MODES])
    return results


def transfer_metrics(curve: np.ndarray) -> dict[str, float]:
    tail = max(1, int(0.05 * curve.size))
    return {
        "jumpstart_mse": float(curve[0]),
        "final_mse": float(curve[-tail:].mean()),
        "auc_log_mse": float(np.log10(np.maximum(curve, 1e-12)).mean()),
    }
