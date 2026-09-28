"""Model-Agnostic Meta-Learning for few-shot sine regression (Finn et al., 2017).

The implementation is purely functional (``torch.func.functional_call``) so that the
meta-gradient flows through the inner-loop updates. ``first_order=True`` gives FOMAML.

Baselines evaluated with the same fine-tuning protocol:
  * ``pretrained`` - a network trained by plain regression on tasks drawn from the same
    distribution (same number of gradient evaluations as MAML),
  * ``scratch`` - random initialisation.

Bug fixed relative to the original implementation: the previous loop adapted a
*copy* of the model with in-place optimiser steps and back-propagated the query loss
into that copy, so the meta-parameters never received gradients and were never updated.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from common.io import ensure_dir, write_csv
from continual.config import ContinualConfig
from continual.data import SineTask, sample_task
from continual.models import SineMLP

Params = dict[str, torch.Tensor]


def adapt_functional(
    model: nn.Module,
    params: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    first_order: bool,
) -> Params:
    """Inner-loop SGD in parameter space; differentiable unless ``first_order``."""
    fast = dict(params)
    for _ in range(steps):
        loss = F.mse_loss(functional_call(model, fast, (x,)), y)
        grads = torch.autograd.grad(loss, list(fast.values()), create_graph=not first_order)
        fast = {n: p - lr * g for (n, p), g in zip(fast.items(), grads, strict=True)}
    return fast


def _new_model(cfg: ContinualConfig, device) -> SineMLP:
    return SineMLP(hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers).to(device)


def train_maml(cfg: ContinualConfig, seed: int, device) -> tuple[SineMLP, np.ndarray]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = _new_model(cfg, device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.maml_meta_lr)
    history = np.empty(cfg.maml_iterations, dtype=np.float32)

    for it in range(cfg.maml_iterations):
        params = dict(model.named_parameters())
        meta_loss = torch.zeros((), device=device)
        for _ in range(cfg.maml_meta_batch):
            task = sample_task(rng, cfg.amplitude_range, cfg.phase_range)
            xs, ys = task.sample(cfg.maml_k_shot, rng, device, cfg.x_range)
            xq, yq = task.sample(cfg.maml_k_shot, rng, device, cfg.x_range)
            fast = adapt_functional(
                model, params, xs, ys, cfg.maml_inner_steps, cfg.maml_inner_lr, cfg.maml_first_order
            )
            meta_loss = meta_loss + F.mse_loss(functional_call(model, fast, (xq,)), yq)
        meta_loss = meta_loss / cfg.maml_meta_batch
        meta_opt.zero_grad(set_to_none=True)
        meta_loss.backward()
        meta_opt.step()
        history[it] = float(meta_loss.item())
    return model, history


def train_pretrained(cfg: ContinualConfig, seed: int, device) -> SineMLP:
    """Multi-task regression baseline with a matched number of optimiser steps."""
    rng = np.random.default_rng(seed + 10_000)
    torch.manual_seed(seed + 10_000)
    model = _new_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.maml_meta_lr)
    for _ in range(cfg.maml_iterations):
        xs, ys = [], []
        for _ in range(cfg.maml_meta_batch):
            task = sample_task(rng, cfg.amplitude_range, cfg.phase_range)
            x, y = task.sample(cfg.maml_k_shot, rng, device, cfg.x_range)
            xs.append(x)
            ys.append(y)
        loss = F.mse_loss(model(torch.cat(xs)), torch.cat(ys))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def few_shot_curve(
    model: nn.Module, task: SineTask, cfg: ContinualConfig, rng: np.random.Generator, device
) -> np.ndarray:
    """Test MSE after 0..eval_steps SGD steps on a K-shot support set."""
    params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
    xs, ys = task.sample(cfg.maml_k_shot, rng, device, cfg.x_range)
    xt, yt = task.grid(cfg.eval_points, device, cfg.x_range)
    curve = np.empty(cfg.maml_eval_steps + 1, dtype=np.float32)
    for step in range(cfg.maml_eval_steps + 1):
        with torch.no_grad():
            curve[step] = float(F.mse_loss(functional_call(model, params, (xt,)), yt))
        if step < cfg.maml_eval_steps:
            params = {
                n: p.detach().requires_grad_(True)
                for n, p in adapt_functional(
                    model, params, xs, ys, 1, cfg.maml_inner_lr, first_order=True
                ).items()
            }
    return curve


def evaluate_initialisations(
    models: Mapping[str, nn.Module], cfg: ContinualConfig, seed: int, device
) -> dict[str, np.ndarray]:
    """Evaluate each initialisation on the *same* held-out tasks and support sets."""
    task_rng = np.random.default_rng(10_000_000 + seed)
    tasks = [sample_task(task_rng, cfg.amplitude_range, cfg.phase_range) for _ in range(cfg.maml_eval_tasks)]
    out: dict[str, np.ndarray] = {}
    for label, model in models.items():
        support_rng = np.random.default_rng(20_000_000 + seed)  # identical support sets per init
        out[label] = np.stack([few_shot_curve(model, t, cfg, support_rng, device) for t in tasks])
    return out


def run_maml_seed(cfg: ContinualConfig, seed: int, device, out_dir: Path):
    ensure_dir(out_dir)
    maml_model, history = train_maml(cfg, seed, device)
    pretrained = train_pretrained(cfg, seed, device)
    torch.manual_seed(seed + 20_000)
    scratch = _new_model(cfg, device)
    models = {"MAML": maml_model, "pretrained": pretrained, "scratch": scratch}
    curves = evaluate_initialisations(models, cfg, seed, device)

    torch.save(maml_model.state_dict(), out_dir / f"maml_init_seed{seed}.pt")
    write_csv(out_dir / f"meta_loss_seed{seed}.csv", enumerate(history.tolist()), ["iteration", "meta_loss"])
    rows = [
        (step, *(float(curves[k][:, step].mean()) for k in models)) for step in range(cfg.maml_eval_steps + 1)
    ]
    write_csv(out_dir / f"few_shot_seed{seed}.csv", rows, ["step", *models])
    return models, curves, history
