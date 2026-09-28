from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from continual.config import ContinualConfig
from continual.data import SineTask
from continual.ewc import EWCRegularizer, fisher_diagonal, run_sequence
from continual.maml import adapt_functional, train_maml
from continual.metrics import continual_metrics
from continual.models import SineMLP

TINY = ContinualConfig(
    n_seeds=1,
    maml_iterations=3,
    maml_meta_batch=2,
    maml_eval_tasks=2,
    cl_steps_per_task=20,
    ewc_fisher_samples=16,
    task_params=((1.0, 0.0), (2.0, 1.0), (0.5, 2.0)),
)


def test_maml_meta_parameters_are_updated():
    """Regression test: the original loop never produced meta-gradients."""
    model, history = train_maml(TINY, seed=0, device="cpu")
    torch.manual_seed(0)
    init = SineMLP(hidden_dim=TINY.hidden_dim, hidden_layers=TINY.hidden_layers)
    assert any(not torch.equal(a, b) for a, b in zip(init.parameters(), model.parameters()))
    assert np.isfinite(history).all()


@pytest.mark.parametrize("first_order", [False, True])
def test_adapt_functional_gradient_flow(first_order):
    model = SineMLP()
    params = dict(model.named_parameters())
    x, y = SineTask(1.0, 0.0).sample(10, np.random.default_rng(0))
    fast = adapt_functional(model, params, x, y, steps=1, lr=0.01, first_order=first_order)
    loss = F.mse_loss(torch.func.functional_call(model, fast, (x,)), y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_second_order_differs_from_first_order():
    torch.manual_seed(0)
    model = SineMLP()
    x, y = SineTask(2.0, 0.5).sample(10, np.random.default_rng(0))
    grads = []
    for fo in (False, True):
        model.zero_grad()
        fast = adapt_functional(model, dict(model.named_parameters()), x, y, 1, 0.1, fo)
        F.mse_loss(torch.func.functional_call(model, fast, (x,)), y).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert not torch.allclose(grads[0], grads[1])


def test_ewc_penalty_zero_at_anchor_and_positive_elsewhere():
    model = SineMLP()
    x, _ = SineTask(1.0, 0.0).sample(32, np.random.default_rng(0))
    reg = EWCRegularizer(lam=10.0)
    reg.consolidate(model, fisher_diagonal(model, x))
    assert float(reg.penalty(model)) == pytest.approx(0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    assert float(reg.penalty(model)) > 0.0


def test_fisher_matches_manual_jacobian():
    torch.manual_seed(0)
    model = SineMLP(hidden_dim=8, hidden_layers=1)
    x = torch.linspace(-1, 1, 5).unsqueeze(1)
    fisher = fisher_diagonal(model, x)
    manual = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for xi in x:
        model.zero_grad()
        model(xi.unsqueeze(0)).sum().backward()
        for n, p in model.named_parameters():
            manual[n] += p.grad.pow(2) / len(x)
    for n in fisher:
        torch.testing.assert_close(fisher[n], manual[n])


def test_separate_mode_keeps_all_tasks_online_mode_keeps_one():
    model = SineMLP()
    x, _ = SineTask(1.0, 0.0).sample(8, np.random.default_rng(0))
    sep, onl = EWCRegularizer(1.0, "separate"), EWCRegularizer(1.0, "online")
    for _ in range(3):
        f = fisher_diagonal(model, x)
        sep.consolidate(model, f)
        onl.consolidate(model, f)
    assert len(sep.anchors) == 3 and len(onl.anchors) == 1


@pytest.mark.parametrize("scenario", ["task", "domain"])
def test_run_sequence_shapes(scenario):
    cfg = ContinualConfig(**{**TINY.__dict__, "cl_scenario": scenario})
    out = run_sequence(cfg, "ewc", seed=0, device="cpu")
    R = out["R"]
    assert R.shape == (3, 3)
    assert np.isfinite(np.tril(R)[np.tril_indices(3)]).all()
    if scenario == "task":
        assert np.isnan(R[0, 1])


def test_continual_metrics_known_matrix():
    R = np.array([[1.0, np.nan, np.nan], [2.0, 0.5, np.nan], [3.0, 1.5, 0.2]])
    m = continual_metrics(R)
    assert m["avg_final_mse"] == pytest.approx((3.0 + 1.5 + 0.2) / 3)
    assert m["bwt"] == pytest.approx(((3.0 - 1.0) + (1.5 - 0.5)) / 2)
    assert m["forgetting"] == pytest.approx(((3.0 - 1.0) + (1.5 - 0.5)) / 2)
