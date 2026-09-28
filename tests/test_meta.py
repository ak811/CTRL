from __future__ import annotations

import numpy as np
import torch

from envs import make_env
from meta.config import MetaConfig
from meta.policy import PolicyNetwork
from meta.reinforce import Episode, reinforce_loss, reward_to_go, run_episode
from meta.reptile import clone_model, reptile_update
from meta.train import train


def test_reward_to_go():
    np.testing.assert_allclose(reward_to_go(np.array([1.0, 1.0, 1.0]), 0.5), [1.75, 1.5, 1.0])


def test_reinforce_loss_finite_for_single_step_episode():
    """Regression test: per-episode std normalisation used to yield NaN."""
    lp = torch.zeros(1, requires_grad=True)
    ep = Episode(lp, torch.zeros(1), np.array([1.0], dtype=np.float32))
    loss = reinforce_loss([ep], gamma=0.99, entropy_coef=0.01)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lp.grad).all()


def test_policy_shapes():
    mlp = PolicyNetwork((6,), 4)
    assert mlp(torch.zeros(6)).shape == (1, 4)
    cnn = PolicyNetwork((4, 40, 40), 3)
    assert cnn(torch.zeros(2, 4, 40, 40)).shape == (2, 3)


def test_reptile_update_interpolates():
    meta = torch.nn.Linear(2, 2)
    adapted = clone_model(meta)
    with torch.no_grad():
        for p in adapted.parameters():
            p.add_(1.0)
    before = [p.clone() for p in meta.parameters()]
    reptile_update(meta, [adapted], step_size=0.25)
    for b, p in zip(before, meta.parameters()):
        torch.testing.assert_close(p, b + 0.25)


def test_rollout_runs():
    env = make_env("snake")
    ep = run_episode(env, PolicyNetwork(env.observation_space.shape, 4), torch.device("cpu"), 30, seed=0)
    assert 1 <= len(ep) <= 30 and ep.log_probs.requires_grad


def test_end_to_end_training(tmp_path):
    cfg = MetaConfig(
        env="snake",
        iterations=2,
        meta_batch_size=2,
        inner_steps=2,
        episodes_per_step=1,
        max_episode_steps=20,
        eval_every=2,
        eval_tasks=1,
        eval_episodes=1,
        final_eval_tasks=1,
        output_root=str(tmp_path),
    )
    run_dir = train(cfg)
    for name in ("policy.pt", "history.csv", "adaptation.png", "adaptation_summary.json", "config.yaml"):
        assert (run_dir / name).exists(), name
