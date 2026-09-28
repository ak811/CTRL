"""Meta-train a policy initialisation over a task family.

    python -m meta.train --env puckworld --algorithm reptile
    python -m meta.train --config configs/meta/snake.yaml --seed 1
    python -m meta.train --env puckworld --algorithm multitask     # non-meta baseline

``reptile``  : batched Reptile over ``meta_batch_size`` tasks per iteration.
``multitask``: joint REINFORCE over tasks with the same environment-interaction budget;
               this is the standard "pretrained" baseline for meta-learning claims.

After training, the learned initialisation is compared against a random initialisation
on held-out test tasks (``adaptation.png``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from common.config import parse_config
from common.io import create_run_dir, save_run_metadata, write_csv
from common.logger import get_logger
from common.plotting import plot_curves
from common.seed import set_seed
from common.torch_utils import count_parameters, resolve_device
from envs.tasks import TaskDistribution, get_task_distribution
from meta.config import MetaConfig
from meta.evaluate import adaptation_curves, save_checkpoint, write_report
from meta.policy import PolicyNetwork
from meta.reinforce import evaluate, reinforce_loss, run_episode
from meta.reptile import adapt, clone_model, linear_anneal, make_optimizer, reptile_update

log = get_logger("meta")


def build_policy(cfg: MetaConfig, tasks: TaskDistribution, rng: np.random.Generator) -> PolicyNetwork:
    probe = tasks.make(tasks.sample(rng))
    return PolicyNetwork(probe.observation_space.shape, probe.action_space.n, cfg.hidden_sizes)


def _adapt_kwargs(cfg: MetaConfig, device, rng) -> dict:
    return dict(
        steps=cfg.inner_steps,
        episodes_per_step=cfg.episodes_per_step,
        lr=cfg.inner_lr,
        optimizer=cfg.inner_optimizer,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        max_steps=cfg.max_episode_steps,
        device=device,
        rng=rng,
    )


def reptile_iteration(policy, tasks, cfg, step_size, rng, device) -> tuple[float, float]:
    adapted, pre, post = [], [], []
    for _ in range(cfg.meta_batch_size):
        env = tasks.make(tasks.sample(rng, "train"))
        model = clone_model(policy)
        returns, _ = adapt(model, env, **_adapt_kwargs(cfg, device, rng))
        adapted.append(model)
        pre.append(returns[0])
        post.append(returns[-1])
    reptile_update(policy, adapted, step_size)
    return float(np.mean(pre)), float(np.mean(post))


def multitask_iteration(policy, tasks, cfg, opt, rng, device) -> tuple[float, float]:
    step_returns = []
    for _ in range(cfg.inner_steps):
        losses, rets = [], []
        for _ in range(cfg.meta_batch_size):
            env = tasks.make(tasks.sample(rng, "train"))
            eps = [
                run_episode(env, policy, device, cfg.max_episode_steps, seed=int(rng.integers(2**31)))
                for _ in range(cfg.episodes_per_step)
            ]
            losses.append(reinforce_loss(eps, cfg.gamma, cfg.entropy_coef))
            rets.append(np.mean([e.total_reward for e in eps]))
        opt.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        step_returns.append(float(np.mean(rets)))
    return step_returns[0], step_returns[-1]


def quick_test_eval(policy, tasks, cfg, device, n_tasks: int, seed: int) -> float:
    """Mean post-adaptation return on a fixed set of held-out tasks."""
    task_rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_tasks):
        env = tasks.make(tasks.sample(task_rng, "test"))
        model = clone_model(policy)
        rng = np.random.default_rng(seed * 7919 + i)
        adapt(model, env, **_adapt_kwargs(cfg, device, rng))
        scores.append(evaluate(env, model, device, cfg.eval_episodes, cfg.max_episode_steps, rng))
    return float(np.mean(scores))


def train(cfg: MetaConfig) -> Path:
    rng = set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = create_run_dir(
        cfg.output_root, "meta", cfg.env, run_name=cfg.run_name or f"{cfg.algorithm}_seed{cfg.seed}"
    )
    save_run_metadata(run_dir, cfg)
    get_logger("meta", run_dir / "train.log")

    tasks = get_task_distribution(cfg.env, **cfg.env_kwargs())
    policy = build_policy(cfg, tasks, rng).to(device)
    torch.manual_seed(cfg.seed + 1)
    random_init = PolicyNetwork(policy.obs_shape, policy.n_actions, cfg.hidden_sizes).to(device)
    log.info(
        "env=%s algorithm=%s params=%d device=%s -> %s",
        cfg.env,
        cfg.algorithm,
        count_parameters(policy),
        device,
        run_dir,
    )

    outer_opt = (
        make_optimizer(policy.parameters(), cfg.inner_optimizer, cfg.inner_lr)
        if cfg.algorithm == "multitask"
        else None
    )
    history: list[tuple] = []
    t0 = time.time()
    for it in range(cfg.iterations):
        step_size = (
            linear_anneal(it, cfg.iterations, cfg.meta_lr, cfg.anneal_meta_lr)
            if cfg.algorithm == "reptile"
            else float("nan")
        )
        if cfg.algorithm == "reptile":
            pre, post = reptile_iteration(policy, tasks, cfg, step_size, rng, device)
        else:
            pre, post = multitask_iteration(policy, tasks, cfg, outer_opt, rng, device)

        test_score = float("nan")
        if (it + 1) % cfg.eval_every == 0 or it == cfg.iterations - 1:
            test_score = quick_test_eval(policy, tasks, cfg, device, cfg.eval_tasks, seed=cfg.seed + 999)
            save_checkpoint(run_dir / "checkpoint.pt", policy, cfg, {"iteration": it + 1})
            log.info(
                "iter %4d/%d | outer step %.3f | train pre %.3f -> post %.3f | test post-adapt %.3f | %.0fs",
                it + 1,
                cfg.iterations,
                step_size,
                pre,
                post,
                test_score,
                time.time() - t0,
            )
        history.append((it, step_size, pre, post, test_score))

    save_checkpoint(run_dir / "policy.pt", policy, cfg, {"iteration": cfg.iterations})
    write_csv(
        run_dir / "history.csv", history, ["iteration", "step_size", "train_pre", "train_post", "test_post"]
    )
    h = np.asarray(history, dtype=np.float64)
    plot_curves(
        {"pre-adaptation": h[None, :, 2], "post-adaptation": h[None, :, 3]},
        run_dir / "training.png",
        title=f"{cfg.algorithm} on {cfg.env}: training-task returns",
        xlabel="Meta-iteration",
        ylabel="Episode return",
        smoothing=20,
    )

    log.info("final evaluation on %d held-out tasks", cfg.final_eval_tasks)
    curves = adaptation_curves(
        {cfg.algorithm: policy, "random-init": random_init},
        tasks,
        cfg,
        cfg.final_eval_tasks,
        seed=cfg.seed + 2024,
        device=device,
    )
    write_report(curves, run_dir, f"{cfg.env}: adaptation on held-out tasks")
    log.info("outputs written to %s", run_dir)
    return run_dir


def main(argv: Sequence[str] | None = None) -> Path:
    cfg, _ = parse_config(
        MetaConfig, "Meta-RL (Reptile / multi-task) over Snake, PuckWorld or Pong task families.", argv
    )
    return train(cfg)


if __name__ == "__main__":
    main()
