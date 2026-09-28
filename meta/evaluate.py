"""Few-shot adaptation benchmark for meta-RL initialisations on held-out tasks.

Every initialisation is adapted on the *same* test tasks with the *same* episode seeds,
so differences in the adaptation curves are attributable to the initialisation alone.

    python -m meta.evaluate \
        --checkpoint reptile=outputs/meta/puckworld/reptile_seed0/policy.pt \
        --checkpoint reptile=outputs/meta/puckworld/reptile_seed1/policy.pt \
        --checkpoint multitask=outputs/meta/puckworld/multitask_seed0/policy.pt \
        --include-random --tasks 32
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.config import build_config, config_to_dict
from common.io import create_run_dir, ensure_dir, write_csv, write_json
from common.logger import get_logger
from common.plotting import plot_curves
from common.stats import summarize
from common.torch_utils import resolve_device
from envs.tasks import TaskDistribution, get_task_distribution
from meta.config import MetaConfig
from meta.policy import PolicyNetwork
from meta.reptile import adapt, clone_model

log = get_logger("meta.evaluate")


def save_checkpoint(path: Path, policy: PolicyNetwork, cfg: MetaConfig, extra: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_shape": policy.obs_shape,
            "n_actions": policy.n_actions,
            "config": config_to_dict(cfg),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[PolicyNetwork, MetaConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = build_config(MetaConfig, overrides=ckpt["config"])
    policy = PolicyNetwork(ckpt["obs_shape"], ckpt["n_actions"], cfg.hidden_sizes).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    return policy, cfg


def adaptation_curves(
    policies: Mapping[str, nn.Module],
    tasks: TaskDistribution,
    cfg: MetaConfig,
    n_tasks: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return ``{label: array[n_tasks, inner_steps + 1]}`` of evaluation returns."""
    task_rng = np.random.default_rng(seed)
    test_tasks = [tasks.sample(task_rng, split="test") for _ in range(n_tasks)]
    curves: dict[str, np.ndarray] = {}
    for label, policy in policies.items():
        rows = []
        for i, task in enumerate(test_tasks):
            model = clone_model(policy)
            env = tasks.make(task)
            _, curve = adapt(
                model,
                env,
                steps=cfg.inner_steps,
                episodes_per_step=cfg.episodes_per_step,
                lr=cfg.inner_lr,
                optimizer=cfg.inner_optimizer,
                gamma=cfg.gamma,
                entropy_coef=cfg.entropy_coef,
                max_steps=cfg.max_episode_steps,
                device=device,
                rng=np.random.default_rng(seed * 100_003 + i),  # shared across initialisations
                eval_episodes=cfg.eval_episodes,
            )
            rows.append(curve)
        curves[label] = np.asarray(rows, dtype=np.float64)
        log.info(
            "%-12s return before: %8.3f | after %d updates: %8.3f",
            label,
            curves[label][:, 0].mean(),
            cfg.inner_steps,
            curves[label][:, -1].mean(),
        )
    return curves


def write_report(curves: Mapping[str, np.ndarray], out_dir: Path, title: str) -> dict:
    ensure_dir(out_dir)
    labels = list(curves)
    steps = next(iter(curves.values())).shape[1]
    write_csv(
        out_dir / "adaptation.csv",
        [(s, *(float(curves[k][:, s].mean()) for k in labels)) for s in range(steps)],
        header=["update", *labels],
    )
    for k, v in curves.items():
        np.save(out_dir / f"adaptation_{k}.npy", v)
    plot_curves(
        dict(curves),
        out_dir / "adaptation.png",
        title=title,
        xlabel="Policy-gradient updates on held-out task",
        ylabel="Evaluation return",
    )
    summary = {
        k: {
            "return_before": summarize(v[:, 0]),
            "return_after": summarize(v[:, -1]),
            "adaptation_gain": summarize(v[:, -1] - v[:, 0]),
        }
        for k, v in curves.items()
    }
    write_json(out_dir / "adaptation_summary.json", summary)
    return summary


def _parse_checkpoint(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        return Path(arg).parent.name, Path(arg)
    label, path = arg.split("=", 1)
    return label, Path(path)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compare meta-RL initialisations on held-out tasks.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="label=path/to/policy.pt (repeatable; a repeated label pools seeds)",
    )
    parser.add_argument("--include-random", action="store_true", help="Add a randomly initialised baseline.")
    parser.add_argument("--tasks", type=int, default=32, help="Number of held-out tasks.")
    parser.add_argument("--adapt-steps", type=int, default=None, help="Override inner_steps.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    groups: dict[str, list[nn.Module]] = {}
    cfg: MetaConfig | None = None
    for item in args.checkpoint:
        label, path = _parse_checkpoint(item)
        policy, ckpt_cfg = load_checkpoint(path, device)
        if cfg is not None and ckpt_cfg.env != cfg.env:
            raise ValueError("All checkpoints must come from the same task family.")
        cfg = cfg or ckpt_cfg
        groups.setdefault(label, []).append(policy)  # repeated labels = seeds of one method
    assert cfg is not None
    if args.adapt_steps is not None:
        cfg.inner_steps = args.adapt_steps
    if args.include_random:
        ref = next(iter(groups.values()))[0]
        torch.manual_seed(args.seed)
        groups["random-init"] = [PolicyNetwork(ref.obs_shape, ref.n_actions, cfg.hidden_sizes).to(device)]

    tasks = get_task_distribution(cfg.env, **cfg.env_kwargs())
    curves: dict[str, np.ndarray] = {}
    for label, members in groups.items():
        per_seed = [
            adaptation_curves({label: pol}, tasks, cfg, args.tasks, args.seed, device)[label]
            for pol in members
        ]
        curves[label] = np.concatenate(per_seed, axis=0)  # rows: (seed, task) pairs
    out = args.output or create_run_dir(cfg.output_root, "meta", cfg.env, "evaluation")
    write_report(curves, out, f"Adaptation on held-out {cfg.env} tasks")
    log.info("results written to %s", out)
    return out


if __name__ == "__main__":
    main()
