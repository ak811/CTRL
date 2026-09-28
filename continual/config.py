"""Configuration for the sine-regression benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

Task = tuple[float, float]  # (amplitude, phase)


@dataclass(frozen=True)
class ContinualConfig:
    # ---- shared ------------------------------------------------------------------
    n_seeds: int = 3
    hidden_dim: int = 40
    hidden_layers: int = 2
    batch_size: int = 100
    lr: float = 1e-3
    x_range: tuple[float, float] = (-5.0, 5.0)
    amplitude_range: tuple[float, float] = (0.1, 5.0)
    phase_range: tuple[float, float] = (0.0, math.pi)
    eval_points: int = 200
    device: str = "cpu"
    output_root: str = "outputs"
    run_name: str | None = None

    # ---- supervised transfer (scratch / frozen / finetune) ------------------------
    transfer_steps: int = 5_000
    transfer_pairs: tuple[tuple[Task, Task], ...] = (
        ((1.0, 0.0), (2.0, math.pi / 4)),
        ((2.0, math.pi / 4), (0.5, math.pi / 2)),
        ((0.5, math.pi / 2), (3.5, 2.4)),
        ((1.5, math.pi / 3), (0.2, 0.1)),
    )

    # ---- MAML (Finn et al., 2017) --------------------------------------------------
    maml_iterations: int = 10_000
    maml_meta_batch: int = 10
    maml_k_shot: int = 10
    maml_inner_steps: int = 1
    maml_inner_lr: float = 0.01
    maml_meta_lr: float = 1e-3
    maml_first_order: bool = False
    maml_eval_tasks: int = 100
    maml_eval_steps: int = 10

    # ---- continual learning (EWC; Kirkpatrick et al., 2017) ------------------------
    cl_steps_per_task: int = 3_000
    cl_scenario: str = "task"  # "task" (multi-head) | "domain" (single head)
    ewc_lambda: float = 100.0
    ewc_mode: str = "separate"  # "separate" penalties | "online" running Fisher
    ewc_online_gamma: float = 1.0
    ewc_fisher_samples: int = 500
    task_params: tuple[Task, ...] = (
        (1.0, 0.0),
        (2.0, math.pi / 4),
        (0.5, math.pi / 2),
        (1.5, math.pi / 3),
        (0.8, math.pi / 6),
        (3.5, 2.4),
        (0.2, 0.1),
        (4.8, 0.0),
    )

    def __post_init__(self) -> None:
        if self.cl_scenario not in ("task", "domain"):
            raise ValueError("cl_scenario must be 'task' or 'domain'")
        if self.ewc_mode not in ("separate", "online"):
            raise ValueError("ewc_mode must be 'separate' or 'online'")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if not 0.0 < self.ewc_online_gamma <= 1.0:
            raise ValueError("ewc_online_gamma must be in (0, 1]")
