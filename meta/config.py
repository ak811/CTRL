"""Configuration for meta-RL training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

ENVS = ("snake", "puckworld", "pong")
ALGORITHMS = ("reptile", "multitask")


@dataclass
class MetaConfig:
    env: str = "puckworld"  # task family: snake | puckworld | pong
    algorithm: str = "reptile"  # reptile | multitask (non-meta baseline)
    iterations: int = 300
    meta_batch_size: int = 4  # tasks per meta-iteration
    inner_steps: int = 5  # policy-gradient updates per task (the "k-shot" budget)
    episodes_per_step: int = 2  # episodes per policy-gradient update
    inner_lr: float = 1e-3
    inner_optimizer: str = "adam"  # adam | sgd
    meta_lr: float = 0.5  # Reptile outer step size (epsilon); Adam lr for multitask
    anneal_meta_lr: bool = True  # linear decay of the outer step size to 0
    gamma: float = 0.99
    entropy_coef: float = 0.01
    max_episode_steps: int = 300
    hidden_sizes: tuple[int, ...] = (128, 64)
    frame_stack: int = 4  # pong only
    eval_every: int = 50
    eval_tasks: int = 8
    eval_episodes: int = 3
    final_eval_tasks: int = 16
    seed: int = 0
    device: str = "cpu"
    output_root: str = "outputs"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.env not in ENVS:
            raise ValueError(f"env must be one of {ENVS}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if self.inner_optimizer not in ("adam", "sgd"):
            raise ValueError("inner_optimizer must be 'adam' or 'sgd'")
        for name in ("iterations", "meta_batch_size", "inner_steps", "episodes_per_step"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    def env_kwargs(self) -> dict:
        if self.env == "pong":
            return {"frame_stack": self.frame_stack}
        return {"max_steps": self.max_episode_steps}
