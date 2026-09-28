"""PPO / transfer configuration."""

from __future__ import annotations

from dataclasses import dataclass

ENVS = ("pong", "snake", "puckworld")
TRANSFER_LAYERS = ("encoder", "encoder_mlp")


@dataclass
class TransferConfig:
    env: str = "pong"
    timesteps: int = 1_000_000
    seed: int = 0

    # ---- environment -------------------------------------------------------------
    n_envs: int = 8
    frame_stack: int = 4
    vec_env: str = "dummy"  # dummy | subproc

    # ---- PPO (Schulman et al., 2017) ----------------------------------------------
    learning_rate: float = 2.5e-4
    lr_schedule: str = "linear"  # constant | linear
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    features_dim: int = 512
    net_arch: tuple[int, ...] = (256,)

    # ---- transfer ----------------------------------------------------------------
    init_from: str | None = None  # path to a source model .zip
    transfer_layers: str = "encoder"  # encoder | encoder_mlp
    freeze_encoder: bool = False

    # ---- evaluation / logging ------------------------------------------------------
    eval_freq: int = 25_000  # in total environment steps
    eval_episodes: int = 10
    eval_deterministic: bool = True
    checkpoint_freq: int = 0  # 0 disables periodic checkpoints
    tensorboard: bool = True
    device: str = "auto"
    output_root: str = "outputs"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.env not in ENVS:
            raise ValueError(f"env must be one of {ENVS}")
        if self.transfer_layers not in TRANSFER_LAYERS:
            raise ValueError(f"transfer_layers must be one of {TRANSFER_LAYERS}")
        if self.lr_schedule not in ("constant", "linear"):
            raise ValueError("lr_schedule must be 'constant' or 'linear'")
        if self.vec_env not in ("dummy", "subproc"):
            raise ValueError("vec_env must be 'dummy' or 'subproc'")
        rollout = self.n_steps * self.n_envs
        if rollout % self.batch_size != 0:
            raise ValueError(
                f"n_steps * n_envs ({rollout}) must be divisible by batch_size ({self.batch_size})"
            )
        if self.freeze_encoder and not self.init_from:
            raise ValueError(
                "freeze_encoder requires init_from (freezing a random encoder is a no-op baseline)"
            )
