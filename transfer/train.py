"""Train PPO on one environment, optionally initialised from a model trained on another.

    # 1) source task
    python -m transfer.train --env snake --timesteps 500000 --run-name snake_source
    # 2) target task from scratch vs. with a transferred encoder
    python -m transfer.train --env puckworld --timesteps 300000
    python -m transfer.train --env puckworld --timesteps 300000 \
        --init-from outputs/transfer/snake/snake_source/final_model.zip
    python -m transfer.train --env puckworld --timesteps 300000 \
        --init-from outputs/transfer/snake/snake_source/final_model.zip --freeze-encoder

Each run directory contains the resolved config, provenance metadata, evaluation curve
(``eval/evaluations.npz``, including an evaluation at t=0 for the jumpstart metric),
training-episode log, TensorBoard logs, the final model and the best checkpoint.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from common.config import parse_config
from common.io import create_run_dir, save_run_metadata, write_json
from common.logger import get_logger
from common.seed import set_seed
from common.torch_utils import count_parameters, resolve_device
from transfer.callbacks import EpisodeStatsCallback
from transfer.config import TransferConfig
from transfer.env_factory import make_vec_env
from transfer.extractor import NatureCNNEncoder
from transfer.weights import freeze_encoder, load_policy_state_dict, transfer_weights

log = get_logger("transfer")


class EvalAtStartCallback(EvalCallback):
    """``EvalCallback`` that also evaluates before the first update (t = 0).

    The t=0 evaluation is needed for the *jumpstart* transfer metric.
    """

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._on_step()  # n_calls == 0 -> the periodic condition is satisfied


def linear_schedule(initial: float) -> Callable[[float], float]:
    return lambda progress_remaining: progress_remaining * initial


def run_label(cfg: TransferConfig) -> str:
    if not cfg.init_from:
        return "scratch"
    return f"transfer-{cfg.transfer_layers}" + ("-frozen" if cfg.freeze_encoder else "")


def build_model(cfg: TransferConfig, env, run_dir: Path) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=NatureCNNEncoder,
        features_extractor_kwargs=dict(features_dim=cfg.features_dim),
        net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
        share_features_extractor=True,
    )
    use_tb = cfg.tensorboard and importlib.util.find_spec("tensorboard") is not None
    if cfg.tensorboard and not use_tb:
        log.warning("tensorboard not installed; disabling TensorBoard logging")
    lr = linear_schedule(cfg.learning_rate) if cfg.lr_schedule == "linear" else cfg.learning_rate
    return PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tb") if use_tb else None,
        seed=cfg.seed,
        device=str(resolve_device(cfg.device)),
        verbose=0,
    )


def train(cfg: TransferConfig) -> Path:
    set_seed(cfg.seed)
    label = run_label(cfg)
    run_dir = create_run_dir(
        cfg.output_root, "transfer", cfg.env, run_name=cfg.run_name or f"{label}_seed{cfg.seed}"
    )
    save_run_metadata(run_dir, cfg)
    get_logger("transfer", run_dir / "train.log")

    train_env = make_vec_env(cfg.env, cfg.n_envs, cfg.seed, cfg.frame_stack, cfg.vec_env)
    eval_env = make_vec_env(cfg.env, min(4, cfg.n_envs), cfg.seed + 10_000, cfg.frame_stack)
    model = build_model(cfg, train_env, run_dir)

    transfer_info: dict = {"label": label, "init_from": cfg.init_from}
    if cfg.init_from:
        keys = transfer_weights(model.policy, load_policy_state_dict(cfg.init_from), cfg.transfer_layers)
        transfer_info["transferred_tensors"] = len(keys)
        if cfg.freeze_encoder:
            transfer_info["frozen_tensors"] = freeze_encoder(model.policy)
        log.info("transferred %d tensors from %s (frozen=%s)", len(keys), cfg.init_from, cfg.freeze_encoder)
    transfer_info["trainable_parameters"] = count_parameters(model.policy)
    write_json(run_dir / "transfer.json", transfer_info)
    log.info(
        "env=%s label=%s trainable_params=%d timesteps=%d -> %s",
        cfg.env,
        label,
        transfer_info["trainable_parameters"],
        cfg.timesteps,
        run_dir,
    )

    callbacks = [
        EpisodeStatsCallback(run_dir),
        EvalAtStartCallback(
            eval_env,
            n_eval_episodes=cfg.eval_episodes,
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            log_path=str(run_dir / "eval"),
            best_model_save_path=str(run_dir / "best"),
            deterministic=cfg.eval_deterministic,
            verbose=0,
        ),
    ]
    if cfg.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(max(cfg.checkpoint_freq // cfg.n_envs, 1), str(run_dir / "checkpoints"), "ppo")
        )

    model.learn(total_timesteps=cfg.timesteps, callback=CallbackList(callbacks), tb_log_name=label)
    model.save(run_dir / "final_model.zip")

    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.eval_episodes, deterministic=cfg.eval_deterministic
    )
    evals = np.load(run_dir / "eval" / "evaluations.npz")
    summary = {
        **transfer_info,
        "env": cfg.env,
        "seed": cfg.seed,
        "timesteps": cfg.timesteps,
        "final_eval_mean": float(mean_r),
        "final_eval_std": float(std_r),
        "eval_timesteps": evals["timesteps"].tolist(),
        "eval_means": evals["results"].mean(axis=1).tolist(),
    }
    write_json(run_dir / "summary.json", summary)
    log.info("final evaluation: %.3f ± %.3f | outputs in %s", mean_r, std_r, run_dir)
    train_env.close()
    eval_env.close()
    return run_dir


def main(argv: Sequence[str] | None = None) -> Path:
    cfg, _ = parse_config(TransferConfig, "PPO training with optional cross-task weight transfer.", argv)
    return train(cfg)


if __name__ == "__main__":
    main()
