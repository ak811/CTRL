"""Evaluate a trained PPO model and optionally record a video.

python -m transfer.evaluate --model outputs/transfer/snake/<run>/final_model.zip --episodes 20
python -m transfer.evaluate --model <run>/best/best_model.zip --video-dir videos/  # needs [video] extra
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from common.config import load_yaml
from common.io import write_json
from common.logger import get_logger
from common.stats import summarize
from transfer.config import ENVS
from transfer.env_factory import make_vec_env

log = get_logger("transfer.evaluate")


def _infer_env(model_path: Path) -> str | None:
    for parent in (model_path.parent, model_path.parent.parent):
        cfg = parent / "config.yaml"
        if cfg.exists():
            return load_yaml(cfg).get("env")
    return None


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--env", choices=ENVS, default=None, help="Inferred from the run's config.yaml if omitted."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-length", type=int, default=2000)
    args = parser.parse_args(argv)

    env_name = args.env or _infer_env(args.model)
    if env_name is None:
        parser.error("--env is required when the run's config.yaml cannot be found")
    model = PPO.load(args.model, device="cpu")
    frame_stack = int(model.observation_space.shape[0])
    venv = make_vec_env(env_name, 1, args.seed, frame_stack, render_mode="rgb_array")

    if args.video_dir is not None:
        try:
            from stable_baselines3.common.vec_env import VecVideoRecorder

            venv = VecVideoRecorder(
                venv,
                str(args.video_dir),
                record_video_trigger=lambda step: step == 0,
                video_length=args.video_length,
                name_prefix=f"{env_name}_ppo",
            )
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Video recording requires `pip install -e .[video]`") from exc

    returns, lengths = evaluate_policy(
        model,
        venv,
        n_eval_episodes=args.episodes,
        deterministic=not args.stochastic,
        return_episode_rewards=True,
    )
    venv.close()
    result = {
        "env": env_name,
        "model": str(args.model),
        "returns": summarize(returns),
        "lengths": summarize(lengths),
    }
    log.info(
        "%s: return %.3f (95%% CI %.3f..%.3f) over %d episodes | mean length %.0f",
        env_name,
        result["returns"]["mean"],
        result["returns"]["ci_low"],
        result["returns"]["ci_high"],
        args.episodes,
        float(np.mean(lengths)),
    )
    write_json(args.model.with_suffix(".eval.json"), result)
    return result


if __name__ == "__main__":
    main()
