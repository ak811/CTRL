from __future__ import annotations

import pytest
import torch

from transfer.config import TransferConfig
from transfer.env_factory import make_vec_env
from transfer.train import build_model, train
from transfer.weights import freeze_encoder, load_policy_state_dict, transfer_weights

SMALL = dict(
    n_envs=2, n_steps=64, batch_size=64, n_epochs=1, eval_freq=128, eval_episodes=1, tensorboard=False
)


def test_vec_env_is_channels_first_stack():
    """Regression test: VecTransposeImage used to crash on (1, 84, 84) observations."""
    venv = make_vec_env("snake", n_envs=2, seed=0, frame_stack=4)
    assert venv.observation_space.shape == (4, 84, 84)
    assert venv.reset().shape == (2, 4, 84, 84)
    venv.close()


def test_config_validation():
    with pytest.raises(ValueError):
        TransferConfig(n_steps=100, n_envs=1, batch_size=64)
    with pytest.raises(ValueError):
        TransferConfig(freeze_encoder=True)


def test_cross_domain_transfer_and_freeze(tmp_path):
    src_dir = train(TransferConfig(env="snake", timesteps=128, output_root=str(tmp_path), **SMALL))
    src_state = load_policy_state_dict(src_dir / "final_model.zip")

    # Discrete(4) -> Box(2): encoder transfers, action heads do not.
    cfg = TransferConfig(env="puckworld", timesteps=128, output_root=str(tmp_path), **SMALL)
    venv = make_vec_env("puckworld", 2, 0, 4)
    model = build_model(cfg, venv, tmp_path)
    keys = transfer_weights(model.policy, src_state, "encoder")
    assert keys and all("action_net" not in k for k in keys)
    sd = model.policy.state_dict()
    for k in keys:
        torch.testing.assert_close(sd[k], src_state[k])
    assert freeze_encoder(model.policy) > 0
    assert not any(p.requires_grad for p in model.policy.features_extractor.parameters())
    venv.close()


def test_frozen_encoder_unchanged_after_training(tmp_path):
    src_dir = train(
        TransferConfig(env="snake", timesteps=128, output_root=str(tmp_path), run_name="src", **SMALL)
    )
    run = train(
        TransferConfig(
            env="puckworld",
            timesteps=256,
            output_root=str(tmp_path),
            init_from=str(src_dir / "final_model.zip"),
            freeze_encoder=True,
            **SMALL,
        )
    )
    src = load_policy_state_dict(src_dir / "final_model.zip")
    tgt = load_policy_state_dict(run / "final_model.zip")
    for k in src:
        if k.startswith("features_extractor."):
            torch.testing.assert_close(src[k], tgt[k])
    assert (run / "eval" / "evaluations.npz").exists()
