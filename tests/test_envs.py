from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import make_env
from envs.snake import SnakeGridEnv
from envs.tasks import TEST_PERMUTATIONS, TRAIN_PERMUTATIONS, get_task_distribution
from tests.conftest import requires_ale

LIGHT_ENVS = ["snake", "snake_image", "puckworld", "puckworld_image"]


@pytest.mark.parametrize("key", LIGHT_ENVS)
def test_gymnasium_api_compliance(key):
    check_env(make_env(key), skip_render_check=True)


@pytest.mark.parametrize("key", LIGHT_ENVS)
def test_seeded_reset_and_rollout_are_deterministic(key):
    def rollout():
        env = make_env(key)
        obs, _ = env.reset(seed=123)
        env.action_space.seed(0)
        traj = [obs]
        for _ in range(20):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            traj.append(obs)
            if term or trunc:
                break
        return np.stack(traj)

    np.testing.assert_array_equal(rollout(), rollout())


@requires_ale
@pytest.mark.parametrize("key,shape", [("pong_stack", (4, 40, 40)), ("pong_image", (1, 84, 84))])
def test_pong_wrappers(key, shape):
    env = make_env(key)
    obs, _ = env.reset(seed=0)
    assert obs.shape == shape and env.observation_space.contains(obs)
    obs, *_ = env.step(1)
    assert obs.shape == shape
    env.close()


def test_snake_can_move_into_vacating_tail():
    env = SnakeGridEnv(grid_size=6)
    env.reset(seed=0)
    # 2x2 loop: head at (2,2) moving into the cell the tail is about to vacate.
    env._snake.clear()
    env._snake.extend([(2, 2), (3, 2), (3, 3), (2, 3)])
    env._food = (0, 0)
    _, reward, terminated, _, _ = env.step(3)  # right -> (2,3) == current tail
    assert not terminated and reward == pytest.approx(env.step_reward)


def test_snake_wall_collision_terminates():
    env = SnakeGridEnv(grid_size=4)
    env.reset(seed=0)
    terminated = False
    for _ in range(4):
        _, reward, terminated, _, _ = env.step(0)  # keep moving up
        if terminated:
            break
    assert terminated and reward == env.death_penalty


def test_snake_head_is_distinguishable():
    env = SnakeGridEnv()
    obs, _ = env.reset(seed=0)
    assert (obs == SnakeGridEnv.HEAD).sum() == 1


def test_action_permutation_changes_semantics():
    a = SnakeGridEnv(action_permutation=(1, 0, 2, 3))
    b = SnakeGridEnv()
    a.reset(seed=0)
    b.reset(seed=0)
    oa, *_ = a.step(0)  # permuted: "up" behaves as "down"
    ob, *_ = b.step(1)
    np.testing.assert_array_equal(oa, ob)


def test_puckworld_reaches_goal_and_terminates():
    env = make_env("puckworld", thrust=0.5, max_steps=2000)
    env.reset(seed=0)
    env.position[:] = [0.5, 0.5]
    env.target[:] = [0.9, 0.5]
    total, terminated = 0.0, False
    for _ in range(2000):
        _, r, terminated, truncated, _ = env.step(3)  # thrust east
        total += r
        if terminated or truncated:
            break
    assert terminated and total > env.success_bonus


def test_task_splits_are_disjoint(rng):
    assert set(TRAIN_PERMUTATIONS).isdisjoint(TEST_PERMUTATIONS)
    assert len(TRAIN_PERMUTATIONS) + len(TEST_PERMUTATIONS) == 24
    for family in ("snake", "puckworld"):
        dist = get_task_distribution(family)
        train = {dist.sample(rng, "train").params["action_permutation"] for _ in range(200)}
        test = {dist.sample(rng, "test").params["action_permutation"] for _ in range(200)}
        assert train.isdisjoint(test)
        assert dist.make(dist.sample(rng)).observation_space.shape
