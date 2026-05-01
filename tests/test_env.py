from __future__ import annotations

import numpy as np

from env.rl_sat_env import RLSatelliteEnv
from env.sat_env import SatelliteEnv
from utils.config import EnvConfig, RewardConfig


def test_satellite_env_reset_shapes():
    env = SatelliteEnv(
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        seed=42,
    )

    obs = env.reset()

    assert obs["adjacency"].shape == (60, 60)
    assert obs["node_features"].shape == (60, 5)
    assert isinstance(obs["current_node"], int)
    assert len(obs["visited"]) == 1
    assert isinstance(obs["time"], float)

    assert obs["adjacency"].dtype == np.float32
    assert obs["node_features"].dtype == np.float32


def test_satellite_env_valid_step():
    env = SatelliteEnv(
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        seed=42,
    )

    obs = env.reset()
    current = obs["current_node"]

    valid_actions = list(env.graph.neighbors(current))
    assert len(valid_actions) > 0

    next_obs, reward, done, info = env.step(valid_actions[0])

    assert next_obs["adjacency"].shape == (60, 60)
    assert next_obs["node_features"].shape == (60, 5)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "time" in info
    assert "total_delay" in info


def test_rl_env_reset_returns_obs_and_info():
    env_cfg = EnvConfig(
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        max_steps_factor=3,
        seed=42,
    )
    reward_cfg = RewardConfig()

    env = RLSatelliteEnv(env_cfg=env_cfg, reward_cfg=reward_cfg)

    obs, info = env.reset()

    assert obs["adjacency"].shape == (60, 60)
    assert obs["node_features"].shape == (60, 5)

    assert info["step_count"] == 0
    assert info["coverage_count"] == 1
    assert info["coverage_ratio"] == 1 / 60
    assert info["repeat_count"] == 0
    assert info["repeat_ratio"] == 0.0
    assert info["action_mask"].shape == (60,)
    assert info["valid_action_count"] > 0


def test_rl_env_valid_step():
    env_cfg = EnvConfig(
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        max_steps_factor=3,
        seed=42,
    )
    reward_cfg = RewardConfig()

    env = RLSatelliteEnv(env_cfg=env_cfg, reward_cfg=reward_cfg)

    obs, info = env.reset()

    valid_actions = [
        idx for idx, value in enumerate(info["action_mask"])
        if value > 0
    ]

    assert len(valid_actions) > 0

    next_obs, reward, done, next_info = env.step(valid_actions[0])

    assert next_obs["adjacency"].shape == (60, 60)
    assert next_obs["node_features"].shape == (60, 5)

    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert next_info["step_count"] == 1
    assert next_info["coverage_count"] >= 1
    assert 0.0 <= next_info["coverage_ratio"] <= 1.0
    assert next_info["repeat_count"] >= 0
    assert 0.0 <= next_info["repeat_ratio"] <= 1.0


def test_rl_env_invalid_action_terminates():
    env_cfg = EnvConfig(
        num_planes=6,
        sats_per_plane=10,
        failure_prob=0.0,
        max_link_distance=10000e3,
        max_steps_factor=3,
        seed=42,
    )
    reward_cfg = RewardConfig(fail=-5.0)

    env = RLSatelliteEnv(env_cfg=env_cfg, reward_cfg=reward_cfg)

    obs, info = env.reset()

    valid_actions = {
        idx for idx, value in enumerate(info["action_mask"])
        if value > 0
    }

    invalid_actions = [
        idx for idx in range(env.num_satellites)
        if idx not in valid_actions and idx != obs["current_node"]
    ]

    assert len(invalid_actions) > 0

    _, reward, done, next_info = env.step(invalid_actions[0])

    assert done is True
    assert reward == reward_cfg.fail
    assert next_info["success"] is False
    assert next_info["termination_reason"] == "invalid_action"
