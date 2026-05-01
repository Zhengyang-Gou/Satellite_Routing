# env/rl_sat_env.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from env.sat_env import SatelliteEnv
from utils.config import EnvConfig, RewardConfig


class RLSatelliteEnv(SatelliteEnv):
    """
    RL version of SatelliteEnv.

    Compared with SatelliteEnv:
        - Invalid actions do not raise exceptions; they terminate the episode.
        - Step limit is enforced.
        - Dead-end states terminate the episode.
        - Reward is configurable through RewardConfig.
        - Repeat visits are tracked for evaluation.
    """

    def __init__(self, env_cfg: EnvConfig, reward_cfg: RewardConfig):
        super().__init__(
            num_planes=env_cfg.num_planes,
            sats_per_plane=env_cfg.sats_per_plane,
            failure_prob=env_cfg.failure_prob,
            max_link_distance=env_cfg.max_link_distance,
            seed=env_cfg.seed,
        )

        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg

        self.max_steps = env_cfg.max_steps_factor * self.num_satellites

        self.step_count = 0
        self.repeat_count = 0

    def _get_valid_actions(self) -> List[int]:
        """
        Return valid next-hop actions from the current node.
        """
        if self.graph is None or self.current_node is None:
            return []

        return list(self.graph.neighbors(self.current_node))

    def _get_action_mask(self) -> np.ndarray:
        """
        Return action mask with shape [num_satellites].

        mask[i] = 1.0 means action i is currently legal.
        mask[i] = 0.0 means action i is illegal.
        """
        mask = np.zeros(self.num_satellites, dtype=np.float32)

        for action in self._get_valid_actions():
            mask[action] = 1.0

        return mask

    def _build_info(
        self,
        *,
        delay: Optional[float],
        is_new_visit: bool,
        success: bool,
        termination_reason: Optional[str],
    ) -> Dict:
        """
        Build info dict returned by reset() and step().
        """
        valid_action_count = len(self._get_valid_actions())

        repeat_ratio = (
            self.repeat_count / self.step_count
            if self.step_count > 0
            else 0.0
        )

        return {
            "time": self.time,
            "total_delay": self.total_delay,
            "step_count": self.step_count,
            "coverage_count": len(self.visited) if self.visited is not None else 0,
            "coverage_ratio": (
                len(self.visited) / self.num_satellites
                if self.visited is not None
                else 0.0
            ),
            "valid_action_count": valid_action_count,
            "action_mask": self._get_action_mask(),
            "is_new_visit": is_new_visit,
            "delay": delay,
            "success": success,
            "termination_reason": termination_reason,
            "repeat_count": self.repeat_count,
            "repeat_ratio": repeat_ratio,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment.

        Args:
            seed:
                Optional episode seed. Forwarded to SatelliteEnv.reset(seed=...).

        Returns:
            obs, info
        """
        obs = super().reset(seed=seed)

        self.step_count = 0
        self.repeat_count = 0

        info = self._build_info(
            delay=None,
            is_new_visit=False,
            success=False,
            termination_reason=None,
        )

        return obs, info

    def step(self, action: int):
        """
        Execute one RL step.

        Invalid action:
            terminate with fail reward.

        Valid action:
            move to next node, update dynamic topology, compute reward.
        """
        done = False
        success = False
        termination_reason = None
        delay = None
        is_new_visit = False

        valid_actions = self._get_valid_actions()

        # 1. Current node has no valid outgoing edge.
        if len(valid_actions) == 0:
            done = True
            termination_reason = "dead_end_no_valid_actions"
            reward = self.reward_cfg.fail

            obs = self._get_state()
            info = self._build_info(
                delay=None,
                is_new_visit=False,
                success=False,
                termination_reason=termination_reason,
            )
            return obs, reward, done, info

        # 2. Invalid action terminates the episode.
        if action not in valid_actions:
            done = True
            termination_reason = "invalid_action"
            reward = self.reward_cfg.fail

            obs = self._get_state()
            info = self._build_info(
                delay=None,
                is_new_visit=False,
                success=False,
                termination_reason=termination_reason,
            )
            return obs, reward, done, info

        # 3. Valid transition.
        old_visited_count = len(self.visited)
        delay = float(self.graph[self.current_node][action]["delay"])

        self.total_delay += delay
        self.current_node = action
        self.visited.add(action)
        self.time += delay
        self.step_count += 1

        is_new_visit = len(self.visited) > old_visited_count

        if not is_new_visit:
            self.repeat_count += 1

        # Dynamic topology update after time advances.
        self._update_environment()

        # 4. Success termination.
        if len(self.visited) == self.num_satellites:
            done = True
            success = True
            termination_reason = "all_nodes_visited"

        # 5. Failure termination.
        if not done and self.step_count >= self.max_steps:
            done = True
            termination_reason = "step_limit_exceeded"

        if not done and len(self._get_valid_actions()) == 0:
            done = True
            termination_reason = "dead_end_no_valid_actions"

        # 6. Configurable reward.
        reward = -self.reward_cfg.delay_scale * delay
        reward += self.reward_cfg.new_node if is_new_visit else self.reward_cfg.repeat

        if done and success:
            reward += self.reward_cfg.success

        if done and not success:
            reward += self.reward_cfg.fail

        obs = self._get_state()
        info = self._build_info(
            delay=delay,
            is_new_visit=is_new_visit,
            success=success,
            termination_reason=termination_reason,
        )

        return obs, reward, done, info