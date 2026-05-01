# utils/metrics.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np


def summarize_episode_stats(episode_stats: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a list of episode-level stats.

    Expected keys in each episode stat:
        - success: bool
        - steps: int
        - coverage_ratio: float
        - total_delay: float
        - termination_reason: str
        - episode_return: float

    Optional keys:
        - repeat_count: int
        - repeat_ratio: float
    """
    stats: List[Dict[str, Any]] = list(episode_stats)

    if not stats:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "mean_coverage_ratio": 0.0,
            "mean_total_delay": 0.0,
            "mean_episode_return": 0.0,
            "mean_repeat_count": 0.0,
            "mean_repeat_ratio": 0.0,
            "termination_reasons": {},
        }

    success_values = [float(s.get("success", False)) for s in stats]
    step_values = [float(s.get("steps", 0)) for s in stats]
    coverage_values = [float(s.get("coverage_ratio", 0.0)) for s in stats]
    delay_values = [float(s.get("total_delay", 0.0)) for s in stats]
    return_values = [float(s.get("episode_return", 0.0)) for s in stats]
    repeat_counts = [float(s.get("repeat_count", 0.0)) for s in stats]
    repeat_ratios = [float(s.get("repeat_ratio", 0.0)) for s in stats]

    termination_reasons = Counter(
        str(s.get("termination_reason", "unknown")) for s in stats
    )

    return {
        "num_episodes": len(stats),
        "success_rate": float(np.mean(success_values)),
        "mean_steps": float(np.mean(step_values)),
        "mean_coverage_ratio": float(np.mean(coverage_values)),
        "mean_total_delay": float(np.mean(delay_values)),
        "mean_episode_return": float(np.mean(return_values)),
        "mean_repeat_count": float(np.mean(repeat_counts)),
        "mean_repeat_ratio": float(np.mean(repeat_ratios)),
        "termination_reasons": dict(termination_reasons),
    }


class MetricTracker:
    """
    Simple online metric tracker for scalar logs.
    """

    def __init__(self) -> None:
        self.values: Dict[str, List[float]] = {}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.values.setdefault(key, []).append(float(value))

    def mean(self, key: str, default: float = 0.0) -> float:
        values = self.values.get(key, [])
        if not values:
            return default
        return float(np.mean(values))

    def latest(self, key: str, default: float = 0.0) -> float:
        values = self.values.get(key, [])
        if not values:
            return default
        return float(values[-1])

    def summary(self) -> Dict[str, float]:
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in self.values.items()
        }

    def reset(self) -> None:
        self.values.clear()