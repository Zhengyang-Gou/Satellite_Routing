from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# =========================
# Dataclass config schemas
# =========================

@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42


@dataclass
class EnvConfig:
    num_planes: int = 6
    sats_per_plane: int = 10
    failure_prob: float = 0.05
    max_link_distance: float = 10000e3
    max_steps_factor: int = 3
    seed: int = 42


@dataclass
class RewardConfig:
    delay_scale: float = 1.0
    new_node: float = 0.2
    repeat: float = -0.1
    fail: float = -5.0
    success: float = 10.0


@dataclass
class ModelConfig:
    input_dim: int = 5
    hidden_dim: int = 64


@dataclass
class ExpertConfig:
    policy: str = "greedy_shortest_unvisited"
    num_episodes: int = 5000
    chunk_episode_size: int = 100
    save_dir: str = "outputs/datasets/expert_greedy_clean"


@dataclass
class SLConfig:
    train_path: str = "outputs/datasets/expert_greedy_clean"
    epochs: int = 25
    batch_size: int = 512
    lr: float = 2e-3
    num_workers: int = 0


@dataclass
class PPOConfig:
    pretrained_checkpoint: Optional[str] = None
    num_episodes: int = 4000
    eval_every: int = 100

    lr: float = 1e-4
    gamma: float = 0.99
    k_epochs: int = 10
    eps_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 1.0


@dataclass
class LoggingConfig:
    output_root: str = "outputs/runs"
    save_latest_every: int = 100
    save_plot_every: int = 10


@dataclass
class EvalConfig:
    checkpoint: str
    num_episodes: int = 100
    greedy: bool = True
    save_rollouts: bool = False


# =========================
# YAML utilities
# =========================

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file as a plain dictionary.

    The project keeps configs as dictionaries at runtime to make saving,
    checkpointing, and JSON/YAML serialization simple.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(cfg, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {config_path}")

    return cfg


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    """
    Save a config dictionary to YAML.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


# =========================
# Section helpers
# =========================

def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Return a required config section.

    Raises a clear error if the section is missing.
    """
    if section not in cfg:
        raise KeyError(f"Missing required config section: '{section}'")

    value = cfg[section]

    if not isinstance(value, dict):
        raise TypeError(f"Config section '{section}' must be a mapping")

    return value


def optional_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Return an optional config section.

    Missing sections become empty dictionaries.
    """
    value = cfg.get(section, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(f"Config section '{section}' must be a mapping")

    return value


# =========================
# Builder helpers
# =========================

def build_experiment_config(cfg: Dict[str, Any]) -> ExperimentConfig:
    section = require_section(cfg, "experiment")
    return ExperimentConfig(**section)


def build_env_config(cfg: Dict[str, Any]) -> EnvConfig:
    env_section = require_section(cfg, "env")
    experiment_section = require_section(cfg, "experiment")

    merged = dict(env_section)
    merged["seed"] = experiment_section.get("seed", merged.get("seed", 42))

    return EnvConfig(**merged)


def build_reward_config(cfg: Dict[str, Any]) -> RewardConfig:
    section = require_section(cfg, "reward")
    return RewardConfig(**section)


def build_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    section = require_section(cfg, "model")
    return ModelConfig(**section)


def build_expert_config(cfg: Dict[str, Any]) -> ExpertConfig:
    section = require_section(cfg, "expert")
    return ExpertConfig(**section)


def build_sl_config(cfg: Dict[str, Any]) -> SLConfig:
    section = require_section(cfg, "sl")
    return SLConfig(**section)


def build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    section = require_section(cfg, "ppo")
    return PPOConfig(**section)


def build_logging_config(cfg: Dict[str, Any]) -> LoggingConfig:
    section = optional_section(cfg, "logging")
    return LoggingConfig(**section)


def build_eval_config(cfg: Dict[str, Any]) -> EvalConfig:
    section = require_section(cfg, "eval")
    return EvalConfig(**section)