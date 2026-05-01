# scripts/evaluate_sl.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from env.rl_sat_env import RLSatelliteEnv
from models.policy_net import PolicyNet
from utils.checkpoint import load_checkpoint
from utils.config import (
    EnvConfig,
    RewardConfig,
    build_env_config,
    build_eval_config,
    build_model_config,
    build_reward_config,
    load_config,
)
from utils.metrics import summarize_episode_stats
from utils.paths import resolve_project_path


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate supervised PolicyNet.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SL evaluation YAML config file.",
    )
    return parser.parse_args()


def infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def build_sl_policy_from_checkpoint(
    checkpoint: Dict[str, Any],
    eval_cfg_dict: Dict[str, Any],
    device: torch.device,
) -> PolicyNet:
    """
    Build PolicyNet using model config saved in checkpoint.

    If checkpoint config does not contain model section, fall back to eval config.
    """
    train_cfg = checkpoint.get("config", {})
    model_source_cfg = train_cfg if "model" in train_cfg else eval_cfg_dict
    model_cfg = build_model_config(model_source_cfg)

    model = PolicyNet(
        input_dim=model_cfg.input_dim,
        hidden_dim=model_cfg.hidden_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model


@torch.no_grad()
def collect_sl_episode(
    env: RLSatelliteEnv,
    model: PolicyNet,
    device: torch.device,
    greedy: bool = True,
) -> Dict[str, Any]:
    """
    Run one SL policy episode.
    """
    state, info = env.reset()
    done = False

    rewards: List[float] = []

    while not done:
        adj_t = torch.tensor(
            state["adjacency"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        feat_t = torch.tensor(
            state["node_features"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        logits = model(adj_t, feat_t).squeeze(0)

        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        state, reward, done, info = env.step(int(action.item()))
        rewards.append(float(reward))

    episode_stats = {
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "coverage_ratio": float(info["coverage_ratio"]),
        "total_delay": float(info["total_delay"]),
        "termination_reason": info["termination_reason"],
        "episode_return": float(sum(rewards)),
        "repeat_count": int(info.get("repeat_count", 0)),
        "repeat_ratio": float(info.get("repeat_ratio", 0.0)),
    }

    return episode_stats


def save_eval_result(
    result: Dict[str, Any],
    checkpoint_path: Path,
    eval_config: Dict[str, Any],
) -> Path:
    run_dir = infer_run_dir_from_checkpoint(checkpoint_path)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    result_path = eval_dir / "sl_eval_result.json"

    payload = {
        "checkpoint": str(checkpoint_path),
        "eval_config": eval_config,
        "result": result,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return result_path


def evaluate_sl_from_config(eval_cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = build_eval_config(eval_cfg_dict)

    checkpoint_path = resolve_project_path(eval_cfg.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    device = choose_device()
    print(f"Using device: {device}")
    print(f"SL checkpoint: {checkpoint_path}")

    env_cfg: EnvConfig = build_env_config(eval_cfg_dict)
    reward_cfg: RewardConfig = build_reward_config(eval_cfg_dict)

    env = RLSatelliteEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
    )

    model = build_sl_policy_from_checkpoint(
        checkpoint=checkpoint,
        eval_cfg_dict=eval_cfg_dict,
        device=device,
    )

    episode_stats = []

    for _ in range(eval_cfg.num_episodes):
        stats = collect_sl_episode(
            env=env,
            model=model,
            device=device,
            greedy=eval_cfg.greedy,
        )
        episode_stats.append(stats)

    result = summarize_episode_stats(episode_stats)

    result_path = save_eval_result(
        result=result,
        checkpoint_path=checkpoint_path,
        eval_config=eval_cfg_dict,
    )

    print("SL evaluation finished")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved SL eval result to: {result_path}")

    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_sl_from_config(cfg)


if __name__ == "__main__":
    main()
