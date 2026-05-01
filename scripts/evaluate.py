from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from trainers.ppo_trainer import (
    PPOTrainer,
    build_env_from_config,
    build_model_from_config,
    choose_device,
    evaluate_policy,
)
from utils.checkpoint import load_checkpoint
from utils.config import build_eval_config, load_config
from utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO satellite routing policy.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation YAML config file.",
    )
    return parser.parse_args()


def infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    """
    Infer run directory from checkpoint path.

    Expected:
        outputs/runs/<run_name>/checkpoints/best.pt

    Returns:
        outputs/runs/<run_name>
    """
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent

    return checkpoint_path.parent


def save_eval_result(
    result: Dict[str, Any],
    checkpoint_path: Path,
    eval_config: Dict[str, Any],
) -> Path:
    """
    Save evaluation result next to the run directory.
    """
    run_dir = infer_run_dir_from_checkpoint(checkpoint_path)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    result_path = eval_dir / "eval_result.json"

    payload = {
        "checkpoint": str(checkpoint_path),
        "eval_config": eval_config,
        "result": result,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return result_path


def evaluate_from_config(eval_cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a trained PPO checkpoint from eval config.
    """
    eval_cfg = build_eval_config(eval_cfg_dict)

    checkpoint_path = resolve_project_path(eval_cfg.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    train_cfg = checkpoint.get("config")
    if not train_cfg:
        raise KeyError(
            "Checkpoint does not contain training config. "
            "Please use a checkpoint saved by utils.checkpoint.save_checkpoint."
        )

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    env = build_env_from_config(train_cfg)
    model = build_model_from_config(train_cfg, device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    ppo_cfg = train_cfg.get("ppo", {})

    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        lr=float(ppo_cfg.get("lr", 1e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        k_epochs=int(ppo_cfg.get("k_epochs", 1)),
        eps_clip=float(ppo_cfg.get("eps_clip", 0.2)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.01)),
        grad_clip=float(ppo_cfg.get("grad_clip", 1.0)),
    )

    result = evaluate_policy(
        trainer=trainer,
        num_episodes=eval_cfg.num_episodes,
        greedy=eval_cfg.greedy,
    )

    result_path = save_eval_result(
        result=result,
        checkpoint_path=checkpoint_path,
        eval_config=eval_cfg_dict,
    )

    print("Evaluation finished")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved eval result to: {result_path}")

    return result


def main() -> None:
    args = parse_args()
    eval_cfg_dict = load_config(args.config)
    evaluate_from_config(eval_cfg_dict)


if __name__ == "__main__":
    main()