# scripts/train_ppo.py

from __future__ import annotations

import argparse

from trainers.ppo_trainer import train_ppo_from_config
from utils.config import build_experiment_config, build_logging_config, load_config
from utils.paths import create_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO satellite routing policy.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to PPO YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    experiment_cfg = build_experiment_config(cfg)
    logging_cfg = build_logging_config(cfg)

    run_dir = create_run_dir(
        output_root=logging_cfg.output_root,
        experiment_name=experiment_cfg.name,
        config=cfg,
    )

    train_ppo_from_config(cfg=cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()