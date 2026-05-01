# scripts/generate_expert.py

from __future__ import annotations

import argparse

from data.generate_expert import generate_dataset_from_config
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate expert routing dataset.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to expert dataset YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    generate_dataset_from_config(cfg)


if __name__ == "__main__":
    main()