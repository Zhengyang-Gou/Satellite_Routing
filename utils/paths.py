# utils/paths.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from utils.config import save_config


def get_project_root() -> Path:
    """
    Return project root directory.

    Assumption:
        utils/paths.py is located at:
            <project_root>/utils/paths.py
    """
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """
    Resolve a path relative to project root.

    Absolute paths are returned unchanged.
    Relative paths are interpreted from project root.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def make_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_dir(
    output_root: str | Path,
    experiment_name: str,
    config: Dict[str, Any],
) -> Path:
    """
    Create a timestamped run directory.

    Example:
        outputs/runs/20260425_153012_ppo_warm_start/

    Structure:
        run_dir/
          config.yaml
          checkpoints/
          logs/
          plots/
          eval/
    """
    output_root = resolve_project_path(output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{experiment_name}"
    run_dir = output_root / run_name

    make_dir(run_dir)
    make_dir(run_dir / "checkpoints")
    make_dir(run_dir / "logs")
    make_dir(run_dir / "plots")
    make_dir(run_dir / "eval")

    save_config(config, run_dir / "config.yaml")

    return run_dir


def get_checkpoints_dir(run_dir: str | Path) -> Path:
    return make_dir(Path(run_dir) / "checkpoints")


def get_logs_dir(run_dir: str | Path) -> Path:
    return make_dir(Path(run_dir) / "logs")


def get_plots_dir(run_dir: str | Path) -> Path:
    return make_dir(Path(run_dir) / "plots")


def get_eval_dir(run_dir: str | Path) -> Path:
    return make_dir(Path(run_dir) / "eval")


def get_best_checkpoint_path(run_dir: str | Path) -> Path:
    return get_checkpoints_dir(run_dir) / "best.pt"


def get_latest_checkpoint_path(run_dir: str | Path) -> Path:
    return get_checkpoints_dir(run_dir) / "latest.pt"


def get_metrics_path(run_dir: str | Path) -> Path:
    return get_logs_dir(run_dir) / "metrics.csv"


def get_eval_result_path(run_dir: str | Path) -> Path:
    return get_eval_dir(run_dir) / "eval_result.json"
