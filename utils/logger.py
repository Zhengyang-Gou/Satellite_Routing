# utils/logger.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class CSVLogger:
    """
    Simple CSV logger.

    Example:
        logger = CSVLogger("outputs/runs/demo/logs/metrics.csv")
        logger.log({"episode": 1, "loss": 0.5})
        logger.log({"episode": 2, "loss": 0.4})
        logger.close()
    """

    def __init__(self, path: str | Path, fieldnames: Optional[Iterable[str]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames else None
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer: Optional[csv.DictWriter] = None

        if self.fieldnames is not None:
            self._init_writer(self.fieldnames)

    def _init_writer(self, fieldnames: List[str]) -> None:
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, Any]) -> None:
        """
        Write one row.

        If fieldnames were not provided, they are inferred from the first row.
        Later rows may contain extra keys; extra keys are ignored.
        """
        if self.writer is None:
            self.fieldnames = list(row.keys())
            self._init_writer(self.fieldnames)

        assert self.writer is not None
        self.writer.writerow(row)
        self.file.flush()

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.log(row)

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PPOLogger:
    """
    Backward-compatible PPO logger.

    This keeps the old trainer easy to adapt:
        logger.log_step(...)
    """

    def __init__(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = save_dir / "metrics.csv"
        self.csv = CSVLogger(self.metrics_path)

    def log_step(
        self,
        episode: int,
        ep_return: float,
        success: bool,
        p_loss: float,
        v_loss: float,
        entropy: float,
        **extra: Any,
    ) -> None:
        row = {
            "episode": episode,
            "episode_return": ep_return,
            "success": int(success),
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "entropy": entropy,
        }
        row.update(extra)
        self.csv.log(row)

    def save_plots(self, filename: str = "training_metrics.png") -> None:
        """
        Placeholder for compatibility.

        Plotting will be added later after the training loop is stabilized.
        """
        return None

    def close(self) -> None:
        self.csv.close()