# utils/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    episode: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    The checkpoint always contains:
        - model_state_dict
        - config
        - metrics
        - epoch
        - episode

    It optionally contains:
        - optimizer_state_dict
        - extra
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {},
        "epoch": epoch,
        "episode": episode,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got: {type(checkpoint)}")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {path}")

    return checkpoint


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights from a checkpoint into an existing model.

    Returns the full checkpoint dictionary.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint