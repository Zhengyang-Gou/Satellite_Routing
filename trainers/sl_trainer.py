from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_sl import get_dataloader
from models.policy_net import PolicyNet
from utils.checkpoint import save_checkpoint
from utils.config import (
    ModelConfig,
    SLConfig,
    build_model_config,
    build_sl_config,
)
from utils.logger import CSVLogger
from utils.paths import (
    get_best_checkpoint_path,
    get_latest_checkpoint_path,
    get_metrics_path,
    resolve_project_path,
)
from utils.seed import set_seed


def choose_device() -> torch.device:
    """
    Choose training device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_policy_model_from_config(
    cfg: Dict[str, Any],
    device: torch.device,
) -> PolicyNet:
    """
    Build supervised policy model from config.
    """
    model_cfg: ModelConfig = build_model_config(cfg)

    model = PolicyNet(
        input_dim=model_cfg.input_dim,
        hidden_dim=model_cfg.hidden_dim,
    )

    return model.to(device)


def train_supervised_learning_from_config(
    cfg: Dict[str, Any],
    run_dir: str | Path,
) -> Path:
    """
    Supervised behavior cloning training entry.

    Args:
        cfg:
            Full config dictionary loaded from YAML.
        run_dir:
            Output directory created by utils.paths.create_run_dir.

    Returns:
        Path to best checkpoint.
    """
    run_dir = Path(run_dir)

    experiment_cfg = cfg.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    sl_cfg: SLConfig = build_sl_config(cfg)

    device = choose_device()
    print(f"Using device: {device}")

    train_path = resolve_project_path(sl_cfg.train_path)
    print(f"Training data path: {train_path}")

    dataloader = get_dataloader(
        data_path=str(train_path),
        batch_size=sl_cfg.batch_size,
        shuffle=True,
        num_workers=sl_cfg.num_workers,
        seed=seed,
    )

    model = build_policy_model_from_config(cfg, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=sl_cfg.lr)

    metrics_path = get_metrics_path(run_dir)
    logger = CSVLogger(metrics_path)

    best_loss = float("inf")
    best_checkpoint_path = get_best_checkpoint_path(run_dir)
    latest_checkpoint_path = get_latest_checkpoint_path(run_dir)

    print("Starting supervised learning")
    print(f"Run dir: {run_dir}")

    for epoch in range(1, sl_cfg.epochs + 1):
        model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for adj, features, target in dataloader:
            adj = adj.to(device)
            features = features.to(device)
            target = target.to(device)

            logits = model(adj, features)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = target.size(0)
            total_loss += float(loss.item()) * batch_size

            predictions = logits.argmax(dim=-1)
            correct_predictions += int((predictions == target).sum().item())
            total_samples += int(batch_size)

        if total_samples == 0:
            raise RuntimeError("No training samples were loaded from the dataset.")

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        log_row = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": total_samples,
        }
        logger.log(log_row)

        print(
            f"Epoch {epoch:03d}/{sl_cfg.epochs:03d} | "
            f"Loss {avg_loss:.6f} | "
            f"Accuracy {accuracy:.2%}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=cfg,
                metrics={
                    "best_loss": best_loss,
                    "accuracy": accuracy,
                    "total_samples": total_samples,
                },
                epoch=epoch,
            )

            print(f"Saved best checkpoint: {best_checkpoint_path}")

    save_checkpoint(
        path=latest_checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=cfg,
        metrics={
            "best_loss": best_loss,
            "final_loss": avg_loss,
            "final_accuracy": accuracy,
            "total_samples": total_samples,
        },
        epoch=sl_cfg.epochs,
    )

    logger.close()

    print("Supervised learning finished")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")

    return best_checkpoint_path