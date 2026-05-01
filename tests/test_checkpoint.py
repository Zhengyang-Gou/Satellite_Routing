from __future__ import annotations

import torch

from models.actor_critic_net import ActorCriticNet
from utils.checkpoint import load_checkpoint, load_model_weights, save_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    model = ActorCriticNet(input_dim=5, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    config = {
        "experiment": {
            "name": "test_checkpoint",
            "seed": 42,
        }
    }

    metrics = {
        "success_rate": 0.75,
        "mean_total_delay": 1.23,
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        metrics=metrics,
        episode=10,
    )

    checkpoint = load_checkpoint(checkpoint_path)

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "metrics" in checkpoint
    assert "episode" in checkpoint

    assert checkpoint["config"] == config
    assert checkpoint["metrics"] == metrics
    assert checkpoint["episode"] == 10


def test_load_model_weights(tmp_path):
    model = ActorCriticNet(input_dim=5, hidden_dim=64)

    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config={"experiment": {"name": "load_weights_test"}},
        metrics={"score": 1.0},
    )

    new_model = ActorCriticNet(input_dim=5, hidden_dim=64)

    checkpoint = load_model_weights(
        model=new_model,
        checkpoint_path=checkpoint_path,
        map_location="cpu",
        strict=True,
    )

    assert "model_state_dict" in checkpoint

    for key, value in model.state_dict().items():
        assert torch.equal(value, new_model.state_dict()[key])


def test_load_checkpoint_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.pt"

    try:
        load_checkpoint(missing_path)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_checkpoint_contains_extra_payload(tmp_path):
    model = ActorCriticNet(input_dim=5, hidden_dim=64)

    checkpoint_path = tmp_path / "extra.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config={"experiment": {"name": "extra_test"}},
        metrics={"loss": 0.5},
        epoch=3,
        extra={
            "note": "hello",
            "custom_value": 123,
        },
    )

    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert "extra" in checkpoint
    assert checkpoint["extra"]["note"] == "hello"
    assert checkpoint["extra"]["custom_value"] == 123
