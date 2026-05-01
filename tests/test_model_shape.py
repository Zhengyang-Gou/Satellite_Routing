from __future__ import annotations

import torch

from models.actor_critic_net import ActorCriticNet
from models.gnn_encoder import GNNEncoder
from models.policy_net import PolicyNet


def build_dummy_inputs(batch_size: int = 2, num_nodes: int = 60):
    adj = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.float32)

    node_features = torch.zeros(batch_size, num_nodes, 5, dtype=torch.float32)

    # Mark node 0 as current node.
    node_features[:, 0, 3] = 1.0

    # Mark node 0 as visited.
    node_features[:, 0, 2] = 1.0

    return adj, node_features


def test_gnn_encoder_output_shape():
    adj, node_features = build_dummy_inputs()

    encoder = GNNEncoder(
        input_dim=5,
        hidden_dim=64,
        num_layers=3,
    )

    out = encoder(adj, node_features)

    assert out.shape == (2, 60, 64)


def test_policy_net_output_shape():
    adj, node_features = build_dummy_inputs()

    model = PolicyNet(
        input_dim=5,
        hidden_dim=64,
    )

    logits = model(adj, node_features)

    assert logits.shape == (2, 60)


def test_actor_critic_output_shape():
    adj, node_features = build_dummy_inputs()

    model = ActorCriticNet(
        input_dim=5,
        hidden_dim=64,
    )

    logits, value = model(adj, node_features)

    assert logits.shape == (2, 60)
    assert value.shape == (2,)


def test_policy_net_masks_invalid_actions():
    batch_size = 2
    num_nodes = 60

    adj = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.float32)

    # Current node is 0.
    # Only action 1 is valid.
    adj[:, 0, 1] = 1.0

    node_features = torch.zeros(batch_size, num_nodes, 5, dtype=torch.float32)
    node_features[:, 0, 3] = 1.0
    node_features[:, 0, 2] = 1.0

    model = PolicyNet(
        input_dim=5,
        hidden_dim=64,
    )

    logits = model(adj, node_features)

    assert logits.shape == (batch_size, num_nodes)

    # Valid action should not be masked.
    assert torch.all(logits[:, 1] > -1e8)

    # Invalid actions should be heavily masked.
    invalid_indices = [idx for idx in range(num_nodes) if idx != 1]
    assert torch.all(logits[:, invalid_indices] < -1e8)


def test_actor_critic_masks_invalid_actions():
    batch_size = 2
    num_nodes = 60

    adj = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.float32)

    # Current node is 0.
    # Only action 1 is valid.
    adj[:, 0, 1] = 1.0

    node_features = torch.zeros(batch_size, num_nodes, 5, dtype=torch.float32)
    node_features[:, 0, 3] = 1.0
    node_features[:, 0, 2] = 1.0

    model = ActorCriticNet(
        input_dim=5,
        hidden_dim=64,
    )

    logits, value = model(adj, node_features)

    assert logits.shape == (batch_size, num_nodes)
    assert value.shape == (batch_size,)

    # Valid action should not be masked.
    assert torch.all(logits[:, 1] > -1e8)

    # Invalid actions should be heavily masked.
    invalid_indices = [idx for idx in range(num_nodes) if idx != 1]
    assert torch.all(logits[:, invalid_indices] < -1e8)
