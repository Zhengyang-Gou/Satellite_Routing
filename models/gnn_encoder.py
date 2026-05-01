from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGCNLayer(nn.Module):
    """
    Dense GCN layer.

    Operation:
        H = A_norm @ Linear(X)

    Args:
        in_features:
            Input node feature dimension.

        out_features:
            Output node embedding dimension.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Node features with shape [B, N, in_features].

            adj_norm:
                Normalized adjacency matrix with shape [B, N, N].

        Returns:
            out:
                Node embeddings with shape [B, N, out_features].
        """
        h = self.linear(x)
        out = torch.bmm(adj_norm, h)

        return out


class GNNEncoder(nn.Module):
    """
    GNN encoder for satellite topology.

    It uses stacked dense GCN layers to encode graph-structured satellite states.

    Default:
        3 GCN layers.

    Interpretation:
        With 3 layers, each node representation can aggregate information from
        nodes up to roughly 3 hops away in the current topology.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got: {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []

        layers.append(DenseGCNLayer(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            layers.append(DenseGCNLayer(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        Row-normalize adjacency matrix with self-loops.

        Args:
            adj:
                Binary adjacency matrix with shape [B, N, N].

        Returns:
            adj_norm:
                Row-normalized adjacency matrix with shape [B, N, N].
        """
        if adj.dim() != 3:
            raise ValueError(f"adj must have shape [B, N, N], got: {tuple(adj.shape)}")

        batch_size, n_nodes, n_nodes_2 = adj.shape

        if n_nodes != n_nodes_2:
            raise ValueError(f"adj must be square, got: {tuple(adj.shape)}")

        eye = torch.eye(
            n_nodes,
            device=adj.device,
            dtype=adj.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        adj_hat = adj + eye

        degree = adj_hat.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        adj_norm = adj_hat / degree

        return adj_norm

    def forward(
        self,
        adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            adj:
                Binary adjacency matrix with shape [B, N, N].

            node_features:
                Node features with shape [B, N, input_dim].

        Returns:
            node_embeddings:
                Node embeddings with shape [B, N, hidden_dim].
        """
        if node_features.dim() != 3:
            raise ValueError(
                f"node_features must have shape [B, N, D], "
                f"got: {tuple(node_features.shape)}"
            )

        if node_features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected node feature dim {self.input_dim}, "
                f"got {node_features.size(-1)}"
            )

        adj_norm = self.normalize_adj(adj)

        x = node_features

        for layer in self.layers:
            x = F.relu(layer(x, adj_norm))

        return x