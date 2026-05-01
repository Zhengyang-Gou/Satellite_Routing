from __future__ import annotations

import torch
import torch.nn as nn

from models.gnn_encoder import GNNEncoder


class PolicyNet(nn.Module):
    """
    Supervised policy network for satellite routing.

    Input:
        adj:
            [B, N, N] binary adjacency matrix.

        node_features:
            [B, N, 5] node feature matrix.

    Output:
        masked_logits:
            [B, N] action logits.
            Invalid next-hop actions are masked to -1e9.

    Node features:
        0: plane ratio
        1: index inside plane ratio
        2: visited flag
        3: current-node flag
        4: normalized one-hop delay from current node
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            adj:
                Tensor with shape [B, N, N].

            node_features:
                Tensor with shape [B, N, input_dim].

        Returns:
            masked_logits:
                Tensor with shape [B, N].
        """
        batch_size, n_nodes, _ = node_features.shape

        node_embeddings = self.gnn(adj, node_features)

        curr_node_mask = node_features[:, :, 3].unsqueeze(-1)
        curr_embeddings = (node_embeddings * curr_node_mask).sum(
            dim=1,
            keepdim=True,
        )

        curr_embeddings_expanded = curr_embeddings.expand(-1, n_nodes, -1)

        pair_features = torch.cat(
            [curr_embeddings_expanded, node_embeddings],
            dim=-1,
        )

        raw_scores = self.score_mlp(pair_features).squeeze(-1)

        curr_node_indices = node_features[:, :, 3].argmax(dim=1)

        valid_moves_mask = adj[
            torch.arange(batch_size, device=adj.device),
            curr_node_indices,
            :,
        ]

        masked_logits = raw_scores.masked_fill(
            valid_moves_mask == 0,
            -1e9,
        )

        return masked_logits