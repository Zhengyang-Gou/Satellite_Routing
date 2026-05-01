from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from models.gnn_encoder import GNNEncoder


class ActorCriticNet(nn.Module):
    """
    Actor-Critic network with shared GNN encoder.

    Actor:
        Same structure as PolicyNet:
            GNNEncoder + score_mlp

        This allows partial loading from supervised PolicyNet checkpoints.

    Critic:
        State value V(s) is computed from:
            - current node embedding
            - global mean pooled embedding
            - unvisited node mean pooled embedding
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _current_node_mask(node_features: torch.Tensor) -> torch.Tensor:
        """
        node_features[..., 3] = 1 means current node.
        """
        return node_features[:, :, 3].unsqueeze(-1)

    @staticmethod
    def _visited_mask(node_features: torch.Tensor) -> torch.Tensor:
        """
        node_features[..., 2] = 1 means visited node.
        """
        return node_features[:, :, 2].unsqueeze(-1)

    @staticmethod
    def _safe_masked_mean(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, N, D]

            mask:
                [B, N, 1]

        Returns:
            pooled:
                [B, D]
        """
        mask = mask.float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled

    def encode_graph(
        self,
        adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return node embeddings.

        Args:
            adj:
                [B, N, N]

            node_features:
                [B, N, input_dim]

        Returns:
            node_embeddings:
                [B, N, hidden_dim]
        """
        return self.gnn(adj, node_features)

    def policy_logits(
        self,
        node_embeddings: torch.Tensor,
        adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked action logits.

        Invalid actions are filled with -1e9.
        """
        batch_size, n_nodes, _ = node_features.shape

        curr_node_mask = self._current_node_mask(node_features)
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

    def value(
        self,
        node_embeddings: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute state value V(s).

        Returns:
            value:
                [B]
        """
        curr_node_mask = self._current_node_mask(node_features)
        visited_mask = self._visited_mask(node_features)
        unvisited_mask = 1.0 - visited_mask

        h_curr = (node_embeddings * curr_node_mask).sum(dim=1)
        h_global = node_embeddings.mean(dim=1)
        h_unvisited = self._safe_masked_mean(
            node_embeddings,
            unvisited_mask,
        )

        graph_repr = torch.cat(
            [h_curr, h_global, h_unvisited],
            dim=-1,
        )

        value = self.value_mlp(graph_repr).squeeze(-1)

        return value

    def forward(
        self,
        adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adj:
                [B, N, N]

            node_features:
                [B, N, input_dim]

        Returns:
            logits:
                [B, N]

            value:
                [B]
        """
        node_embeddings = self.encode_graph(adj, node_features)
        logits = self.policy_logits(
            node_embeddings=node_embeddings,
            adj=adj,
            node_features=node_features,
        )
        value = self.value(
            node_embeddings=node_embeddings,
            node_features=node_features,
        )

        return logits, value

    def load_pretrained_policy(
        self,
        policy_ckpt_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """
        Load compatible actor weights from a supervised PolicyNet checkpoint.

        Supports:
            1. New checkpoint format:
                {"model_state_dict": ...}

            2. Old raw state_dict format:
                {"gnn.conv1.linear.weight": ...}

        Only keys with matching names and shapes are loaded.
        """
        raw_checkpoint = torch.load(
            policy_ckpt_path,
            map_location=map_location,
        )

        if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
            loaded_state = raw_checkpoint["model_state_dict"]
        else:
            loaded_state = raw_checkpoint

        current_state = self.state_dict()

        compatible_state = {}
        skipped_keys = []

        for key, value in loaded_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)

        missing, unexpected = self.load_state_dict(
            compatible_state,
            strict=False,
        )

        print(f"Loaded compatible supervised policy weights: {policy_ckpt_path}")

        if skipped_keys:
            print(f"Skipped incompatible keys: {skipped_keys}")

        if missing:
            print(f"Missing keys after partial load: {missing}")

        if unexpected:
            print(f"Unexpected keys after partial load: {unexpected}")