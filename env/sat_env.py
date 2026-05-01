# env/sat_env.py

from __future__ import annotations

import random
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np

from env.orbit_dynamics import OrbitDynamics
from env.topology import TopologyBuilder


class SatelliteEnv:
    """
    Base satellite routing environment.

    This environment models:
        - dynamic satellite positions
        - dynamic topology updates
        - optional random link failures
        - next-hop routing actions
        - visited-node coverage state

    State:
        {
            "adjacency": np.ndarray,      shape [N, N]
            "node_features": np.ndarray,  shape [N, 5]
            "current_node": int
            "visited": list[int]
            "time": float
            "delay_matrix": np.ndarray,   shape [N, N]
        }

    Node features:
        0: plane ratio
        1: index ratio inside plane
        2: visited flag
        3: current-node flag
        4: normalized one-hop delay from current node
    """

    def __init__(
        self,
        num_planes: int = 6,
        sats_per_plane: int = 10,
        failure_prob: float = 0.0,
        max_link_distance: float = 10000e3,
        seed: int = 42,
    ):
        self.num_planes = int(num_planes)
        self.sats_per_plane = int(sats_per_plane)
        self.num_satellites = self.num_planes * self.sats_per_plane

        self.failure_prob = float(failure_prob)
        self.seed = int(seed)

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.dynamics = OrbitDynamics(
            num_planes=self.num_planes,
            sats_per_plane=self.sats_per_plane,
        )

        self.topology = TopologyBuilder(
            num_planes=self.num_planes,
            sats_per_plane=self.sats_per_plane,
            max_link_distance=max_link_distance,
        )

        self.graph: Optional[nx.Graph] = None
        self.delay_matrix: Optional[np.ndarray] = None

        self.current_node: Optional[int] = None
        self.visited: Optional[Set[int]] = None

        self.time = 0.0
        self.total_delay = 0.0

    def _set_seed(self, seed: int) -> None:
        """
        Set Python and NumPy RNG seeds for reproducible episode initialization.
        """
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _update_environment(self) -> None:
        """
        Update satellite positions, distance matrix, topology graph,
        random link failures, and full delay matrix according to current
        simulation time.
        """
        positions = self.dynamics.compute_positions(self.time)
        dist_matrix = self.dynamics.compute_distance_matrix(positions)

        self.graph = self.topology.build_topology(dist_matrix)

        self._apply_failures()

        # Build after failures so removed links become np.inf in delay_matrix.
        self.delay_matrix = self._build_delay_matrix_from_graph()

    def _apply_failures(self) -> None:
        """
        Randomly remove graph edges according to failure_prob.
        """
        if self.graph is None:
            return

        if self.failure_prob <= 0.0:
            return

        edges = list(self.graph.edges())
        edges_to_remove = [
            edge for edge in edges
            if random.random() < self.failure_prob
        ]

        self.graph.remove_edges_from(edges_to_remove)

    def _build_delay_matrix_from_graph(self) -> np.ndarray:
        """
        Build full raw delay matrix from current graph.

        Shape:
            [num_satellites, num_satellites]

        Values:
            delay_matrix[i, j] = edge delay if edge i-j exists
            delay_matrix[i, j] = np.inf if no edge exists
            delay_matrix[i, i] = 0.0

        This matrix is for oracle evaluation, especially Dijkstra / delay-aware
        planning. It is raw delay, not normalized node-feature delay.
        """
        if self.graph is None:
            raise RuntimeError("Environment graph is not initialized.")

        delay_matrix = nx.to_numpy_array(
            self.graph,
            nodelist=list(range(self.num_satellites)),
            weight="delay",
            nonedge=np.inf,
            dtype=np.float32,
        )

        np.fill_diagonal(delay_matrix, 0.0)

        return delay_matrix

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Reset environment.

        Args:
            seed:
                Optional episode seed. When provided, this makes the initial
                time, start node, and random link failures reproducible for
                this episode.

        Returns:
            state dictionary.
        """
        if seed is not None:
            self._set_seed(seed)

        self.time = random.uniform(0.0, self.dynamics.orbital_period)
        self.total_delay = 0.0

        self._update_environment()

        self.current_node = random.randint(0, self.num_satellites - 1)
        self.visited = {self.current_node}

        return self._get_state()

    def step(self, action: int):
        """
        Execute one routing action.

        Args:
            action:
                Next-hop satellite node id.

        Returns:
            next_state, reward, done, info

        Note:
            This base environment raises ValueError for invalid actions.
            The RL wrapper env.rl_sat_env.RLSatelliteEnv converts invalid
            actions into terminal failure instead.
        """
        if self.graph is None:
            raise RuntimeError("Environment graph is not initialized. Call reset() first.")

        if self.current_node is None:
            raise RuntimeError("Current node is not initialized. Call reset() first.")

        if self.visited is None:
            raise RuntimeError("Visited set is not initialized. Call reset() first.")

        action = int(action)

        if not self.graph.has_edge(self.current_node, action):
            raise ValueError(
                f"Invalid action: no link between {self.current_node} and {action}"
            )

        delay = float(self.graph[self.current_node][action]["delay"])

        self.total_delay += delay
        self.current_node = action
        self.visited.add(action)
        self.time += delay

        self._update_environment()

        done = len(self.visited) == self.num_satellites
        reward = -delay

        info = {
            "time": self.time,
            "total_delay": self.total_delay,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> Dict:
        """
        Build current observation.
        """
        if self.graph is None:
            raise RuntimeError("Environment graph is not initialized.")

        if self.current_node is None:
            raise RuntimeError("Current node is not initialized.")

        if self.visited is None:
            raise RuntimeError("Visited set is not initialized.")

        if self.delay_matrix is None:
            self.delay_matrix = self._build_delay_matrix_from_graph()

        adjacency = nx.to_numpy_array(
            self.graph,
            nodelist=list(range(self.num_satellites)),
            dtype=np.float32,
        )

        node_features = self._node_features()

        return {
            "adjacency": adjacency,
            "node_features": node_features,
            "current_node": int(self.current_node),
            "visited": list(self.visited),
            "time": float(self.time),

            # For oracle evaluation only.
            # Full raw link-delay matrix, not normalized node feature delay.
            "delay_matrix": self.delay_matrix.copy().astype(np.float32),
        }

    def _node_features(self) -> np.ndarray:
        """
        Build node feature matrix with shape [num_satellites, 5].
        """
        if self.visited is None:
            raise RuntimeError("Visited set is not initialized.")

        features = np.zeros(
            (self.num_satellites, 5),
            dtype=np.float32,
        )

        max_hop_delay = max(
            self.topology.max_link_distance / self.topology.c,
            1e-8,
        )

        for node in range(self.num_satellites):
            plane = node // self.sats_per_plane
            index = node % self.sats_per_plane

            features[node, 0] = plane / self.num_planes
            features[node, 1] = index / self.sats_per_plane
            features[node, 2] = 1.0 if node in self.visited else 0.0
            features[node, 3] = 1.0 if node == self.current_node else 0.0

            if (
                self.graph is not None
                and self.current_node is not None
                and node != self.current_node
                and self.graph.has_edge(self.current_node, node)
            ):
                delay = float(self.graph[self.current_node][node]["delay"])
                features[node, 4] = delay / max_hop_delay
            else:
                features[node, 4] = 0.0

        return features

    def render(self) -> None:
        """
        Print current environment status.
        """
        visited_count = len(self.visited) if self.visited is not None else 0

        print(
            f"Time: {self.time:.4f}s | "
            f"Current: {self.current_node} | "
            f"Visited: {visited_count}/{self.num_satellites} | "
            f"Delay: {self.total_delay:.4f}s"
        )