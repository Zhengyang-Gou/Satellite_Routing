from __future__ import annotations

import networkx as nx
import numpy as np


class TopologyBuilder:
    """
    Build satellite inter-satellite-link topology from distance matrix.

    Nodes:
        Satellite ids from 0 to num_satellites - 1.

    Candidate edges:
        1. Intra-plane link:
            satellite (plane, i) <-> satellite (plane, i + 1)

        2. Inter-plane link:
            satellite (plane, i) <-> satellite (plane + 1, i)

    An edge is added only if its physical distance is <= max_link_distance.

    Edge attribute:
        delay = distance / speed_of_light
    """

    def __init__(
        self,
        num_planes: int,
        sats_per_plane: int,
        max_link_distance: float = 2000e3,
        speed_of_light: float = 3e8,
    ):
        self.num_planes = int(num_planes)
        self.sats_per_plane = int(sats_per_plane)
        self.num_satellites = self.num_planes * self.sats_per_plane

        self.max_link_distance = float(max_link_distance)
        self.c = float(speed_of_light)

        if self.num_planes <= 0:
            raise ValueError(f"num_planes must be positive, got: {num_planes}")

        if self.sats_per_plane <= 0:
            raise ValueError(f"sats_per_plane must be positive, got: {sats_per_plane}")

        if self.max_link_distance <= 0:
            raise ValueError(
                f"max_link_distance must be positive, got: {max_link_distance}"
            )

        if self.c <= 0:
            raise ValueError(f"speed_of_light must be positive, got: {speed_of_light}")

    def node_id(self, plane: int, index: int) -> int:
        """
        Convert (plane, index) to flattened node id.
        """
        plane = plane % self.num_planes
        index = index % self.sats_per_plane

        return plane * self.sats_per_plane + index

    def build_topology(self, dist_matrix: np.ndarray) -> nx.Graph:
        """
        Build NetworkX graph from distance matrix.

        Args:
            dist_matrix:
                Pairwise distance matrix with shape [N, N].

        Returns:
            G:
                Undirected graph with delay-weighted edges.
        """
        dist_matrix = np.asarray(dist_matrix)

        expected_shape = (self.num_satellites, self.num_satellites)

        if dist_matrix.shape != expected_shape:
            raise ValueError(
                f"dist_matrix must have shape {expected_shape}, "
                f"got: {dist_matrix.shape}"
            )

        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_satellites))

        edges_to_add = []

        for plane in range(self.num_planes):
            for index in range(self.sats_per_plane):
                node = self.node_id(plane, index)

                # Intra-plane neighbor: next satellite in the same orbital plane.
                intra_node = self.node_id(plane, index + 1)
                intra_distance = float(dist_matrix[node, intra_node])

                if intra_distance <= self.max_link_distance:
                    edges_to_add.append(
                        (
                            node,
                            intra_node,
                            {"delay": intra_distance / self.c},
                        )
                    )

                # Inter-plane neighbor: same index in the next orbital plane.
                inter_node = self.node_id(plane + 1, index)
                inter_distance = float(dist_matrix[node, inter_node])

                if inter_distance <= self.max_link_distance:
                    edges_to_add.append(
                        (
                            node,
                            inter_node,
                            {"delay": inter_distance / self.c},
                        )
                    )

        graph.add_edges_from(edges_to_add)

        return graph