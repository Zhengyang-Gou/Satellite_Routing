# utils/graph_oracle.py

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_adjacency(adjacency) -> np.ndarray:
    adjacency = _to_numpy(adjacency)

    if adjacency.ndim == 3:
        if adjacency.shape[0] != 1:
            raise ValueError(
                f"Expected batched adjacency shape [1, N, N], got {adjacency.shape}"
            )
        adjacency = adjacency[0]

    if adjacency.ndim != 2:
        raise ValueError(f"Expected adjacency shape [N, N], got {adjacency.shape}")

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adjacency.shape}")

    return adjacency


def _normalize_delay_matrix(delay_matrix) -> np.ndarray:
    delay_matrix = _to_numpy(delay_matrix)

    if delay_matrix.ndim == 3:
        if delay_matrix.shape[0] != 1:
            raise ValueError(
                f"Expected batched delay_matrix shape [1, N, N], got {delay_matrix.shape}"
            )
        delay_matrix = delay_matrix[0]

    if delay_matrix.ndim != 2:
        raise ValueError(f"Expected delay_matrix shape [N, N], got {delay_matrix.shape}")

    if delay_matrix.shape[0] != delay_matrix.shape[1]:
        raise ValueError(f"Delay matrix must be square, got {delay_matrix.shape}")

    return delay_matrix


def _normalize_current_node(current_node) -> int:
    if hasattr(current_node, "detach"):
        current_node = current_node.detach().cpu().item()

    if isinstance(current_node, np.ndarray):
        current_node = np.asarray(current_node).reshape(-1)[0]

    return int(current_node)


def _normalize_visited(visited, n: int) -> np.ndarray:
    """
    Return bool mask with shape [N].

    Supports:
        mask:          [N]
        batched mask:  [1, N]
        column mask:   [N, 1]
        index list:    [34] or [34, 12, ...]
        set/list of visited node ids
    """
    mask = np.zeros(n, dtype=bool)

    if visited is None:
        return mask

    # Handle set/list/tuple of visited indices first.
    if isinstance(visited, (set, list, tuple)):
        arr = np.asarray(list(visited))
    else:
        arr = _to_numpy(visited)

    arr = np.asarray(arr)

    if arr.size == 0:
        return mask

    squeezed = np.squeeze(arr)

    if squeezed.ndim == 0:
        value = int(squeezed.item())
        if 0 <= value < n:
            mask[value] = True
            return mask
        raise ValueError(f"Visited scalar index out of range: {value}, N={n}")

    flat = squeezed.reshape(-1)

    # Case 1: full mask.
    if flat.shape[0] == n:
        # If values are 0/1 or bool, treat as mask.
        unique_values = np.unique(flat)
        if np.all(np.isin(unique_values, [0, 1, False, True])):
            return flat.astype(bool)

        # Otherwise, if full length but not binary, still treat nonzero as mask.
        return flat.astype(bool)

    # Case 2: index list.
    # Example: [34] means node 34 has been visited.
    if np.issubdtype(flat.dtype, np.integer) or np.all(np.equal(flat, np.round(flat))):
        for value in flat:
            idx = int(value)
            if 0 <= idx < n:
                mask[idx] = True
            else:
                raise ValueError(f"Visited index out of range: {idx}, N={n}")
        return mask

    raise ValueError(
        f"Cannot normalize visited. Expected mask length {n} or index list, "
        f"got shape {arr.shape}, values sample={flat[:10]}"
    )


def _neighbors(adjacency: np.ndarray, node: int) -> List[int]:
    return np.where(adjacency[node] > 0)[0].astype(int).tolist()


def _reconstruct_path(
    parent: Dict[int, Optional[int]],
    target: int,
) -> List[int]:
    path = []
    cur = target

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()
    return path


def bfs_path_to_nearest_unvisited(
    adjacency,
    current_node: int,
    visited,
) -> Tuple[int, List[int], int]:
    """
    Hop-first snapshot oracle.

    Finds the nearest unvisited node by hop count under the current topology
    snapshot and returns only the first hop.
    """
    adjacency = _normalize_adjacency(adjacency)
    n = adjacency.shape[0]

    current_node = _normalize_current_node(current_node)
    visited_mask = _normalize_visited(visited, n)

    if current_node < 0 or current_node >= n:
        raise ValueError(f"current_node out of range: {current_node}, N={n}")

    queue = deque([current_node])
    parent: Dict[int, Optional[int]] = {current_node: None}

    best_target = None

    while queue:
        node = queue.popleft()

        if node != current_node and not visited_mask[node]:
            best_target = node
            break

        for nxt in _neighbors(adjacency, node):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)

    if best_target is None:
        neigh = _neighbors(adjacency, current_node)

        if not neigh:
            return current_node, [current_node], 0

        return int(neigh[0]), [current_node, int(neigh[0])], 1

    path = _reconstruct_path(parent, best_target)

    if len(path) <= 1:
        next_action = current_node
    else:
        next_action = path[1]

    return int(next_action), path, int(len(path) - 1)


def dijkstra_path_to_nearest_unvisited(
    adjacency,
    delay_matrix,
    current_node: int,
    visited,
) -> Tuple[int, List[int], float, int]:
    """
    Delay-first snapshot oracle.

    Finds the nearest unvisited node by shortest-path delay under the current
    topology snapshot and returns only the first hop.
    """
    adjacency = _normalize_adjacency(adjacency)
    delay_matrix = _normalize_delay_matrix(delay_matrix)

    n = adjacency.shape[0]

    if delay_matrix.shape != adjacency.shape:
        raise ValueError(
            f"delay_matrix shape {delay_matrix.shape} does not match "
            f"adjacency shape {adjacency.shape}"
        )

    current_node = _normalize_current_node(current_node)
    visited_mask = _normalize_visited(visited, n)

    if current_node < 0 or current_node >= n:
        raise ValueError(f"current_node out of range: {current_node}, N={n}")

    dist = np.full(n, np.inf, dtype=np.float64)
    hops = np.full(n, np.inf, dtype=np.float64)
    parent: Dict[int, Optional[int]] = {current_node: None}

    dist[current_node] = 0.0
    hops[current_node] = 0.0

    pq = [(0.0, 0, current_node)]

    while pq:
        cur_dist, cur_hops, node = heapq.heappop(pq)

        if cur_dist > dist[node]:
            continue

        for nxt in _neighbors(adjacency, node):
            edge_delay = float(delay_matrix[node, nxt])

            if not np.isfinite(edge_delay):
                continue

            new_dist = cur_dist + edge_delay
            new_hops = cur_hops + 1

            better = False

            if new_dist < dist[nxt]:
                better = True
            elif np.isclose(new_dist, dist[nxt]) and new_hops < hops[nxt]:
                better = True

            if better:
                dist[nxt] = new_dist
                hops[nxt] = new_hops
                parent[nxt] = node
                heapq.heappush(pq, (new_dist, new_hops, nxt))

    candidates = [
        node
        for node in range(n)
        if node != current_node and not visited_mask[node] and np.isfinite(dist[node])
    ]

    if not candidates:
        neigh = _neighbors(adjacency, current_node)

        if not neigh:
            return current_node, [current_node], 0.0, 0

        nxt = int(neigh[0])
        return (
            nxt,
            [current_node, nxt],
            float(delay_matrix[current_node, nxt]),
            1,
        )

    best_target = min(
        candidates,
        key=lambda node: (dist[node], hops[node], node),
    )

    path = _reconstruct_path(parent, best_target)

    if len(path) <= 1:
        next_action = current_node
    else:
        next_action = path[1]

    return (
        int(next_action),
        path,
        float(dist[best_target]),
        int(hops[best_target]),
    )


def snapshot_oracle_action(
    adjacency,
    current_node: int,
    visited,
    mode: str = "hop_first",
    delay_matrix=None,
) -> int:
    if mode == "hop_first":
        action, _, _ = bfs_path_to_nearest_unvisited(
            adjacency=adjacency,
            current_node=current_node,
            visited=visited,
        )
        return int(action)

    if mode == "delay_first":
        if delay_matrix is None:
            raise ValueError("delay_first oracle requires a full delay_matrix.")

        action, _, _, _ = dijkstra_path_to_nearest_unvisited(
            adjacency=adjacency,
            delay_matrix=delay_matrix,
            current_node=current_node,
            visited=visited,
        )
        return int(action)

    raise ValueError(f"Unsupported oracle mode: {mode}")