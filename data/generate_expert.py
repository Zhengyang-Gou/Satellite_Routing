# data/generate_expert.py

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from env.sat_env import SatelliteEnv
from utils.config import (
    EnvConfig,
    ExpertConfig,
    build_env_config,
    build_expert_config,
    save_config,
)
from utils.paths import resolve_project_path
from utils.seed import set_seed


def greedy_expert_policy(env: SatelliteEnv) -> Optional[int]:
    """
    Greedy expert policy.

    Strategy:
        1. From current node, compute shortest delay distance to all nodes.
        2. Pick the nearest unvisited node as the target.
        3. Return the next hop on the shortest path to that target.

    Note:
        This is a greedy heuristic, not a globally optimal solver.
    """
    current = env.current_node
    unvisited = [n for n in range(env.num_satellites) if n not in env.visited]

    if not unvisited:
        return None

    try:
        lengths = nx.single_source_dijkstra_path_length(
            env.graph,
            current,
            weight="delay",
        )
    except Exception:
        return None

    closest_target = min(
        unvisited,
        key=lambda n: lengths.get(n, float("inf")),
    )

    if lengths.get(closest_target, float("inf")) == float("inf"):
        return None

    try:
        path = nx.shortest_path(
            env.graph,
            source=current,
            target=closest_target,
            weight="delay",
        )
    except nx.NetworkXNoPath:
        return None

    if len(path) <= 1:
        return None

    return int(path[1])


def _prepare_output_dir(output_dir: Path) -> None:
    """
    Create output directory and remove old generated dataset files.

    This avoids mixing old chunks with new chunks when regenerating data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.glob("chunk_*.pkl"):
        path.unlink()

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()


def _flush_chunk(
    chunk_data: List[Dict[str, Any]],
    chunk_index: int,
    output_dir: Path,
) -> Path:
    """
    Save one chunk of expert data.
    """
    chunk_path = output_dir / f"chunk_{chunk_index:05d}.pkl"

    with open(chunk_path, "wb") as f:
        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return chunk_path


def build_expert_env(env_cfg: EnvConfig) -> SatelliteEnv:
    """
    Build SatelliteEnv for expert data generation.
    """
    return SatelliteEnv(
        num_planes=env_cfg.num_planes,
        sats_per_plane=env_cfg.sats_per_plane,
        failure_prob=env_cfg.failure_prob,
        max_link_distance=env_cfg.max_link_distance,
        seed=env_cfg.seed,
    )


def generate_dataset_from_config(cfg: Dict[str, Any]) -> Path:
    """
    Generate expert dataset from config.

    Args:
        cfg:
            Full YAML config dictionary.

    Returns:
        Path to dataset output directory.
    """
    env_cfg: EnvConfig = build_env_config(cfg)
    expert_cfg: ExpertConfig = build_expert_config(cfg)

    set_seed(env_cfg.seed)

    output_dir = resolve_project_path(expert_cfg.save_dir)
    _prepare_output_dir(output_dir)

    save_config(cfg, output_dir / "config.yaml")

    env = build_expert_env(env_cfg)

    print("Starting expert dataset generation")
    print(f"Output dir: {output_dir}")
    print(f"Expert policy: {expert_cfg.policy}")
    print(f"Target successful episodes: {expert_cfg.num_episodes}")

    if expert_cfg.policy != "greedy_shortest_unvisited":
        raise ValueError(
            f"Unsupported expert policy: {expert_cfg.policy}. "
            "Currently supported: greedy_shortest_unvisited"
        )

    current_chunk: List[Dict[str, Any]] = []
    chunk_files: List[str] = []
    chunk_sizes: List[int] = []

    successful_episodes = 0
    attempted_episodes = 0
    total_samples = 0
    chunk_index = 0

    while successful_episodes < expert_cfg.num_episodes:
        attempted_episodes += 1

        state = env.reset()
        episode_data: List[Dict[str, Any]] = []

        done = False
        steps = 0
        max_steps = env.num_satellites * 3

        while not done and steps < max_steps:
            action = greedy_expert_policy(env)

            if action is None:
                break

            episode_data.append(
                {
                    "state": state,
                    "action": action,
                }
            )

            state, reward, done, info = env.step(action)
            steps += 1

        # Keep only successful full-coverage episodes.
        if done:
            current_chunk.extend(episode_data)
            successful_episodes += 1

            if successful_episodes % expert_cfg.chunk_episode_size == 0:
                chunk_path = _flush_chunk(
                    chunk_data=current_chunk,
                    chunk_index=chunk_index,
                    output_dir=output_dir,
                )

                chunk_files.append(chunk_path.name)
                chunk_sizes.append(len(current_chunk))
                total_samples += len(current_chunk)

                print(
                    f"Saved chunk {chunk_index:05d} | "
                    f"successful episodes {successful_episodes}/{expert_cfg.num_episodes} | "
                    f"samples {len(current_chunk)}"
                )

                current_chunk = []
                chunk_index += 1

            if successful_episodes % 50 == 0:
                print(
                    f"Progress: successful episodes "
                    f"{successful_episodes}/{expert_cfg.num_episodes} "
                    f"| attempted {attempted_episodes}"
                )

    if current_chunk:
        chunk_path = _flush_chunk(
            chunk_data=current_chunk,
            chunk_index=chunk_index,
            output_dir=output_dir,
        )

        chunk_files.append(chunk_path.name)
        chunk_sizes.append(len(current_chunk))
        total_samples += len(current_chunk)

        print(
            f"Saved final chunk {chunk_index:05d} | "
            f"successful episodes {successful_episodes}/{expert_cfg.num_episodes} | "
            f"samples {len(current_chunk)}"
        )

    metadata = {
        "format": "chunked_expert_dataset",
        "policy": expert_cfg.policy,
        "num_episodes": successful_episodes,
        "attempted_episodes": attempted_episodes,
        "total_samples": total_samples,
        "chunk_episode_size": expert_cfg.chunk_episode_size,
        "chunk_files": chunk_files,
        "chunk_sizes": chunk_sizes,
        "env": {
            "num_planes": env_cfg.num_planes,
            "sats_per_plane": env_cfg.sats_per_plane,
            "failure_prob": env_cfg.failure_prob,
            "max_link_distance": env_cfg.max_link_distance,
            "seed": env_cfg.seed,
        },
    }

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Expert dataset generation finished")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Attempted episodes: {attempted_episodes}")
    print(f"Total samples: {total_samples}")
    print(f"Dataset saved to: {output_dir}")

    return output_dir