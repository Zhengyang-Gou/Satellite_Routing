# scripts/compare_policies.py

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from env.rl_sat_env import RLSatelliteEnv
from models.policy_net import PolicyNet
from models.actor_critic_net import ActorCriticNet

from utils.config import (
    load_config,
    build_env_config,
    build_reward_config,
    build_model_config,
)
from utils.paths import resolve_project_path
from utils.seed import set_seed
from utils.graph_oracle import snapshot_oracle_action

try:
    from utils.checkpoint import load_checkpoint
except Exception:
    load_checkpoint = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to compare config YAML, e.g. configs/eval/compare_same_env.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_path(path_like: str | Path) -> Path:
    path = Path(path_like)

    if path.is_absolute():
        return path

    return Path(resolve_project_path(str(path)))


def build_section(build_fn, raw_cfg: Dict[str, Any], section_name: str):
    """
    Compatible wrapper.

    Some builder functions may expect raw_cfg[section_name],
    while others may expect the full raw config.
    """
    section = raw_cfg.get(section_name, {})

    first_error = None

    try:
        return build_fn(section)
    except Exception as exc:
        first_error = exc

    try:
        return build_fn(raw_cfg)
    except Exception:
        raise first_error


def create_compare_run_dir(raw_cfg: Dict[str, Any]) -> Path:
    experiment = raw_cfg.get("experiment", {}) or {}
    logging_cfg = raw_cfg.get("logging", {}) or {}

    name = experiment.get("name", "compare_policies")
    output_root = logging_cfg.get("output_root", "outputs/runs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_path(output_root) / f"{timestamp}_{name}"
    eval_dir = run_dir / "eval"

    eval_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def load_checkpoint_any(path: str | Path, device: torch.device) -> Any:
    ckpt_path = project_path(path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if load_checkpoint is not None:
        try:
            return load_checkpoint(str(ckpt_path), map_location=device)
        except TypeError:
            try:
                return load_checkpoint(str(ckpt_path))
            except Exception:
                pass
        except Exception:
            pass

    return torch.load(str(ckpt_path), map_location=device)


def _is_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    if len(obj) == 0:
        return False

    return all(torch.is_tensor(v) for v in obj.values())


def clean_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        prefixes = [
            "module.",
            "model.",
            "policy.",
            "net.",
            "actor_critic.",
        ]

        changed = True
        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats:
        raw state_dict
        {"model_state_dict": ...}
        {"state_dict": ...}
        {"model": ...}
        {"policy_state_dict": ...}
        {"actor_critic_state_dict": ...}
    """
    if _is_state_dict(checkpoint):
        return clean_state_dict_keys(checkpoint)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "policy_state_dict",
        "net_state_dict",
        "actor_critic_state_dict",
        "actor_critic",
    ]

    for key in candidate_keys:
        value = checkpoint.get(key)

        if _is_state_dict(value):
            return clean_state_dict_keys(value)

    for _, value in checkpoint.items():
        if _is_state_dict(value):
            return clean_state_dict_keys(value)

    raise ValueError(
        "Could not find a model state_dict in checkpoint. "
        f"Available top-level keys: {list(checkpoint.keys())}"
    )


def build_policy_model(
    policy_type: str,
    model_cfg: Any,
    device: torch.device,
) -> torch.nn.Module:
    input_dim = int(getattr(model_cfg, "input_dim", 5))
    hidden_dim = int(getattr(model_cfg, "hidden_dim", 64))

    if policy_type == "sl":
        model = PolicyNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

    elif policy_type == "ppo":
        model = ActorCriticNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

    else:
        raise ValueError(f"Unsupported policy type for model loading: {policy_type}")

    return model.to(device)


def load_policy(
    policy_spec: Dict[str, Any],
    model_cfg: Any,
    device: torch.device,
) -> torch.nn.Module:
    name = policy_spec["name"]
    policy_type = policy_spec["type"]
    checkpoint_path = policy_spec["checkpoint"]

    model = build_policy_model(
        policy_type=policy_type,
        model_cfg=model_cfg,
        device=device,
    )

    checkpoint = load_checkpoint_any(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint)

    load_result = model.load_state_dict(state_dict, strict=False)

    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))

    if missing:
        print(f"[WARN] {name}: missing keys when loading checkpoint:")
        for key in missing[:20]:
            print(f"  missing: {key}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print(f"[WARN] {name}: unexpected keys when loading checkpoint:")
        for key in unexpected[:20]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    model.eval()
    return model


def obs_to_tensors(
    obs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    adjacency = torch.as_tensor(
        obs["adjacency"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    node_features = torch.as_tensor(
        obs["node_features"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    visited_mask = extract_visited_mask_from_obs(obs)

    visited = torch.as_tensor(
        visited_mask,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    current_node = torch.as_tensor(
        [int(obs["current_node"])],
        dtype=torch.long,
        device=device,
    )

    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "visited": visited,
        "current_node": current_node,
    }


def unpack_logits(model_output: Any) -> torch.Tensor:
    """
    Supports:
        logits
        (logits, value)
        {"logits": logits}
        {"actor_logits": logits}
        {"action_logits": logits}
    """
    if torch.is_tensor(model_output):
        logits = model_output

    elif isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model returned an empty tuple/list.")
        logits = model_output[0]

    elif isinstance(model_output, dict):
        logits = None

        for key in ["logits", "actor_logits", "action_logits", "policy_logits"]:
            if key in model_output:
                logits = model_output[key]
                break

        if logits is None:
            raise ValueError(
                f"Could not find logits in model output dict keys: "
                f"{list(model_output.keys())}"
            )

    else:
        raise ValueError(
            f"Could not unpack logits from model output type: {type(model_output)}"
        )

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    return logits


def forward_logits(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Compatible forward wrapper.

    Your current PolicyNet.forward() appears to use:
        model(adj, node_features)

    So this function tries two-argument signatures first.
    """
    tensors = obs_to_tensors(obs, device)

    adjacency = tensors["adjacency"]
    node_features = tensors["node_features"]
    current_node = tensors["current_node"]
    visited = tensors["visited"]

    call_errors = []

    candidate_calls = [
        # Current likely signature:
        lambda: model(adjacency, node_features),
        lambda: model(node_features, adjacency),

        # Keyword variants:
        lambda: model(adj=adjacency, node_features=node_features),
        lambda: model(adjacency=adjacency, node_features=node_features),
        lambda: model(adj_matrix=adjacency, node_features=node_features),
        lambda: model(adjacency_matrix=adjacency, node_features=node_features),

        # Older / alternative signatures:
        lambda: model(node_features, adjacency, current_node, visited),
        lambda: model(adjacency, node_features, current_node, visited),
        lambda: model(
            node_features=node_features,
            adjacency=adjacency,
            current_node=current_node,
            visited=visited,
        ),
        lambda: model(
            node_features=node_features,
            adj=adjacency,
            current_node=current_node,
            visited=visited,
        ),
    ]

    for call in candidate_calls:
        try:
            output = call()
            return unpack_logits(output)
        except Exception as exc:
            call_errors.append(str(exc))

    raise RuntimeError(
        "Failed to call model forward. Tried several common signatures. "
        f"Last errors: {call_errors[-5:]}"
    )


def mask_logits_by_adjacency(
    logits: torch.Tensor,
    obs: Dict[str, Any],
) -> torch.Tensor:
    """
    Safety mask.

    Revisits are valid, so this only masks non-neighbor actions.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError(f"Expected logits shape [B, N], got {tuple(logits.shape)}")

    adjacency = np.asarray(obs["adjacency"])

    if adjacency.ndim == 3:
        adjacency = adjacency[0]

    current_node = int(obs["current_node"])

    valid = adjacency[current_node] > 0

    if valid.sum() == 0:
        return logits

    valid_t = torch.as_tensor(
        valid,
        dtype=torch.bool,
        device=logits.device,
    ).unsqueeze(0)

    masked = logits.clone()
    masked = masked.masked_fill(~valid_t, -1e9)

    return masked


@torch.no_grad()
def select_action(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    device: torch.device,
    greedy: bool,
) -> int:
    logits = forward_logits(model, obs, device)
    logits = mask_logits_by_adjacency(logits, obs)

    if greedy:
        return int(torch.argmax(logits, dim=-1).item())

    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().item())


def get_delay_matrix_from_env(env: Any):
    """
    Best-effort extractor for full delay matrix.

    Needed only for delay_first oracle.
    """
    candidate_attrs = [
        "delay_matrix",
        "link_delay_matrix",
        "edge_delay_matrix",
        "delays",
        "edge_delays",
    ]

    objects = [env]

    for attr in ["env", "base_env", "sat_env", "wrapped_env"]:
        if hasattr(env, attr):
            obj = getattr(env, attr)
            if obj is not None:
                objects.append(obj)

    for obj in objects:
        for attr in candidate_attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if value is not None:
                    return value

    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "y"]

    return default


def get_num_nodes_from_obs(obs: Dict[str, Any]) -> int:
    if "adjacency" in obs:
        adjacency = np.asarray(obs["adjacency"])

        if adjacency.ndim == 3:
            adjacency = adjacency[0]

        if adjacency.ndim == 2:
            return int(adjacency.shape[0])

    if "node_features" in obs:
        node_features = np.asarray(obs["node_features"])

        if node_features.ndim == 3:
            node_features = node_features[0]

        if node_features.ndim == 2:
            return int(node_features.shape[0])

    raise ValueError("Cannot infer number of nodes from obs.")


def extract_visited_mask_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Robustly extract visited mask [N].

    Priority:
        1. node_features[:, 2], because project state design says feature 2 is visited flag.
        2. obs["visited"] as full binary mask.
        3. obs["visited"] as visited node index list.
        4. current_node fallback.
    """
    n = get_num_nodes_from_obs(obs)

    # Best source according to project convention:
    # node_features:
    #   0 plane ratio
    #   1 index ratio
    #   2 visited flag
    #   3 current-node flag
    #   4 normalized one-hop delay
    if "node_features" in obs:
        node_features = np.asarray(obs["node_features"])

        if node_features.ndim == 3:
            node_features = node_features[0]

        if (
            node_features.ndim == 2
            and node_features.shape[0] == n
            and node_features.shape[1] >= 3
        ):
            visited_flag = node_features[:, 2]
            return (visited_flag > 0.5).astype(bool)

    mask = np.zeros(n, dtype=bool)

    visited = obs.get("visited", None)

    if visited is None:
        current_node = int(obs.get("current_node", 0))
        if 0 <= current_node < n:
            mask[current_node] = True
        return mask

    if isinstance(visited, (set, list, tuple)):
        arr = np.asarray(list(visited))
    else:
        if hasattr(visited, "detach"):
            arr = visited.detach().cpu().numpy()
        else:
            arr = np.asarray(visited)

    if arr.size == 0:
        current_node = int(obs.get("current_node", 0))
        if 0 <= current_node < n:
            mask[current_node] = True
        return mask

    arr = np.squeeze(arr).reshape(-1)

    # Case 1: visited is full mask.
    if arr.shape[0] == n:
        unique_values = np.unique(arr)
        if np.all(np.isin(unique_values, [0, 1, False, True])):
            return arr.astype(bool)

        return arr.astype(bool)

    # Case 2: visited is index list, e.g. [34] or [0, 4, 9].
    if np.issubdtype(arr.dtype, np.integer) or np.all(np.equal(arr, np.round(arr))):
        for value in arr:
            idx = int(value)
            if 0 <= idx < n:
                mask[idx] = True

        return mask

    current_node = int(obs.get("current_node", 0))
    if 0 <= current_node < n:
        mask[current_node] = True

    return mask


def final_coverage_from_obs(obs: Dict[str, Any]) -> float:
    """
    Compute coverage ratio robustly.

    Supports:
        obs["visited"] as mask [N]
        obs["visited"] as visited node index list
        obs["node_features"][:, 2] as visited flag
    """
    try:
        visited_mask = extract_visited_mask_from_obs(obs)
    except Exception:
        return 0.0

    if visited_mask.size == 0:
        return 0.0

    return float(np.mean(visited_mask.astype(bool)))


def get_info_float(
    info: Dict[str, Any],
    keys: List[str],
    default: float = 0.0,
) -> float:
    for key in keys:
        if key in info:
            return safe_float(info[key], default)

    return default


def get_info_bool(
    info: Dict[str, Any],
    keys: List[str],
    default: bool = False,
) -> bool:
    for key in keys:
        if key in info:
            return safe_bool(info[key], default)

    return default


def compute_coverage_ratio(
    final_info: Dict[str, Any],
    obs: Dict[str, Any],
) -> float:
    """
    Prefer explicit ratio keys.
    If final_info["coverage"] is a count, convert it to ratio.
    Otherwise compute from obs.
    """
    coverage_ratio = get_info_float(
        final_info,
        keys=["coverage_ratio", "visited_ratio"],
        default=final_coverage_from_obs(obs),
    )

    if (
        "coverage" in final_info
        and "coverage_ratio" not in final_info
        and "visited_ratio" not in final_info
    ):
        raw_coverage = safe_float(final_info["coverage"], default=coverage_ratio)

        try:
            n_nodes = get_num_nodes_from_obs(obs)
        except Exception:
            n_nodes = 1

        if raw_coverage > 1.0 and n_nodes > 1:
            coverage_ratio = raw_coverage / n_nodes
        else:
            coverage_ratio = raw_coverage

    # Clamp only for reporting.
    if coverage_ratio < 0.0:
        coverage_ratio = 0.0

    if coverage_ratio > 1.0:
        coverage_ratio = 1.0

    return float(coverage_ratio)


def get_valid_neighbors_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    adjacency = np.asarray(obs["adjacency"])

    if adjacency.ndim == 3:
        adjacency = adjacency[0]

    current_node = int(obs["current_node"])

    return np.where(adjacency[current_node] > 0)[0]


def evaluate_one_episode(
    model: Optional[torch.nn.Module],
    env_cfg: Any,
    reward_cfg: Any,
    episode_seed: int,
    device: torch.device,
    greedy: bool,
    oracle_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one episode.

    model:
        SL / PPO model for learned policies.
        None for oracle policies.

    oracle_mode:
        None:
            use learned model.
        "hop_first":
            snapshot BFS oracle.
        "delay_first":
            snapshot Dijkstra oracle. Requires full delay_matrix.
    """
    set_seed(episode_seed)

    env = RLSatelliteEnv(env_cfg=env_cfg, reward_cfg=reward_cfg)

    try:
        obs, info = env.reset(seed=episode_seed)
    except TypeError:
        obs, info = env.reset()

    done = False
    steps = 0
    episode_return = 0.0
    total_delay_acc = 0.0
    invalid_action = False
    final_info: Dict[str, Any] = {}

    num_planes = int(getattr(env_cfg, "num_planes", 6))
    sats_per_plane = int(getattr(env_cfg, "sats_per_plane", 10))
    max_steps_factor = int(getattr(env_cfg, "max_steps_factor", 3))

    max_guard_steps = num_planes * sats_per_plane * max_steps_factor + 10

    while not done:
        try:
            if model is None:
                if oracle_mode is None:
                    oracle_mode = "hop_first"

                delay_matrix = None

                if oracle_mode == "delay_first":
                    if isinstance(obs, dict) and "delay_matrix" in obs:
                        delay_matrix = obs["delay_matrix"]
                    else:
                        delay_matrix = get_delay_matrix_from_env(env)

                    if delay_matrix is None:
                        raise RuntimeError(
                            "delay_first oracle requested, but no full delay_matrix "
                            "was found. Either expose delay_matrix from the env or "
                            "use oracle mode='hop_first'."
                        )

                visited_mask = extract_visited_mask_from_obs(obs)

                action = snapshot_oracle_action(
                    adjacency=obs["adjacency"],
                    current_node=int(obs["current_node"]),
                    visited=visited_mask,
                    mode=oracle_mode,
                    delay_matrix=delay_matrix,
                )

                # Debug guard: if oracle picks invalid action, expose details.
                valid_neighbors = get_valid_neighbors_from_obs(obs)
                if action not in set(valid_neighbors.astype(int).tolist()):
                    raise ValueError(
                        "Oracle selected invalid action. "
                        f"current_node={obs['current_node']}, "
                        f"action={action}, "
                        f"valid_neighbors={valid_neighbors.tolist()}, "
                        f"visited_count={int(visited_mask.sum())}"
                    )

            else:
                action = select_action(
                    model=model,
                    obs=obs,
                    device=device,
                    greedy=greedy,
                )

            next_obs, reward, done, step_info = env.step(action)

        except ValueError as exc:
            invalid_action = True
            reward = safe_float(getattr(reward_cfg, "fail", -5.0), -5.0)
            done = True
            step_info = {
                "success": False,
                "failed": True,
                "invalid_action": True,
                "error": str(exc),
            }
            next_obs = obs

        steps += 1
        episode_return += safe_float(reward, 0.0)

        step_delay = get_info_float(
            step_info,
            keys=["delay", "step_delay", "link_delay"],
            default=0.0,
        )
        total_delay_acc += step_delay

        obs = next_obs
        final_info = step_info

        if steps >= max_guard_steps:
            done = True
            final_info = dict(final_info)
            final_info["success"] = False
            final_info["failed"] = True
            final_info["truncated_by_guard"] = True
            break

    coverage_ratio = compute_coverage_ratio(final_info, obs)

    success = get_info_bool(
        final_info,
        keys=["success", "is_success"],
        default=(coverage_ratio >= 1.0 - 1e-8 and not invalid_action),
    )

    failed = get_info_bool(
        final_info,
        keys=["failed", "fail", "is_failed"],
        default=(not success),
    )

    invalid = invalid_action or get_info_bool(
        final_info,
        keys=["invalid_action", "invalid"],
        default=False,
    )

    repeat_count = get_info_float(
        final_info,
        keys=["repeat_count", "num_repeats"],
        default=safe_float(getattr(env, "repeat_count", 0.0), 0.0),
    )

    repeat_ratio = get_info_float(
        final_info,
        keys=["repeat_ratio"],
        default=(
            safe_float(getattr(env, "repeat_ratio", 0.0), 0.0)
            if hasattr(env, "repeat_ratio")
            else repeat_count / max(steps, 1)
        ),
    )

    total_delay = get_info_float(
        final_info,
        keys=["total_delay", "path_delay", "episode_delay"],
        default=total_delay_acc,
    )

    return {
        "episode_seed": episode_seed,
        "success": int(success),
        "failed": int(failed),
        "invalid_action": int(invalid),
        "coverage_ratio": float(coverage_ratio),
        "total_delay": float(total_delay),
        "steps": int(steps),
        "repeat_count": float(repeat_count),
        "repeat_ratio": float(repeat_ratio),
        "episode_return": float(episode_return),
    }


def mean(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None

    return float(np.mean(values))


def std(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None

    return float(np.std(values))


def summarize_policy_results(
    policy_name: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(rows)

    success_values = [float(row["success"]) for row in rows]
    failed_values = [float(row["failed"]) for row in rows]
    invalid_values = [float(row["invalid_action"]) for row in rows]

    coverage_values = [float(row["coverage_ratio"]) for row in rows]
    delay_values = [float(row["total_delay"]) for row in rows]
    delay_success_values = [
        float(row["total_delay"])
        for row in rows
        if int(row["success"]) == 1
    ]
    steps_values = [float(row["steps"]) for row in rows]
    repeat_count_values = [float(row["repeat_count"]) for row in rows]
    repeat_ratio_values = [float(row["repeat_ratio"]) for row in rows]
    return_values = [float(row["episode_return"]) for row in rows]

    return {
        "policy": policy_name,
        "num_episodes": n,
        "success_rate": mean(success_values),
        "fail_rate": mean(failed_values),
        "invalid_rate": mean(invalid_values),
        "coverage_ratio_mean": mean(coverage_values),
        "coverage_ratio_std": std(coverage_values),
        "total_delay_mean_all": mean(delay_values),
        "total_delay_std_all": std(delay_values),
        "total_delay_mean_success": mean(delay_success_values),
        "steps_mean": mean(steps_values),
        "steps_std": std(steps_values),
        "repeat_count_mean": mean(repeat_count_values),
        "repeat_ratio_mean": mean(repeat_ratio_values),
        "episode_return_mean": mean(return_values),
        "episode_return_std": std(return_values),
    }


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config_copy(path: Path, raw_cfg: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            raw_cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n=== Policy Comparison Summary ===")
    print(
        f"{'policy':<28} "
        f"{'success':>10} "
        f"{'coverage':>10} "
        f"{'delay_succ':>12} "
        f"{'steps':>10} "
        f"{'repeat':>10} "
        f"{'return':>10}"
    )

    for row in summary_rows:

        def fmt(key: str) -> str:
            value = row.get(key)

            if value is None:
                return "NA"

            if isinstance(value, float):
                if math.isnan(value):
                    return "NA"
                return f"{value:.4f}"

            return str(value)

        print(
            f"{fmt('policy'):<28} "
            f"{fmt('success_rate'):>10} "
            f"{fmt('coverage_ratio_mean'):>10} "
            f"{fmt('total_delay_mean_success'):>12} "
            f"{fmt('steps_mean'):>10} "
            f"{fmt('repeat_ratio_mean'):>10} "
            f"{fmt('episode_return_mean'):>10}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    raw_cfg = load_config(args.config)

    if "policies" not in raw_cfg:
        raise KeyError(
            "Missing 'policies' section in compare config. "
            "Expected a list of policy specs."
        )

    policies = raw_cfg["policies"]

    if not isinstance(policies, list) or len(policies) == 0:
        raise ValueError("'policies' must be a non-empty list.")

    experiment_cfg = raw_cfg.get("experiment", {}) or {}
    eval_cfg = raw_cfg.get("eval", {}) or {}

    base_seed = int(experiment_cfg.get("seed", 42))
    num_episodes = int(eval_cfg.get("num_episodes", 100))
    greedy = bool(eval_cfg.get("greedy", True))

    env_cfg = build_section(build_env_config, raw_cfg, "env")
    reward_cfg = build_section(build_reward_config, raw_cfg, "reward")
    model_cfg = build_section(build_model_config, raw_cfg, "model")

    run_dir = create_compare_run_dir(raw_cfg)
    eval_dir = run_dir / "eval"

    save_config_copy(eval_dir / "config.yaml", raw_cfg)

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Num episodes per policy: {num_episodes}")
    print(f"Greedy evaluation: {greedy}")

    all_episode_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for policy_spec in policies:
        policy_name = policy_spec["name"]
        policy_type = policy_spec["type"]

        if policy_type not in ["sl", "ppo", "oracle"]:
            raise ValueError(
                f"Policy '{policy_name}' has unsupported type '{policy_type}'. "
                "Expected 'sl', 'ppo', or 'oracle'."
            )

        print(f"\nEvaluating policy: {policy_name} ({policy_type})")

        if policy_type == "oracle":
            model = None
            oracle_mode = policy_spec.get("mode", "hop_first")
            checkpoint_for_summary = ""

            print(f"Oracle mode: {oracle_mode}")

            if oracle_mode not in ["hop_first", "delay_first"]:
                raise ValueError(
                    f"Unsupported oracle mode for policy '{policy_name}': "
                    f"{oracle_mode}. Expected 'hop_first' or 'delay_first'."
                )

        else:
            if "checkpoint" not in policy_spec:
                raise KeyError(
                    f"Policy '{policy_name}' type '{policy_type}' requires "
                    "'checkpoint' field."
                )

            oracle_mode = None
            checkpoint_for_summary = policy_spec["checkpoint"]

            print(f"Checkpoint: {policy_spec['checkpoint']}")

            model = load_policy(
                policy_spec=policy_spec,
                model_cfg=model_cfg,
                device=device,
            )

        policy_rows: List[Dict[str, Any]] = []

        for ep in range(num_episodes):
            episode_seed = base_seed + ep

            row = evaluate_one_episode(
                model=model,
                env_cfg=env_cfg,
                reward_cfg=reward_cfg,
                episode_seed=episode_seed,
                device=device,
                greedy=greedy,
                oracle_mode=oracle_mode,
            )

            row = {
                "policy": policy_name,
                "policy_type": policy_type,
                "oracle_mode": oracle_mode if oracle_mode is not None else "",
                "episode": ep,
                **row,
            }

            policy_rows.append(row)
            all_episode_rows.append(row)

            if (ep + 1) % max(1, num_episodes // 10) == 0:
                print(f"  {ep + 1}/{num_episodes} episodes done")

        summary = summarize_policy_results(policy_name, policy_rows)
        summary["policy_type"] = policy_type
        summary["oracle_mode"] = oracle_mode if oracle_mode is not None else ""
        summary["checkpoint"] = checkpoint_for_summary

        summary_rows.append(summary)

    comparison = {
        "config": {
            "seed": base_seed,
            "num_episodes": num_episodes,
            "greedy": greedy,
            "device": str(device),
        },
        "summary": summary_rows,
    }

    write_json(eval_dir / "comparison.json", comparison)
    write_csv(eval_dir / "comparison.csv", summary_rows)
    write_csv(eval_dir / "episodes.csv", all_episode_rows)

    print_summary_table(summary_rows)

    print("\nSaved:")
    print(f"  {eval_dir / 'comparison.json'}")
    print(f"  {eval_dir / 'comparison.csv'}")
    print(f"  {eval_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()