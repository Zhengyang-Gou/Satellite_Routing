# trainers/ppo_trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from env.rl_sat_env import RLSatelliteEnv
from models.actor_critic_net import ActorCriticNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import (
    EnvConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    build_env_config,
    build_model_config,
    build_ppo_config,
    build_reward_config,
)
from utils.logger import CSVLogger
from utils.metrics import summarize_episode_stats
from utils.paths import (
    get_best_checkpoint_path,
    get_latest_checkpoint_path,
    get_metrics_path,
    resolve_project_path,
)
from utils.seed import set_seed


def choose_device() -> torch.device:
    """
    Choose training device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class RolloutBuffer:
    """
    PPO rollout buffer.

    Stores one or more episodes of transitions.
    """

    def __init__(self) -> None:
        self.adj_list: List[torch.Tensor] = []
        self.feat_list: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def add(
        self,
        adj: torch.Tensor,
        feat: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        self.adj_list.append(adj.squeeze(0).detach().cpu())
        self.feat_list.append(feat.squeeze(0).detach().cpu())
        self.actions.append(int(action))
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(value.detach().cpu())

    def extend(self, other: "RolloutBuffer") -> None:
        self.adj_list.extend(other.adj_list)
        self.feat_list.extend(other.feat_list)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.values.extend(other.values)

    def clear(self) -> None:
        self.adj_list.clear()
        self.feat_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.actions)


class PPOTrainer:
    """
    PPO trainer for satellite routing.
    """

    def __init__(
        self,
        env: RLSatelliteEnv,
        model: ActorCriticNet,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        k_epochs: int = 10,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
    ) -> None:
        self.env = env
        self.model = model.to(device)
        self.device = device

        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.value_loss_fn = nn.MSELoss()

    def _compute_returns(
        self,
        rewards: List[float],
        dones: List[bool],
    ) -> torch.Tensor:
        """
        Compute discounted Monte Carlo returns.
        """
        returns: List[float] = []
        discounted_reward = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0

            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run PPO update using data stored in buffer.
        """
        if len(buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }

        old_states_adj = torch.stack(buffer.adj_list).to(self.device)
        old_states_feat = torch.stack(buffer.feat_list).to(self.device)
        old_actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(buffer.log_probs).to(self.device).detach()
        old_values = torch.stack(buffer.values).to(self.device).detach()

        returns = self._compute_returns(buffer.rewards, buffer.dones)

        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.k_epochs):
            logits, state_values = self.model(old_states_adj, old_states_feat)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.eps_clip,
                1.0 + self.eps_clip,
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_loss_fn(state_values, returns)

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.mean().item())

        return {
            "policy_loss": total_policy_loss / self.k_epochs,
            "value_loss": total_value_loss / self.k_epochs,
            "entropy": total_entropy / self.k_epochs,
        }

    @torch.no_grad()
    def collect_episode(self, greedy: bool = False) -> Tuple[RolloutBuffer, Dict[str, Any]]:
        """
        Collect one full episode.
        """
        buffer = RolloutBuffer()
        state, info = self.env.reset()
        done = False

        while not done:
            adj_t = torch.tensor(
                state["adjacency"],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            feat_t = torch.tensor(
                state["node_features"],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            logits, value = self.model(adj_t, feat_t)
            logits_1d = logits.squeeze(0)

            if greedy:
                action = torch.argmax(logits_1d, dim=-1)
                dist = Categorical(logits=logits_1d)
                log_prob = dist.log_prob(action)
            else:
                dist = Categorical(logits=logits_1d)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, info = self.env.step(int(action.item()))

            buffer.add(
                adj=adj_t,
                feat=feat_t,
                action=int(action.item()),
                log_prob=log_prob,
                reward=float(reward),
                done=bool(done),
                value=value.squeeze(0),
            )

            state = next_state

        episode_stats = {
            "success": bool(info["success"]),
            "steps": int(info["step_count"]),
            "coverage_ratio": float(info["coverage_ratio"]),
            "total_delay": float(info["total_delay"]),
            "termination_reason": info["termination_reason"],
            "episode_return": float(sum(buffer.rewards)),
            "repeat_count": int(info.get("repeat_count", 0)),
            "repeat_ratio": float(info.get("repeat_ratio", 0.0)),
        }

        return buffer, episode_stats


def build_env_from_config(cfg: Dict[str, Any]) -> RLSatelliteEnv:
    """
    Build RL environment from config dictionary.
    """
    env_cfg: EnvConfig = build_env_config(cfg)
    reward_cfg: RewardConfig = build_reward_config(cfg)
    return RLSatelliteEnv(env_cfg=env_cfg, reward_cfg=reward_cfg)


def build_model_from_config(
    cfg: Dict[str, Any],
    device: torch.device,
) -> ActorCriticNet:
    """
    Build actor-critic model from config dictionary.
    """
    model_cfg: ModelConfig = build_model_config(cfg)

    model = ActorCriticNet(
        input_dim=model_cfg.input_dim,
        hidden_dim=model_cfg.hidden_dim,
    )

    return model.to(device)


def load_pretrained_if_needed(
    model: ActorCriticNet,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> None:
    """
    Load pretrained weights if checkpoint_path is not None.

    Supports two formats:
        1. New checkpoint format:
            {"model_state_dict": ...}

        2. Old raw state_dict format:
            {"gnn.conv1.linear.weight": ...}
    """
    if checkpoint_path is None:
        return

    resolved_path = resolve_project_path(checkpoint_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {resolved_path}")

    raw = torch.load(resolved_path, map_location=device)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        loaded_state = raw["model_state_dict"]
    else:
        loaded_state = raw

    current_state = model.state_dict()

    compatible_state = {}
    skipped_keys = []

    for key, value in loaded_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    print(f"Loaded compatible pretrained weights from: {resolved_path}")

    if skipped_keys:
        print(f"Skipped incompatible keys: {skipped_keys}")

    if missing:
        print(f"Missing keys after partial load: {missing}")

    if unexpected:
        print(f"Unexpected keys after partial load: {unexpected}")


@torch.no_grad()
def evaluate_policy(
    trainer: PPOTrainer,
    num_episodes: int = 20,
    greedy: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate policy for several episodes.
    """
    old_mode = trainer.model.training
    trainer.model.eval()

    episode_stats = []

    for _ in range(num_episodes):
        _, stats = trainer.collect_episode(greedy=greedy)
        episode_stats.append(stats)

    if old_mode:
        trainer.model.train()

    return summarize_episode_stats(episode_stats)


def train_ppo_from_config(
    cfg: Dict[str, Any],
    run_dir: str | Path,
) -> Path:
    """
    Main PPO training entry used by scripts/train_ppo.py.

    Args:
        cfg:
            Full config dictionary loaded from YAML.
        run_dir:
            Output directory created by utils.paths.create_run_dir.

    Returns:
        Path to best checkpoint.
    """
    run_dir = Path(run_dir)

    experiment_cfg = cfg.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    ppo_cfg: PPOConfig = build_ppo_config(cfg)

    device = choose_device()
    print(f"Using device: {device}")

    env = build_env_from_config(cfg)
    model = build_model_from_config(cfg, device)

    load_pretrained_if_needed(
        model=model,
        checkpoint_path=ppo_cfg.pretrained_checkpoint,
        device=device,
    )

    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        lr=ppo_cfg.lr,
        gamma=ppo_cfg.gamma,
        k_epochs=ppo_cfg.k_epochs,
        eps_clip=ppo_cfg.eps_clip,
        value_coef=ppo_cfg.value_coef,
        entropy_coef=ppo_cfg.entropy_coef,
        grad_clip=ppo_cfg.grad_clip,
    )

    metrics_path = get_metrics_path(run_dir)
    logger = CSVLogger(metrics_path)

    best_eval_score: Optional[Tuple[float, float, float, float]] = None
    best_checkpoint_path = get_best_checkpoint_path(run_dir)
    latest_checkpoint_path = get_latest_checkpoint_path(run_dir)

    print("Starting PPO training")
    print(f"Run dir: {run_dir}")

    for episode in range(1, ppo_cfg.num_episodes + 1):
        buffer, stats = trainer.collect_episode(greedy=False)
        train_logs = trainer.update(buffer)

        log_row = {
            "episode": episode,
            "episode_return": stats["episode_return"],
            "success": int(stats["success"]),
            "steps": stats["steps"],
            "coverage_ratio": stats["coverage_ratio"],
            "total_delay": stats["total_delay"],
            "repeat_count": stats["repeat_count"],
            "repeat_ratio": stats["repeat_ratio"],
            "termination_reason": stats["termination_reason"],
            "policy_loss": train_logs["policy_loss"],
            "value_loss": train_logs["value_loss"],
            "entropy": train_logs["entropy"],
        }
        logger.log(log_row)

        if episode % 10 == 0:
            print(
                f"Episode {episode:05d} | "
                f"Return {stats['episode_return']:+8.3f} | "
                f"Success {int(stats['success'])} | "
                f"Coverage {stats['coverage_ratio']:.1%} | "
                f"Repeat {stats['repeat_count']} | "
                f"Reason {stats['termination_reason']}"
            )

        if episode % ppo_cfg.eval_every == 0:
            eval_metrics = evaluate_policy(
                trainer=trainer,
                num_episodes=20,
                greedy=True,
            )

            success_rate = float(eval_metrics["success_rate"])

            print(
                f"[Eval @ Episode {episode}] "
                f"Success Rate: {success_rate:.2%} | "
                f"Mean Delay: {eval_metrics['mean_total_delay']:.4f} | "
                f"Mean Steps: {eval_metrics['mean_steps']:.2f} | "
                f"Mean Repeat: {eval_metrics['mean_repeat_count']:.2f}"
            )

            eval_score = (
                success_rate,
                -float(eval_metrics["mean_repeat_count"]),
                -float(eval_metrics["mean_total_delay"]),
                -float(eval_metrics["mean_steps"]),
            )

            if best_eval_score is None or eval_score > best_eval_score:
                best_eval_score = eval_score

                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=trainer.optimizer,
                    config=cfg,
                    metrics={
                        **eval_metrics,
                        "best_eval_score": list(best_eval_score),
                    },
                    episode=episode,
                )

                print(f"Saved best checkpoint: {best_checkpoint_path}")

        save_latest_every = int(cfg.get("logging", {}).get("save_latest_every", 100))

        if episode % save_latest_every == 0:
            save_checkpoint(
                path=latest_checkpoint_path,
                model=model,
                optimizer=trainer.optimizer,
                config=cfg,
                metrics=stats,
                episode=episode,
            )

    save_checkpoint(
        path=latest_checkpoint_path,
        model=model,
        optimizer=trainer.optimizer,
        config=cfg,
        metrics={
            "best_eval_score": list(best_eval_score) if best_eval_score else None,
        },
        episode=ppo_cfg.num_episodes,
    )

    logger.close()

    print("PPO training finished")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")

    return best_checkpoint_path
