"""
PPO Agent for OBELIX
Proximal Policy Optimization with:
  - Actor-Critic with shared backbone
  - GAE (Generalized Advantage Estimation)
  - Entropy bonus for exploration
  - Observation stacking support
  - Forward-biased action initialization
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic for discrete actions.

    Architecture:
      obs → [shared_backbone] → shared_features
        ├─→ [actor_head]  → action logits (5)
        └─→ [critic_head] → state value (1)

    Shared backbone reduces parameters and enables feature reuse.
    Separate heads allow policy and value to specialize.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init (standard for PPO) + forward bias on actor."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Actor output: small init for stable early policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic output: unit gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        # Forward bias: make FW (action 2) slightly more likely initially
        with torch.no_grad():
            self.actor[-1].bias.data = torch.tensor(
                [-0.5, -0.2, 0.5, -0.2, -0.5], dtype=torch.float32
            )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        If action is None: sample action from policy.
        If action is given: compute log_prob and entropy for that action.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value



class RolloutBuffer:
    """
    Stores rollout trajectories for PPO updates.
    Uses GAE (Generalized Advantage Estimation) for advantage computation.
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        GAE(λ) = Σ (γλ)^t · δ_t
        where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

        WHY GAE:
        - λ=0: high bias, low variance (equivalent to TD(0))
        - λ=1: low bias, high variance (equivalent to MC)
        - λ=0.95: sweet spot that works for most tasks
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def get_batches(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        mini_batch_size: int,
        device: torch.device,
    ):
        """
        Compute GAE, then yield random mini-batches for PPO update.
        """
        advantages, returns = self.compute_gae(last_value, gamma, gae_lambda)

        # Normalize advantages (crucial for stability)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors
        obs_t = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        n = len(self.obs)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]

            yield (
                obs_t[batch_idx],
                actions_t[batch_idx],
                old_log_probs_t[batch_idx],
                advantages_t[batch_idx],
                returns_t[batch_idx],
            )


class PPOAgent:
    """
    PPO-Clip agent for OBELIX.

    Key hyperparameters:
      - clip_epsilon: 0.2 (standard PPO clip range)
      - entropy_coeff: 0.01 (encourage exploration of actions)
      - value_coeff: 0.5 (balance policy vs value loss)
      - ppo_epochs: 4 (number of passes over rollout data)
      - max_grad_norm: 0.5 (gradient clipping for stability)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # Network
        self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Step counter
        self.total_steps = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> tuple[int, float, float]:
        """
        Select action from policy.
        Returns: (action_idx, log_prob, value)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.network(obs_t)
        dist = Categorical(logits=logits)

        if explore:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for an observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.network(obs_t)
        return float(value.item())

    def store_transition(self, obs, action, log_prob, reward, done, value):
        """Store a transition in the rollout buffer."""
        self.buffer.store(obs, action, log_prob, reward, done, value)
        self.total_steps += 1

    def update(self, last_obs: np.ndarray) -> dict[str, float]:
        """
        Run PPO update on collected rollout data.

        Returns dict with loss components for logging.
        """
        last_value = self.get_value(last_obs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for (
                obs_batch,
                action_batch,
                old_log_prob_batch,
                advantage_batch,
                return_batch,
            ) in self.buffer.get_batches(
                last_value=last_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                mini_batch_size=self.mini_batch_size,
                device=self.device,
            ):
                # Forward pass
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    obs_batch, action_batch
                )

                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (new_value - return_batch).pow(2).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        if n_updates == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "total_loss": (total_policy_loss + total_value_loss) / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
