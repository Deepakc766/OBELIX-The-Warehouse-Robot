"""
Exploration Bonus Modules for OBELIX

Two complementary exploration strategies:

1. CountBonus (SimHash)
   - Tracks visit counts for discretized observations
   - Bonus = β / √N(hash(obs))
   - Reference: Tang et al., "#Exploration" (NeurIPS 2017)

2. RNDModule (Random Network Distillation)
   - Fixed random target net + trainable predictor net
   - Bonus = normalized ||predictor(obs) - target(obs)||²
   - Reference: Burda et al., "Exploration by RND" (ICLR 2019)

Both are ADDITIVE to the existing PBRS reward shaping:
  total_reward = env_reward + PBRS_shaping + count_bonus + rnd_bonus
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CountBonus:
    """
    Count-based exploration bonus using direct hashing of binary observations.

    Since OBELIX observations are binary (0/1), we can hash them directly
    without SimHash — the raw binary vector IS a natural hash key.

    For stacked observations (e.g., 108-dim = 6×18), we hash only the
    CURRENT observation (last 18 elements) to keep counts meaningful.

    Bonus formula:  bonus = β / √N(key)
      - First visit:    β / √1 = β        (strong exploration push)
      - 100th visit:    β / √100 = β/10   (weak push, area well-explored)
    """

    def __init__(self, beta: float = 0.5, obs_dim: int = 18) -> None:
        self.beta = beta
        self.obs_dim = obs_dim
        self.counts: dict[tuple, int] = defaultdict(int)

    def get_bonus(self, stacked_obs: np.ndarray) -> float:
        """Compute count-based bonus for the current observation."""
        curr_obs = np.asarray(stacked_obs, dtype=np.float32)[-self.obs_dim:]
        key = tuple(curr_obs.astype(int).tolist())
        self.counts[key] += 1
        return self.beta / np.sqrt(self.counts[key])

    def stats(self) -> dict[str, float]:
        """Return statistics about the count table."""
        if not self.counts:
            return {"unique_states": 0, "total_visits": 0, "max_count": 0}
        counts_arr = np.array(list(self.counts.values()))
        return {
            "unique_states": len(self.counts),
            "total_visits": int(counts_arr.sum()),
            "max_count": int(counts_arr.max()),
            "mean_count": float(counts_arr.mean()),
        }



class RNDModule:
    """
    Random Network Distillation for curiosity-driven exploration.

    Architecture:
      target_net:    obs_dim → 64 → 32  (FIXED random weights, never trained)
      predictor_net: obs_dim → 64 → 32  (TRAINED to match target output)

    Bonus = normalized MSE between predictor and target outputs.

    WHY it works:
      - Novel states: predictor hasn't seen them → HIGH prediction error → HIGH bonus
      - Familiar states: predictor learned them → LOW error → LOW bonus
      - The bonus naturally decays as the agent explores → no manual annealing needed

    WHY normalization matters:
      - Raw MSE can vary wildly (0.001 to 100+)
      - We maintain a running mean/std of prediction errors
      - Normalized bonus = (raw_error - running_mean) / running_std
      - This keeps the bonus on a stable scale relative to env rewards
    """

    def __init__(
        self,
        obs_dim: int = 108,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.beta = beta
        self.device = device or torch.device("cpu")

        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        ).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.predictor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)

        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0
        self._momentum = 0.99

    @torch.no_grad()
    def compute_bonus(self, obs: np.ndarray) -> float:
        """
        Compute the normalized RND exploration bonus for a single observation.
        """
        obs_t = torch.from_numpy(
            np.asarray(obs, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        target_features = self.target_net(obs_t)
        predicted_features = self.predictor_net(obs_t)

        # Raw prediction error (MSE per sample)
        raw_error = float(
            (target_features - predicted_features).pow(2).mean().item()
        )

        # Update running stats (EMA)
        self._count += 1
        if self._count == 1:
            self._running_mean = raw_error
            self._running_var = 1.0
        else:
            delta = raw_error - self._running_mean
            self._running_mean += (1 - self._momentum) * delta
            self._running_var = (
                self._momentum * self._running_var
                + (1 - self._momentum) * delta * delta
            )

        # Normalize: (error - mean) / std, clipped to [0, 5]
        std = max(np.sqrt(self._running_var), 1e-8)
        normalized = max(0.0, (raw_error - self._running_mean) / std)
        normalized = min(normalized, 5.0)  # clip to prevent outliers

        return self.beta * normalized

    def train_step(self, obs_batch: np.ndarray) -> float:
        """
        Train the predictor network on a batch of observations.
        Call this once per agent learning step.

        Args:
            obs_batch: numpy array of shape (batch_size, obs_dim)

        Returns:
            predictor_loss (float)
        """
        obs_t = torch.from_numpy(
            np.asarray(obs_batch, dtype=np.float32)
        ).to(self.device)

        with torch.no_grad():
            target_features = self.target_net(obs_t)

        predicted_features = self.predictor_net(obs_t)
        loss = (target_features - predicted_features).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
