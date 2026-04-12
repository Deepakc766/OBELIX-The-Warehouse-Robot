from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DuelingQNetwork




@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool




class ObservationStacker:
    """
    Stacks the last N observations into a single flat vector.

    WHY: The OBELIX environment is partially observable — a single 18-dim
    binary sensor reading tells the agent what it sees RIGHT NOW, but not:
      - Which direction it was heading
      - Whether it was turning or going straight
      - Whether the box just appeared or has been visible for a while
      - Whether it's stuck (obs[17]=1) for the 1st or 10th step

    HOW: By concatenating the last N observations:
      Single obs:  [s₁, s₂, ..., s₁₈]              → 18 dims
      Stacked (4): [s₁ᵗ⁻³...s₁₈ᵗ⁻³, s₁ᵗ⁻²...s₁₈ᵗ⁻², s₁ᵗ⁻¹...s₁₈ᵗ⁻¹, s₁ᵗ...s₁₈ᵗ] → 72 dims

    The agent can now learn temporal patterns like:
      "Sensors detected something on the left 2 steps ago,
       then nothing, then nothing → the box is behind me, I should turn"

    USAGE:
      stacker = ObservationStacker(n_stack=4, obs_dim=18)
      stacked_obs = stacker.reset(first_obs)    # returns 72-dim
      stacked_obs = stacker.push(next_obs)      # returns 72-dim
    """

    def __init__(self, n_stack: int = 4, obs_dim: int = 18) -> None:
        self.n_stack = n_stack
        self.obs_dim = obs_dim
        self.frames: deque = deque(maxlen=n_stack)

    @property
    def stacked_dim(self) -> int:
        """Total dimension of the stacked observation."""
        return self.n_stack * self.obs_dim

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Fill all slots with the initial observation and return stacked."""
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs_flat.copy())
        return self._get_stacked()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Push a new observation and return the updated stack."""
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        self.frames.append(obs_flat.copy())
        return self._get_stacked()

    def _get_stacked(self) -> np.ndarray:
        """Concatenate all frames into a single flat vector."""
        return np.concatenate(list(self.frames), axis=0)




class PrioritizedReplayBuffer:
    """Sum-tree based PER for efficient priority sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer: list[Transition] = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float64)
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        max_priority = float(self.priorities.max()) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max(max_priority, 1e-6)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = self.priorities[: len(self.buffer)]
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        states = np.stack([t.state for t in samples], axis=0)
        actions = np.array([t.action for t in samples], dtype=np.int64)
        rewards = np.array([t.reward for t in samples], dtype=np.float32)
        next_states = np.stack([t.next_state for t in samples], axis=0)
        dones = np.array([t.done for t in samples], dtype=np.float32)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = float(max(priority, 1e-6))




class D3QNAgent:
    # Forward-biased exploration probabilities
    EXPLORE_PROBS = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)
    # When stuck, only turn (never go forward)
    STUCK_PROBS = np.array([0.25, 0.25, 0.0, 0.25, 0.25], dtype=np.float64)

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 5,
        hidden_dims: tuple[int, int] = (256, 128),
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 200_000,
        target_update_freq: int = 2000,
        # PER parameters
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 200_000,
        per_eps: float = 1e-5,
        # n-step parameters
        n_step: int = 3,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(max(1, epsilon_decay))
        self.epsilon = float(epsilon_start)

        self.target_update_freq = int(max(1, target_update_freq))

        # PER
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = int(max(1, per_beta_frames))
        self.per_eps = float(per_eps)

        # n-step
        self.n_step = int(max(1, n_step))
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.n_step)

        if device is not None:
            selected_device = device
        elif torch.cuda.is_available():
            selected_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_device = "mps"
        else:
            selected_device = "cpu"
        self.device = torch.device(selected_device)

        self.policy_net = DuelingQNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=per_alpha)

        self.steps_done = 0
        self.learn_steps = 0


    def _compute_epsilon(self) -> float:
        ratio = min(1.0, self.steps_done / float(self.epsilon_decay))
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    def _compute_beta(self) -> float:
        ratio = min(1.0, self.learn_steps / float(self.per_beta_frames))
        return self.per_beta_start + ratio * (1.0 - self.per_beta_start)


    def select_action(
        self,
        state: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        explore: bool = True,
        inference_epsilon: float = 0.0,
    ) -> int:
        generator = rng if rng is not None else np.random.default_rng()

        self.epsilon = self._compute_epsilon()
        eps = self.epsilon if explore else float(max(0.0, inference_epsilon))

        self.steps_done += 1

        if generator.random() < eps:
            # Forward-biased exploration
            is_stuck = (state is not None and len(state) >= 18 and state[-1] == 1)
            probs = self.STUCK_PROBS if is_stuck else self.EXPLORE_PROBS
            return int(generator.choice(self.action_dim, p=probs))

        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(state_tensor).argmax(dim=1).item())


    def _compute_n_step_transition(self) -> Optional[Transition]:
        if len(self.n_step_buffer) < self.n_step:
            return None

        reward_sum = 0.0
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward_sum += (self.gamma ** idx) * float(r)
            next_state = ns
            if d:
                done = True
                break

        state, action, _, _, _ = self.n_step_buffer[0]
        return Transition(
            state=np.array(state, dtype=np.float32, copy=True),
            action=int(action),
            reward=float(reward_sum),
            next_state=np.array(next_state, dtype=np.float32, copy=True),
            done=bool(done),
        )

    def _flush_n_step_on_done(self) -> None:
        while self.n_step_buffer:
            transition = self._compute_n_step_transition()
            if transition is not None:
                self.memory.push(transition)
            self.n_step_buffer.popleft()


    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        if self.n_step == 1:
            transition = Transition(
                state=np.array(state, dtype=np.float32, copy=True),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_state, dtype=np.float32, copy=True),
                done=bool(done),
            )
            self.memory.push(transition)
        else:
            self.n_step_buffer.append((
                np.array(state, dtype=np.float32, copy=True),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32, copy=True),
                bool(done),
            ))
            transition = self._compute_n_step_transition()
            if transition is not None:
                self.memory.push(transition)
                self.n_step_buffer.popleft()
            if done:
                self._flush_n_step_on_done()

        if len(self.memory) >= self.batch_size:
            return self.learn()

        return None


    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        beta = self._compute_beta()
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.batch_size, beta=beta
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: use policy_net to select actions, target_net to evaluate
        gamma_n = self.gamma ** self.n_step
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * gamma_n * next_q

        # Weighted Huber loss
        td_errors = targets - q_values
        losses = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights_t * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.detach().abs().cpu().numpy() + self.per_eps
        self.memory.update_priorities(indices=indices, priorities=new_priorities)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())


    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load(self, path: Union[str, Path], map_location: str = "cpu") -> None:
        checkpoint = torch.load(Path(path), map_location=map_location)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(
                checkpoint.get("target_state_dict", checkpoint["model_state_dict"])
            )
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
            self.steps_done = int(checkpoint.get("steps_done", self.steps_done))
            self.learn_steps = int(checkpoint.get("learn_steps", self.learn_steps))
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
