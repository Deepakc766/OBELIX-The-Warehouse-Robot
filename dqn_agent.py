# from __future__ import annotations

# import random
# from collections import deque
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Deque, Dict, Optional, Sequence, Tuple, Union

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import Tensor

# from model import Net

# ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
# ACTION_TO_INDEX: Dict[str, int] = {action: idx for idx, action in enumerate(ACTIONS)}


# @dataclass
# class Transition:
#     state: np.ndarray
#     action: int
#     reward: float
#     next_state: np.ndarray
#     done: bool


# class ReplayBuffer:
#     def __init__(self, capacity: int) -> None:
#         self.capacity = int(capacity)
#         self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

#     def __len__(self) -> int:
#         return len(self.buffer)

#     def push(
#         self,
#         state: np.ndarray,
#         action: int,
#         reward: float,
#         next_state: np.ndarray,
#         done: bool,
#     ) -> None:
#         self.buffer.append(
#             Transition(
#                 state=np.array(state, dtype=np.float32, copy=True),
#                 action=int(action),
#                 reward=float(reward),
#                 next_state=np.array(next_state, dtype=np.float32, copy=True),
#                 done=bool(done),
#             )
#         )

#     def sample(
#         self, batch_size: int
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         batch = random.sample(self.buffer, batch_size)
#         states = np.stack([item.state for item in batch], axis=0)
#         actions = np.array([item.action for item in batch], dtype=np.int64)
#         rewards = np.array([item.reward for item in batch], dtype=np.float32)
#         next_states = np.stack([item.next_state for item in batch], axis=0)
#         dones = np.array([item.done for item in batch], dtype=np.float32)
#         return states, actions, rewards, next_states, dones


# class DQNAgent:
#     def __init__(
#         self,
#         obs_dim: int = 18,
#         action_dim: int = 5,
#         gamma: float = 0.99,
#         lr: float = 1e-3,
#         batch_size: int = 64,
#         buffer_size: int = 100_000,
#         min_buffer_size: int = 1_000,
#         epsilon_start: float = 1.0,
#         epsilon_end: float = 0.05,
#         epsilon_decay: int = 25_000,
#         target_update_freq: int = 1_000,
#         device: Optional[str] = None,
#     ) -> None:
#         if action_dim != len(ACTIONS):
#             raise ValueError(f"action_dim must be {len(ACTIONS)} for OBELIX action space")

#         self.obs_dim = int(obs_dim)
#         self.action_dim = int(action_dim)
#         self.gamma = float(gamma)
#         self.batch_size = int(batch_size)
#         self.min_buffer_size = int(min_buffer_size)

#         self.epsilon = float(epsilon_start)
#         self.epsilon_start = float(epsilon_start)
#         self.epsilon_end = float(epsilon_end)
#         self.epsilon_decay = int(max(1, epsilon_decay))

#         self.target_update_freq = int(max(1, target_update_freq))
#         self.train_updates = 0

#         selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = torch.device(selected_device)

#         self.online_net = Net(input_dim=self.obs_dim, output_dim=self.action_dim).to(self.device)
#         self.target_net = Net(input_dim=self.obs_dim, output_dim=self.action_dim).to(self.device)
#         self.target_net.load_state_dict(self.online_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
#         self.replay_buffer = ReplayBuffer(capacity=buffer_size)

#     def __len__(self) -> int:
#         return len(self.replay_buffer)

#     def _obs_to_tensor(self, obs: np.ndarray) -> Tensor:
#         obs_array = np.asarray(obs, dtype=np.float32)
#         if obs_array.shape != (self.obs_dim,):
#             raise ValueError(f"Expected observation shape ({self.obs_dim},), got {obs_array.shape}")
#         return torch.from_numpy(obs_array).unsqueeze(0).to(self.device)

#     def _action_to_index(self, action: Union[int, str]) -> int:
#         if isinstance(action, str):
#             if action not in ACTION_TO_INDEX:
#                 raise ValueError(f"Invalid action string: {action}")
#             return ACTION_TO_INDEX[action]

#         action_idx = int(action)
#         if action_idx < 0 or action_idx >= self.action_dim:
#             raise ValueError(f"Invalid action index: {action_idx}")
#         return action_idx

#     def select_action_index(
#         self,
#         obs: np.ndarray,
#         rng: Optional[np.random.Generator] = None,
#         explore: bool = True,
#         inference_epsilon: float = 0.0,
#     ) -> int:
#         generator = rng if rng is not None else np.random.default_rng()
#         epsilon = self.epsilon if explore else float(max(0.0, inference_epsilon))

#         if generator.random() < epsilon:
#             return int(generator.integers(0, self.action_dim))

#         with torch.no_grad():
#             q_values = self.online_net(self._obs_to_tensor(obs))
#             return int(torch.argmax(q_values, dim=1).item())

#     def select_action_str(
#         self,
#         obs: np.ndarray,
#         rng: Optional[np.random.Generator] = None,
#         explore: bool = True,
#         inference_epsilon: float = 0.0,
#     ) -> str:
#         action_index = self.select_action_index(
#             obs=obs,
#             rng=rng,
#             explore=explore,
#             inference_epsilon=inference_epsilon,
#         )
#         return ACTIONS[action_index]

#     def store_transition(
#         self,
#         state: np.ndarray,
#         action: Union[int, str],
#         reward: float,
#         next_state: np.ndarray,
#         done: bool,
#     ) -> None:
#         action_index = self._action_to_index(action)
#         self.replay_buffer.push(state, action_index, reward, next_state, done)

#     def _update_epsilon(self) -> None:
#         step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_decay)
#         self.epsilon = max(self.epsilon_end, self.epsilon - step)

#     def update_target_network(self) -> None:
#         self.target_net.load_state_dict(self.online_net.state_dict())

#     def train_step(self) -> Dict[str, float]:
#         if len(self.replay_buffer) < max(self.batch_size, self.min_buffer_size):
#             return {
#                 "trained": 0.0,
#                 "loss": 0.0,
#                 "epsilon": float(self.epsilon),
#                 "buffer_size": float(len(self.replay_buffer)),
#             }

#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

#         states_tensor = torch.from_numpy(states).to(self.device)
#         actions_tensor = torch.from_numpy(actions).to(self.device)
#         rewards_tensor = torch.from_numpy(rewards).to(self.device)
#         next_states_tensor = torch.from_numpy(next_states).to(self.device)
#         dones_tensor = torch.from_numpy(dones).to(self.device)

#         q_values = self.online_net(states_tensor)
#         q_selected = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

#         with torch.no_grad():
#             q_next = self.target_net(next_states_tensor).max(dim=1).values
#             q_target = rewards_tensor + (1.0 - dones_tensor) * self.gamma * q_next

#         loss = F.smooth_l1_loss(q_selected, q_target)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.train_updates += 1
#         if self.train_updates % self.target_update_freq == 0:
#             self.update_target_network()

#         self._update_epsilon()

#         return {
#             "trained": 1.0,
#             "loss": float(loss.item()),
#             "epsilon": float(self.epsilon),
#             "buffer_size": float(len(self.replay_buffer)),
#             "q_mean": float(q_selected.detach().mean().item()),
#         }

#     def save(self, path: Union[str, Path]) -> None:
#         checkpoint_path = Path(path)
#         checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

#         checkpoint = {
#             "model_state_dict": self.online_net.state_dict(),
#             "target_state_dict": self.target_net.state_dict(),
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "epsilon": self.epsilon,
#             "train_updates": self.train_updates,
#             "obs_dim": self.obs_dim,
#             "action_dim": self.action_dim,
#         }
#         torch.save(checkpoint, checkpoint_path)

#     def load(self, path: Union[str, Path], map_location: str = "cpu") -> None:
#         checkpoint = torch.load(Path(path), map_location=map_location)

#         if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#             self.online_net.load_state_dict(checkpoint["model_state_dict"])
#             self.target_net.load_state_dict(
#                 checkpoint.get("target_state_dict", checkpoint["model_state_dict"])
#             )

#             optimizer_state = checkpoint.get("optimizer_state_dict")
#             if optimizer_state is not None:
#                 self.optimizer.load_state_dict(optimizer_state)

#             self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
#             self.train_updates = int(checkpoint.get("train_updates", self.train_updates))
#         else:
#             self.online_net.load_state_dict(checkpoint)
#             self.target_net.load_state_dict(self.online_net.state_dict())

#         self.online_net.to(self.device)
#         self.target_net.to(self.device)
#         self.target_net.eval()






#Harsh vala niche hai 






import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from model import Net

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=18, action_dim=5, lr=1e-4, gamma=0.99, buffer_capacity=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.policy_net = Net(state_dim, action_dim)
        self.target_net = Net(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, rng=None):
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        else:
            if rng:
                idx = rng.integers(0, self.action_dim)
                return int(idx)
            return random.randrange(self.action_dim)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())



ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
def policy(obs, rng):
    """
    Evaluation policy for evaluate.py
    """
    state_dim = 18
    action_dim = 5
    model = Net(state_dim, action_dim)
    import os
    weights_path = os.path.join(os.path.dirname(__file__), "obelix_ddqn_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    state = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action_idx = model(state).max(1)[1].item()
    return ACTIONS[action_idx]