"""
Submission agent for Codabench evaluation.

Supports BOTH architectures (auto-detected from weights.pth):
  - D3QN: DuelingQNetwork with hidden_dims=(256, 128)
  - PPO:  ActorCritic with shared backbone (256)

Input: 6 stacked observations (108-dim) for short-term memory
Loads weights from weights.pth in the same directory.
"""

import numpy as np
import os
from collections import deque

_MODEL = None
_TORCH = None
_STACKER = None
_PREV_RAW_OBS = None
_MODEL_TYPE = None  # "d3qn" or "ppo"
ACTIONS = ("L45", "L22", "FW", "R22", "R45")


N_STACK = 6
OBS_DIM = 18
STACKED_DIM = N_STACK * OBS_DIM  # 108


class ObservationStacker:
    """Stacks last N observations into a single flat vector."""

    def __init__(self, n_stack=6, obs_dim=18):
        self.n_stack = n_stack
        self.obs_dim = obs_dim
        self.frames = deque(maxlen=n_stack)

    def reset(self, obs):
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs_flat.copy())
        return np.concatenate(list(self.frames), axis=0)

    def push(self, obs):
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        if len(self.frames) == 0:
            return self.reset(obs)
        self.frames.append(obs_flat.copy())
        return np.concatenate(list(self.frames), axis=0)


def _detect_episode_reset(prev_obs, curr_obs):
    if prev_obs is None:
        return True
    prev = np.asarray(prev_obs, dtype=np.float32).ravel()[:OBS_DIM]
    curr = np.asarray(curr_obs, dtype=np.float32).ravel()[:OBS_DIM]
    return int(np.sum(prev != curr)) >= 10



class _DuelingQNetwork:
    """Factory that creates a DuelingQNetwork matching D3QN training."""

    def __new__(cls, input_dim=108, output_dim=5, hidden_dims=(256, 128)):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dim1, hidden_dim2 = hidden_dims
                self.feature = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim1),
                    nn.ReLU(),
                    nn.Linear(hidden_dim1, hidden_dim2),
                    nn.ReLU(),
                )
                self.value_stream = nn.Sequential(
                    nn.Linear(hidden_dim2, hidden_dim2 // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2 // 2, 1),
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(hidden_dim2, hidden_dim2 // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2 // 2, output_dim),
                )

            def forward(self, x):
                features = self.feature(x)
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                return value + (advantage - advantage.mean(dim=1, keepdim=True))

        return _Net()




class _ActorCritic:
    """Factory that creates an ActorCritic matching PPO training."""

    def __new__(cls, input_dim=108, output_dim=5, hidden_dim=256):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.actor = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, output_dim),
                )
                self.critic = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                )

            def forward(self, x):
                features = self.backbone(x)
                logits = self.actor(features)
                return logits  # only need logits for inference

        return _Net()


def _fallback_policy(obs, rng):
    """Forward-biased random walk with stuck avoidance."""
    is_stuck = len(obs) > 17 and obs[17] == 1
    if is_stuck:
        probs = np.array([0.25, 0.25, 0.0, 0.25, 0.25])
    else:
        probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05])
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]


def _load_once():
    global _MODEL, _TORCH, _STACKER, _MODEL_TYPE
    if _MODEL is not None:
        return

    try:
        import torch
        _TORCH = torch
        _STACKER = ObservationStacker(n_stack=N_STACK, obs_dim=OBS_DIM)

        path = os.path.join(os.path.dirname(__file__), "weights.pth")
        if not os.path.exists(path):
            print("No weights.pth found, using fallback policy")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        
        if isinstance(checkpoint, dict) and "network" in checkpoint:
            # PPO checkpoint: {"network": ..., "optimizer": ..., "total_steps": ...}
            state_dict = checkpoint["network"]
            _MODEL_TYPE = "ppo"
            _MODEL = _ActorCritic(input_dim=STACKED_DIM, output_dim=5, hidden_dim=256)
            _MODEL.load_state_dict(state_dict)
            print("Loaded PPO weights")
        else:
            # D3QN checkpoint: {"model_state_dict": ...} or raw state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            _MODEL_TYPE = "d3qn"
            _MODEL = _DuelingQNetwork(input_dim=STACKED_DIM, output_dim=5, hidden_dims=(256, 128))
            _MODEL.load_state_dict(state_dict)
            print("Loaded D3QN weights")

        _MODEL.eval()

    except Exception as exc:
        _MODEL = None
        print(f"Weight load error: {exc}")


def policy(obs, rng=None):
    """Main evaluation function for Codabench."""
    global _PREV_RAW_OBS

    if rng is None:
        rng = np.random.default_rng()

    _load_once()

    if _MODEL is None or _TORCH is None or _STACKER is None:
        return _fallback_policy(obs, rng)

    # Detect episode boundaries and reset stacker
    if _detect_episode_reset(_PREV_RAW_OBS, obs):
        stacked_obs = _STACKER.reset(obs)
    else:
        stacked_obs = _STACKER.push(obs)

    _PREV_RAW_OBS = np.asarray(obs, dtype=np.float32).ravel()[:OBS_DIM].copy()

    state_tensor = _TORCH.from_numpy(stacked_obs).unsqueeze(0)
    with _TORCH.no_grad():
        output = _MODEL(state_tensor)

        if _MODEL_TYPE == "ppo":
            # PPO: output is logits → pick argmax (greedy)
            action_idx = int(_TORCH.argmax(output, dim=1).item())
        else:
            # D3QN: output is Q-values → pick argmax
            action_idx = int(_TORCH.argmax(output, dim=1).item())

    return ACTIONS[action_idx]
