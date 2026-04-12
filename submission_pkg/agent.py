import numpy as np
import os

_MODEL = None
_TORCH = None
_LOAD_ERROR = None
ACTIONS = ("L45", "L22", "FW", "R22", "R45")


class Net:
    def __new__(cls, input_dim=18, output_dim=5, hidden_dims=(256, 128)):
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


def _fallback_policy(rng: np.random.Generator) -> str:
    probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]

def _load_once():
    """Loads the model weights one time during evaluation."""
    global _MODEL, _TORCH, _LOAD_ERROR
    if _MODEL is None:
        try:
            import torch

            _TORCH = torch
            _MODEL = Net(input_dim=18, output_dim=5)

            path = os.path.join(os.path.dirname(__file__), "weights.pth")
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                _MODEL.load_state_dict(state_dict)

            _MODEL.eval()
        except Exception as exc:
            _MODEL = None
            _LOAD_ERROR = str(exc)

def policy(obs, rng=None) -> str:
    """
    Main evaluation function for Codabench.
    """
    if rng is None:
        rng = np.random.default_rng()

    _load_once()

    if _MODEL is None or _TORCH is None:
        return _fallback_policy(rng)

    state_tensor = _TORCH.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    with _TORCH.no_grad():
        q_values = _MODEL(state_tensor)
        action_idx = int(_TORCH.argmax(q_values, dim=1).item())

    return ACTIONS[action_idx]
