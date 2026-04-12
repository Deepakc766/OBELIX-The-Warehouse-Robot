import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim=18, output_dim=5):
        """
        Deep Q-Network for the OBELIX robot.
        Input: 18 bits (16 sonar, 1 IR, 1 attachment)
        Output: 5 actions ("L45", "L22", "FW", "R22", "R45")
        """
        super(Net, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class NoisyLinear(nn.Module):
    """
    Linear layer with learnable parametric noise.

    Instead of: y = Wx + b                           (standard Linear)
    We compute: y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)

    Where:
      μ_w, μ_b = learnable mean weights/biases
      σ_w, σ_b = learnable noise magnitude (starts at 0.5/√fan_in)
      ε_w, ε_b = random noise (resampled every forward pass during training)

    WHY: The network LEARNS when and how much to explore:
      - σ large → high noise → more exploration in that part of state space
      - σ small → low noise → exploitation (confident about Q-values)
      - This is state-dependent: different inputs produce different noise levels

    Uses FACTORIZED noise (more efficient than independent noise):
      ε_w = f(ε_i) ⊗ f(ε_j)   where f(x) = sign(x)·√|x|
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable, but need to be on correct device)
        self.register_buffer("eps_weight", torch.zeros(out_features, in_features))
        self.register_buffer("eps_bias", torch.zeros(out_features))

        self._sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self._sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self._sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.eps_weight.copy_(eps_out.outer(eps_in))
        self.eps_bias.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            # At eval time: no noise, just use learned means
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)



class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 5,
        hidden_dims: tuple[int, int] = (256, 128),
    ):
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

    def reset_noise(self):
        """No-op for compatibility (standard net has no noise)."""
        pass



class NoisyDuelingQNetwork(nn.Module):
    """
    Dueling Q-Network with NoisyLinear layers in the value and advantage heads.
    Feature extraction uses standard Linear (noise in heads is sufficient).

    This replaces ε-greedy exploration entirely:
    - During training: noise is active → automatic exploration
    - During eval: noise disabled → pure exploitation
    - The network LEARNS when to explore and when to exploit
    """

    def __init__(
        self,
        input_dim: int = 72,
        output_dim: int = 5,
        hidden_dims: tuple[int, int] = (256, 128),
    ):
        super().__init__()
        hidden_dim1, hidden_dim2 = hidden_dims

        # Feature extraction: standard Linear (no noise needed here)
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )

        # Value stream: NoisyLinear for exploration
        self.value_hidden = NoisyLinear(hidden_dim2, hidden_dim2 // 2)
        self.value_out = NoisyLinear(hidden_dim2 // 2, 1)

        # Advantage stream: NoisyLinear for exploration
        self.adv_hidden = NoisyLinear(hidden_dim2, hidden_dim2 // 2)
        self.adv_out = NoisyLinear(hidden_dim2 // 2, output_dim)

    def forward(self, x):
        features = self.feature(x)

        value = F.relu(self.value_hidden(features))
        value = self.value_out(value)

        advantage = F.relu(self.adv_hidden(features))
        advantage = self.adv_out(advantage)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Resample noise for all NoisyLinear layers."""
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()


def createValueNetwork(input_dim, output_dim):
    """Helper function to instantiate the network."""
    return Net(input_dim, output_dim)