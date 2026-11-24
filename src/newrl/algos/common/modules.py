"""MLP and diagonal Gaussian policy.

Linear layers use explicit input sizes. The first hidden linear and the actor
mean are marked ``_non_muon`` so a later Muon optimizer can leave them on AdamW.
"""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feedforward stack with optional layer normalization around the activation.

    ``num_units`` includes the input size and the width of the last hidden
    layer. A task head, such as the action mean, sits outside this module.
    """

    def __init__(
        self,
        num_units: tuple[int, ...] | list[int],
        activation: type[nn.Module] = nn.Mish,
        layer_norm: str | None = "pre",
        *,
        first_non_muon: bool = False,
    ) -> None:
        super().__init__()
        if len(num_units) < 2:
            raise ValueError("num_units must contain an input size and one layer")
        if layer_norm not in ("pre", "post", None):
            raise ValueError("layer_norm must be 'pre', 'post', or None")
        layers: list[nn.Module] = []
        for index in range(len(num_units) - 1):
            linear = nn.Linear(num_units[index], num_units[index + 1])
            if first_non_muon and index == 0:
                linear.weight._non_muon = True
            layers.append(linear)
            if layer_norm == "pre":
                layers.append(nn.LayerNorm(num_units[index + 1]))
            layers.append(activation())
            if layer_norm == "post":
                layers.append(nn.LayerNorm(num_units[index + 1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)

    def orth(self, gain: float = 1.0) -> "MLP":
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain)
                nn.init.zeros_(module.bias)
        return self


def _bounded_std_raw(min_std: float, max_std: float) -> float:
    """Initialize the unconstrained std near 1 while keeping a sigmoid gradient."""
    margin = 0.25 * (max_std - min_std)
    target = min(max(min(1.0, max_std), min_std + margin), max_std - margin)
    unit = (target - min_std) / (max_std - min_std)
    return math.log(unit / (1.0 - unit))


class NormalActor(nn.Module):
    """Diagonal Gaussian policy with a state-independent bounded std."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256, 256),
        activation: type[nn.Module] = nn.Mish,
        *,
        min_std: float = 1e-2,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if not 0.0 < min_std < max_std:
            raise ValueError("std bounds must satisfy 0 < min_std < max_std")
        self.min_std = min_std
        self.max_std = max_std
        self.backbone = MLP(
            (input_dim, *hidden),
            activation=activation,
            layer_norm="pre",
            first_non_muon=True,
        ).orth()
        self.mean = nn.Linear(hidden[-1], action_dim)
        self.mean.weight._non_muon = True
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.zeros_(self.mean.bias)
        self.std_raw = nn.Parameter(
            torch.full((action_dim,), _bounded_std_raw(min_std, max_std))
        )
        self.std_raw._non_muon = True

    def scale(self) -> torch.Tensor:
        unit = torch.sigmoid(self.std_raw)
        return self.min_std + (self.max_std - self.min_std) * unit

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        loc = self.mean(self.backbone(observation))
        # The scale is positive by construction. Tensor validation syncs on CUDA.
        return torch.distributions.Normal(loc, self.scale(), validate_args=False)

    def log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(observation).log_prob(action).sum(-1, keepdim=True)

    def entropy(self, observation: torch.Tensor) -> torch.Tensor:
        return self.distribution(observation).entropy().sum(-1, keepdim=True)


class Value(nn.Module):
    """Scalar state value. The output linear is excluded from Muon."""

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (512, 256, 256),
        activation: type[nn.Module] = nn.Mish,
    ) -> None:
        super().__init__()
        self.backbone = MLP(
            (input_dim, *hidden),
            activation=activation,
            layer_norm="pre",
            first_non_muon=True,
        ).orth()
        self.head = nn.Linear(hidden[-1], 1)
        self.head.weight._non_muon = True
        nn.init.orthogonal_(self.head.weight, 1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(observation))
