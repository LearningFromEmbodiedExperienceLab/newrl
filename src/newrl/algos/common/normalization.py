"""Running normalization for observations and rewards."""

from collections.abc import Iterator

from typing import Any

from contextlib import contextmanager

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize the last dimension with decayed mean and variance.

    ``decay < 1`` tracks a shifting distribution. ``decay == 1`` gives equal
    weight to every sample seen so far. ``forward`` records statistics in
    training mode unless frozen. Explicit ``update`` calls always record them.
    """

    def __init__(self, size: int, *, decay: float  =  0.999, eps: float  =  1e-5) -> None:
        super().__init__()
        if size < 1:
            raise ValueError("size must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.eps = eps
        self.frozen = False
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("sum_sq", torch.zeros(size))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.sum.shape[0]:
            raise ValueError(
                f"Expected a last dimension of {self.sum.shape[0]}, got {values.shape[-1]}"
            )
        if self.training and not self.frozen:
            self.update(values)
        return self.normalize(values)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1, self.sum.shape[0]).to(self.sum.dtype)
        if self.decay < 1.0:
            self.count.mul_(self.decay).add_(flat.shape[0])
            self.sum.mul_(self.decay).add_(flat.sum(0))
            self.sum_sq.mul_(self.decay).add_(flat.square().sum(0))
        else:
            self.count.add_(flat.shape[0])
            self.sum.add_(flat.sum(0))
            self.sum_sq.add_(flat.square().sum(0))

    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.count.clamp_min(1.0)
        mean = self.sum / count
        variance = (self.sum_sq / count - mean.square()).clamp_min(self.eps)
        return mean, variance.sqrt()

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean_std()
        return torch.where(self.count == 0, values, (values - mean) / std)

    @contextmanager
    def freeze(self) -> Iterator[None]:
        previous = self.frozen
        self.frozen = True
        try:
            yield
        finally:
            self.frozen = previous


class ReturnNormalization(nn.Module):
    """Scale rewards by the root-mean-square of discounted returns.

    The scale does not subtract the mean, so reward signs stay intact. The
    per-environment return carries across rollouts and resets where ``done``
    is true, after that step's reward has been included.
    """

    def __init__(self, gamma: float, *, eps: float  =  1e-8) -> None:
        super().__init__()
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = gamma
        self.eps = eps
        self.register_buffer("square_sum", torch.zeros(()))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("returns", torch.zeros(0, 1))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # The number of worlds is learned from the first rollout.
        saved = state_dict.get(prefix + "returns")
        if (
            saved is not None
            and saved.ndim == 2
            and saved.shape[1] == 1
            and self.returns.shape != saved.shape
        ):
            self.returns = self.returns.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        returns = self._advance(reward, done)
        self.square_sum.add_(returns.square().sum())
        self.count.add_(returns.numel())

    def scale(self) -> torch.Tensor:
        scale = (self.square_sum / self.count.clamp_min(2.0)).sqrt()
        return torch.where(self.count < 2.0, 1.0, scale.clamp_min(self.eps))

    def _advance(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if reward.ndim != 3 or reward.shape[-1] != 1:
            raise ValueError(f"Expected reward shape [N, T, 1], got {tuple(reward.shape)}")
        done = done.to(dtype=torch.bool, device=reward.device)
        if done.shape != reward.shape:
            raise ValueError("done must match the reward shape")
        num_envs = reward.shape[0]
        if self.returns.shape[0] != num_envs or self.returns.device != reward.device:
            self.returns = reward.new_zeros(num_envs, 1)
        carried = self.returns
        steps = []
        for step in range(reward.shape[1]):
            carried = self.gamma * carried + reward[:, step]
            steps.append(carried)
            carried = torch.where(done[:, step], torch.zeros_like(carried), carried)
        self.returns = carried
        return torch.stack(steps, dim=1)
