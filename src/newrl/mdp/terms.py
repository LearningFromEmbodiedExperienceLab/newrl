"""Structural interfaces for terms evaluated by an MDP."""

from collections.abc import Callable, Hashable
from typing import Protocol

import torch
from tensordict import TensorDict


class StepFeatures:
    """Memoize values derived from one physical snapshot during MDP evaluation."""

    def __init__(self, physical_state: TensorDict):
        self.physical_state = physical_state
        self._values: dict[Hashable, torch.Tensor] = {}

    def get(
        self, key: Hashable, compute: Callable[[TensorDict], torch.Tensor]
    ) -> torch.Tensor:
        if key not in self._values:
            self._values[key] = compute(self.physical_state)
        return self._values[key]


class ObservationTerm(Protocol):
    @property
    def shape(self) -> torch.Size:
        """Shape of one world's observation, excluding the batch dimension."""
        ...

    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
        *,
        step_features: StepFeatures | None = None,
    ) -> torch.Tensor:
        """Return observations with shape `[num_envs, *shape]`."""
        ...


class RewardTerm(Protocol):
    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
        *,
        step_features: StepFeatures | None = None,
    ) -> torch.Tensor:
        """Return rewards with shape `[num_envs, 1]`."""
        ...


class EpisodeEndTerm(Protocol):
    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
        *,
        step_features: StepFeatures | None = None,
    ) -> torch.Tensor:
        """Return boolean episode-end flags with shape `[num_envs, 1]`."""
        ...
