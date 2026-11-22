"""Structural interfaces for terms evaluated by an MDP."""

from typing import Protocol

import torch
from tensordict import TensorDict


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
    ) -> torch.Tensor:
        """Return observations with shape `[num_envs, *shape]`."""
        ...


class RewardTerm(Protocol):
    @property
    def range(self) -> tuple[float, float]:
        """Lower and upper per-step bounds, including the term's weight."""
        ...

    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
    ) -> torch.Tensor:
        """Return rewards with shape `[num_envs, 1]`."""
        ...


class EpisodeEndTerm(Protocol):
    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
    ) -> torch.Tensor:
        """Return boolean episode-end flags with shape `[num_envs, 1]`."""
        ...
