"""Pure term evaluations: inputs must never be mutated or RNG advanced.

Terms may declare ``state_history`` as a mapping from task-state keys to
observation readers. The MDP binds those readers, initializes/resets history,
and records successor values after scoring, preserving imagination gradients.
"""

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
