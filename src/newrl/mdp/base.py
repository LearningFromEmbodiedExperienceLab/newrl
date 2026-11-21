"""Composable batched task terms evaluated on the simulator successor."""

from dataclasses import dataclass, field, replace
from functools import cached_property

import torch
from tensordict import TensorDict

from .terms import EpisodeEndTerm, ObservationTerm, RewardTerm


def _bind(term, scene):
    initialize = getattr(term, "_initialize", None)
    return term if initialize is None else initialize(scene)


def _require_groups(groups: dict, kind: str, group_type: type) -> None:
    for name, group in groups.items():
        if not isinstance(group, group_type):
            raise TypeError(
                f"{kind} group {name!r} must be a {group_type.__name__}, got {type(group).__name__}"
            )


@dataclass(frozen=True)
class ObservationGroup:
    """Concatenate named observation terms in definition order.

    ``shape`` is the flattened per-world width. ``__call__`` returns
    ``[num_envs, *shape]``.
    """

    terms: dict[str, ObservationTerm]

    def __post_init__(self):
        if not self.terms:
            raise ValueError("An observation group must contain at least one term")
        object.__setattr__(self, "terms", dict(self.terms))

    def _initialize(self, scene) -> "ObservationGroup":
        return replace(
            self, terms={name: _bind(term, scene) for name, term in self.terms.items()}
        )

    @cached_property
    def shape(self) -> torch.Size:
        return torch.Size((sum(term.shape.numel() for term in self.terms.values()),))

    @cached_property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.terms.keys())

    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
    ) -> torch.Tensor:
        parts = []
        for name, term in self.terms.items():
            value = term(physical_state, mdp_state)
            flat = value.reshape(*mdp_state.batch_size, -1)
            if flat.shape[-1] != term.shape.numel():
                raise ValueError(
                    f"Observation term {name!r} has shape {tuple(value.shape)} "
                    f"but declares {tuple(term.shape)}"
                )
            parts.append(flat)
        return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class RewardGroup:
    """Named reward terms. The group total is their sum."""

    terms: dict[str, RewardTerm]

    def __post_init__(self):
        if not self.terms:
            raise ValueError("A reward group must contain at least one term")
        object.__setattr__(self, "terms", dict(self.terms))

    def _initialize(self, scene) -> "RewardGroup":
        return replace(
            self, terms={name: _bind(term, scene) for name, term in self.terms.items()}
        )

    @cached_property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.terms.keys())

    def contributions(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
    ) -> dict[str, torch.Tensor]:
        return {
            name: term(physical_state, mdp_state) for name, term in self.terms.items()
        }

    def __call__(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        /,
    ) -> torch.Tensor:
        values = list(self.contributions(physical_state, mdp_state).values())
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total


@dataclass
class MDP:
    """Task terms. Observations and rewards are groups; episode ends are flat.

    An observation group evaluates to one concatenated vector at
    ``observation[group]``. Reward terms stay named under
    ``reward_terms[group][term]``, and ``reward`` is the sum of every term.
    """

    observations: dict[str, ObservationGroup] = field(default_factory=dict)
    rewards: dict[str, RewardGroup] = field(default_factory=dict)
    terminations: dict[str, EpisodeEndTerm] = field(default_factory=dict)
    truncations: dict[str, EpisodeEndTerm] = field(default_factory=dict)

    def _initialize(self, scene):
        """Bind terms that supply a scene initialization hook."""
        _require_groups(self.observations, "observation", ObservationGroup)
        _require_groups(self.rewards, "reward", RewardGroup)
        return replace(
            self,
            observations={
                name: _bind(group, scene) for name, group in self.observations.items()
            },
            rewards={name: _bind(group, scene) for name, group in self.rewards.items()},
            terminations={
                name: _bind(term, scene) for name, term in self.terminations.items()
            },
            truncations={
                name: _bind(term, scene) for name, term in self.truncations.items()
            },
        )

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]] | None:
        """Fields required by bound terms. None means an undeclared term needs a full snapshot."""
        terms = [
            term
            for group in (*self.observations.values(), *self.rewards.values())
            for term in group.terms.values()
        ]
        terms.extend(self.terminations.values())
        terms.extend(self.truncations.values())
        fields = set()
        for term in terms:
            required = getattr(term, "physical_fields", None)
            if required is None:
                return None
            fields.update(required)
        return frozenset(fields)

    def initial_state(
        self, num_envs: int, action_dim: int, device: torch.device
    ) -> TensorDict:
        return TensorDict(
            {
                "episode_step": torch.zeros(num_envs, dtype=torch.long, device=device),
                "previous_action": torch.zeros((num_envs, action_dim), device=device),
            },
            batch_size=[num_envs],
            device=device,
        )

    def reset(
        self, mdp_state: TensorDict, ids: torch.Tensor | slice
    ) -> tuple[TensorDict, dict[str, TensorDict]]:
        """Return reset MDP state and optional per-entity poses for selected worlds."""
        next_mdp_state = mdp_state.clone()
        next_mdp_state["episode_step"][ids] = 0
        next_mdp_state["previous_action"][ids] = 0
        return next_mdp_state, {}

    def advance(
        self, physical_state: TensorDict, mdp_state: TensorDict, action: torch.Tensor
    ) -> tuple[TensorDict, TensorDict]:
        next_mdp_state = mdp_state.clone()
        next_mdp_state["episode_step"] += 1
        next_mdp_state["action"] = action
        transition = self.step(physical_state, next_mdp_state)
        self._after_step(next_mdp_state, transition)
        del next_mdp_state["action"]
        next_mdp_state["previous_action"] = action
        transition["observation"] = self.observe(physical_state, next_mdp_state)
        return next_mdp_state, transition

    def _after_step(self, mdp_state: TensorDict, transition: TensorDict) -> None:
        """Update MDP state after scoring, before building the next observation."""

    def observe(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
    ) -> TensorDict:
        return TensorDict(
            {
                name: group(physical_state, mdp_state)
                for name, group in self.observations.items()
            },
            batch_size=mdp_state.batch_size,
            device=mdp_state.device,
        )

    def step(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
    ) -> TensorDict:
        column_shape = (*mdp_state.batch_size, 1)
        reward = mdp_state["previous_action"].new_zeros(column_shape)
        reward_terms = {}
        for name, group in self.rewards.items():
            values = group.contributions(physical_state, mdp_state)
            self._check_column_shape(values, column_shape)
            for value in values.values():
                reward = reward + value
            reward_terms[name] = TensorDict(
                values, batch_size=mdp_state.batch_size, device=mdp_state.device
            )
        terminated = self._any(
            self._evaluate(self.terminations, physical_state, mdp_state), mdp_state
        )
        truncated = self._any(
            self._evaluate(self.truncations, physical_state, mdp_state), mdp_state
        )
        return TensorDict(
            {
                "reward_terms": TensorDict(
                    reward_terms,
                    batch_size=mdp_state.batch_size,
                    device=mdp_state.device,
                ),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated | truncated,
            },
            batch_size=mdp_state.batch_size,
            device=mdp_state.device,
        )

    @staticmethod
    def _evaluate(terms, physical_state, mdp_state):
        return TensorDict(
            {name: term(physical_state, mdp_state) for name, term in terms.items()},
            batch_size=mdp_state.batch_size,
            device=mdp_state.device,
        )

    @staticmethod
    def _check_column_shape(terms, shape):
        for name, value in terms.items():
            if value.shape != shape:
                raise ValueError(
                    f"Expected MDP term {name!r} to have shape {shape}, got {tuple(value.shape)}"
                )

    @staticmethod
    def _any(terms, mdp_state):
        column_shape = (*mdp_state.batch_size, 1)
        MDP._check_column_shape(terms, column_shape)
        flags = torch.zeros(column_shape, device=mdp_state.device, dtype=torch.bool)
        for value in terms.values():
            flags |= value
        return flags
