"""Composable batched task terms evaluated on the simulator successor."""

from dataclasses import dataclass, field, replace
from functools import cached_property

import torch
from tensordict import TensorDict

from newrl.utils.symmetry import SymmetryTransform

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


def _check_column_shape(terms, shape):
    for name, value in terms.items():
        if value.shape != shape:
            raise ValueError(
                f"Expected MDP term {name!r} to have shape {shape}, got {tuple(value.shape)}"
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

    def symmetry_transform(self, scene) -> SymmetryTransform:
        transforms = []
        for name, term in self.terms.items():
            mirror = getattr(term, "symmetry_transform", None)
            if mirror is None:
                raise ValueError(f"Observation term {name!r} does not declare a mirror")
            try:
                transform = mirror(scene)
                if transform.indices.numel() != term.shape.numel():
                    raise ValueError(
                        "Mirror width does not match the term's flattened shape"
                    )
            except ValueError as exc:
                raise ValueError(
                    f"Cannot reflect observation term {name!r}: {exc}"
                ) from exc
            transforms.append(transform)
        return SymmetryTransform.cat(transforms)

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
    """Concatenate named scalar reward terms in definition order."""

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
        values = self.contributions(physical_state, mdp_state)
        _check_column_shape(values, (*mdp_state.batch_size, 1))
        return torch.cat(tuple(values.values()), dim=-1)


@dataclass
class MDP:
    """Task terms. Observations and rewards are groups; episode ends are flat.

    An observation group evaluates to one concatenated vector at
    ``observation[group]``. Each reward group concatenates its terms at
    ``reward[group]``. Its columns follow the group's ``keys`` order.
    Named episode-end flags appear in ``termination_terms`` and
    ``truncation_terms`` alongside their aggregate flags.
    """

    observations: dict[str, ObservationGroup] = field(default_factory=dict)
    rewards: dict[str, RewardGroup] = field(default_factory=dict)
    terminations: dict[str, EpisodeEndTerm] = field(default_factory=dict)
    truncations: dict[str, EpisodeEndTerm] = field(default_factory=dict)
    _history: dict[str, ObservationTerm] = field(default_factory=dict, init=False, repr=False)

    def _initialize(self, scene):
        """Bind terms that supply a scene initialization hook."""
        _require_groups(self.observations, "observation", ObservationGroup)
        _require_groups(self.rewards, "reward", RewardGroup)
        bound = replace(
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
        terms = [term for group in (*bound.observations.values(), *bound.rewards.values()) for term in group.terms.values()]
        terms.extend((*bound.terminations.values(), *bound.truncations.values()))
        for term in terms:
            for name, reader in getattr(term, "state_history", {}).items():
                if name in ("episode_step", "previous_action", "action"):
                    raise ValueError(f"Reserved MDP history name {name!r}")
                reader = _bind(reader, scene)
                if name in bound._history and bound._history[name] != reader:
                    raise ValueError(f"Conflicting MDP history sources for {name!r}")
                bound._history[name] = reader
        return bound

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
        terms.extend(self._history.values())
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
                "episode_step": torch.zeros(
                    (num_envs, 1), dtype=torch.long, device=device
                ),
                "previous_action": torch.zeros((num_envs, action_dim), device=device),
                **{name: torch.zeros((num_envs, *reader.shape), device=device) for name, reader in self._history.items()},
            },
            batch_size=[num_envs],
            device=device,
        )

    @property
    def replay_fields(self) -> tuple[str, ...]:
        """Compact task memory needed to reconstruct replay decision states."""
        # Task preparation runs before scene binding, when reader shapes are
        # unavailable but every history key must already be discoverable.
        history = dict.fromkeys(self._history)
        terms = [term for group in (*self.observations.values(), *self.rewards.values()) for term in group.terms.values()]
        terms.extend((*self.terminations.values(), *self.truncations.values()))
        for term in terms:
            history.update(dict.fromkeys(getattr(term, "state_history", {})))
        return ("episode_step", "previous_action", *history)

    def restore_replay_state(self, state: TensorDict) -> TensorDict:
        """Rebuild task state without advancing time or sampling randomness."""
        return state.clone()

    def reset(
        self, mdp_state: TensorDict, ids: torch.Tensor | slice
    ) -> tuple[TensorDict, dict[str, TensorDict]]:
        """Return reset MDP state and optional per-entity poses for selected worlds."""
        next_mdp_state = mdp_state.clone()
        next_mdp_state["episode_step"][ids] = 0
        next_mdp_state["previous_action"][ids] = 0
        for name in self._history:
            next_mdp_state[name][ids] = 0
        return next_mdp_state, {}

    def step(
        self, physical_state: TensorDict, mdp_state: TensorDict, action: torch.Tensor
    ) -> tuple[TensorDict, TensorDict]:
        """Advance task memory using successor physics, without resetting.

        Rewards see decision-time goals, counters and history, plus the executed
        action. Endings and observations see successor task state. Inputs are
        never mutated; tensor operations retain gradients for imagination.
        """
        state = mdp_state.clone()
        state["action"] = action
        reward = self.compute_reward(physical_state, state)
        del state["action"]
        state["episode_step"] += 1
        state["previous_action"] = action
        for name, reader in self._history.items():
            state[name] = reader(physical_state, mdp_state).clone()
        extra = self._advance_task_state(physical_state, state)
        termination_terms = self.compute_termination(physical_state, state)
        truncation_terms = self.compute_truncation(physical_state, state)
        terminated = self._any(termination_terms, state)
        truncated = self._any(truncation_terms, state)
        return state, TensorDict(
            {
                **extra,
                "reward": reward,
                "termination_terms": termination_terms,
                "truncation_terms": truncation_terms,
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated | truncated,
                "observation": self.compute_observation(physical_state, state),
            },
            batch_size=state.batch_size, device=state.device,
        )

    def _advance_task_state(self, physical_state: TensorDict, state: TensorDict) -> dict:
        """Update owned successor memory after scoring; return extra transition fields."""
        return {}

    def compute_observation(self, physical_state: TensorDict, mdp_state: TensorDict) -> TensorDict:
        """Evaluate observation groups without changing either input."""
        return TensorDict(
            {name: group(physical_state, mdp_state) for name, group in self.observations.items()},
            batch_size=mdp_state.batch_size, device=mdp_state.device,
        )

    def compute_reward(self, physical_state: TensorDict, mdp_state: TensorDict) -> TensorDict:
        """Evaluate reward groups; action-dependent terms require mdp_state.action."""
        return TensorDict(
            {name: group(physical_state, mdp_state) for name, group in self.rewards.items()},
            batch_size=mdp_state.batch_size, device=mdp_state.device,
        )

    def compute_termination(self, physical_state: TensorDict, mdp_state: TensorDict) -> TensorDict:
        """Evaluate named boolean termination terms without advancing state."""
        return self._evaluate(self.terminations, physical_state, mdp_state)

    def compute_truncation(self, physical_state: TensorDict, mdp_state: TensorDict) -> TensorDict:
        """Evaluate named boolean truncation terms without advancing state."""
        return self._evaluate(self.truncations, physical_state, mdp_state)

    def compute_survival(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        """Differentiable imagined continuation where terms support it; excludes truncation."""
        survival = torch.ones((*mdp_state.batch_size, 1), device=mdp_state.device)
        for term in self.terminations.values():
            compute = getattr(term, 'survival', None)
            value = (~term(physical_state, mdp_state)).float() if compute is None else compute(physical_state, mdp_state)
            _check_column_shape({'survival': value}, survival.shape)
            survival = survival * value
        return survival

    @staticmethod
    def _evaluate(terms, physical_state, mdp_state):
        values = TensorDict(
            {name: term(physical_state, mdp_state) for name, term in terms.items()},
            batch_size=mdp_state.batch_size,
            device=mdp_state.device,
        )
        _check_column_shape(values, (*mdp_state.batch_size, 1))
        if any(value.dtype != torch.bool for value in values.values()):
            raise TypeError("Episode-end terms must return boolean tensors")
        return values

    @staticmethod
    def _any(terms, mdp_state):
        column_shape = (*mdp_state.batch_size, 1)
        _check_column_shape(terms, column_shape)
        flags = torch.zeros(column_shape, device=mdp_state.device, dtype=torch.bool)
        for value in terms.values():
            flags |= value
        return flags
