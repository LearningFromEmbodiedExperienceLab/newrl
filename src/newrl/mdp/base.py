"""Composable batched task terms evaluated on the simulator successor."""

from dataclasses import dataclass, field, replace

import torch
from tensordict import TensorDict

from .terms import EpisodeEndTerm, ObservationTerm, RewardTerm, StepFeatures


@dataclass
class MDP:
    observations: dict[str, ObservationTerm] = field(default_factory=dict)
    rewards: dict[str, RewardTerm] = field(default_factory=dict)
    terminations: dict[str, EpisodeEndTerm] = field(default_factory=dict)
    truncations: dict[str, EpisodeEndTerm] = field(default_factory=dict)

    def _initialize(self, scene):
        """Bind terms that supply a scene initialization hook."""

        def bind(terms):
            bound = {}
            for name, term in terms.items():
                initialize = getattr(term, "_initialize", None)
                bound[name] = term if initialize is None else initialize(scene)
            return bound

        return replace(
            self,
            observations=bind(self.observations),
            rewards=bind(self.rewards),
            terminations=bind(self.terminations),
            truncations=bind(self.truncations),
        )

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
        step_features = StepFeatures(physical_state)
        transition = self.step(physical_state, next_mdp_state, _features=step_features)
        self._after_step(next_mdp_state, transition)
        del next_mdp_state["action"]
        next_mdp_state["previous_action"] = action
        transition["observation"] = self.observe(
            physical_state, next_mdp_state, _features=step_features
        )
        return next_mdp_state, transition

    def _after_step(self, mdp_state: TensorDict, transition: TensorDict) -> None:
        """Update MDP state after scoring, before building the next observation."""

    def observe(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        *,
        _features: StepFeatures | None = None,
    ) -> TensorDict:
        return self._evaluate(
            self.observations, _features or StepFeatures(physical_state), mdp_state
        )

    def step(
        self,
        physical_state: TensorDict,
        mdp_state: TensorDict,
        *,
        _features: StepFeatures | None = None,
    ) -> TensorDict:
        step_features = _features or StepFeatures(physical_state)
        rewards = self._evaluate(self.rewards, step_features, mdp_state)
        column_shape = (*mdp_state.batch_size, 1)
        self._check_column_shape(rewards, column_shape)
        reward = sum(
            rewards.values(), mdp_state["previous_action"].new_zeros(column_shape)
        )
        terminated = self._any(
            self._evaluate(self.terminations, step_features, mdp_state), mdp_state
        )
        truncated = self._any(
            self._evaluate(self.truncations, step_features, mdp_state), mdp_state
        )
        return TensorDict(
            {
                "reward_terms": rewards,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated | truncated,
            },
            batch_size=mdp_state.batch_size,
            device=mdp_state.device,
        )

    @staticmethod
    def _evaluate(terms, step_features, mdp_state):
        return TensorDict(
            {
                name: term(
                    step_features.physical_state, mdp_state, step_features=step_features
                )
                for name, term in terms.items()
            },
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
