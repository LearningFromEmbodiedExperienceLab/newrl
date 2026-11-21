import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch

from .observations import BodyAngularVelocity, BodyLinearVelocity
from .terms import StepFeatures


@dataclass(frozen=True)
class ActionRate:
    """Penalize squared changes in normalized action per environment."""

    weight: float

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        delta = mdp_state["action"] - mdp_state["previous_action"]
        return -self.weight * delta.square().sum(dim=-1, keepdim=True)


def _resolve_joint_indices(entity, joint_names: str | Sequence[str]) -> torch.Tensor:
    """Full-match joint labels, or their final path component, in scene order."""
    names = (joint_names,) if isinstance(joint_names, str) else tuple(joint_names)
    if not names:
        raise ValueError("Joint selection must contain at least one pattern")
    selected: set[int] = set()
    for name in names:
        pattern = re.compile(name)
        matches = [
            index
            for index, label in enumerate(entity.joint_names)
            if pattern.fullmatch(label) or pattern.fullmatch(label.rsplit("/", 1)[-1])
        ]
        if not matches:
            raise ValueError(f"No joints match {name!r}")
        selected.update(matches)
    return torch.tensor(sorted(selected), device=entity.qd.device, dtype=torch.long)


@dataclass(frozen=True)
class _JointTerm:
    weight: float
    entity: str = "robot"
    joint_names: str | Sequence[str] = ".*"
    _indices: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def _initialize(self, scene):
        entity = scene.entities[self.entity]
        bound = replace(self)
        object.__setattr__(bound, "_indices", _resolve_joint_indices(entity, self.joint_names))
        return bound

    def _select(self, values: torch.Tensor) -> torch.Tensor:
        if self._indices is None:
            raise RuntimeError("Bind the joint term to a scene before evaluating it")
        return values.index_select(-1, self._indices)

    def _velocity(self, physical_state) -> torch.Tensor:
        return self._select(physical_state[self.entity, "joint_vel"])


@dataclass(frozen=True)
class JointEnergy1(_JointTerm):
    """Penalize the L1 mechanical power of the position actuators.

    Torque is the implicit PD law ``kp (q_target - q) - kd qdot``, with the
    target taken from the action that was just applied.
    """

    _stiffness: torch.Tensor | None = field(default=None, init=False, repr=False)
    _damping: torch.Tensor | None = field(default=None, init=False, repr=False)
    _default: torch.Tensor | None = field(default=None, init=False, repr=False)
    _scale: float | None = field(default=None, init=False, repr=False)

    def _initialize(self, scene):
        import warp as wp

        bound = super()._initialize(scene)
        entity = scene.entities[self.entity]
        object.__setattr__(
            bound,
            "_stiffness",
            bound._select(wp.to_torch(scene.model.joint_target_ke)[entity.qd].detach().clone()),
        )
        object.__setattr__(
            bound,
            "_damping",
            bound._select(wp.to_torch(scene.model.joint_target_kd)[entity.qd].detach().clone()),
        )
        object.__setattr__(
            bound, "_default", bound._select(entity.default_position.detach().clone())
        )
        object.__setattr__(bound, "_scale", entity.action.scale)
        return bound

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        if self._stiffness is None or self._damping is None or self._default is None:
            raise RuntimeError("Bind JointEnergy1 to a scene before evaluating it")
        position = self._select(physical_state[self.entity, "joint_pos"])
        velocity = self._velocity(physical_state)
        target = self._default + self._scale * self._select(mdp_state["action"])
        torque = self._stiffness * (target - position) - self._damping * velocity
        power = (torque * velocity).abs().sum(dim=-1, keepdim=True)
        return -self.weight * power


@dataclass(frozen=True)
class JointAcc2(_JointTerm):
    """Penalize the squared change in joint velocity since the previous step.

    The first step of an episode contributes nothing. The velocity used here is
    carried in ``mdp_state`` so a replayed checkpoint reproduces the same penalty.
    """

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        velocity = physical_state[self.entity, "joint_vel"]
        previous = mdp_state.get("previous_joint_vel", velocity)
        selected = self._select(velocity)
        first_step = mdp_state["episode_step"].unsqueeze(-1) == 1
        delta = torch.where(
            first_step, torch.zeros_like(selected), selected - self._select(previous)
        )
        # Keep every joint so later selections read the same history.
        mdp_state["previous_joint_vel"] = velocity.detach().clone()
        return -self.weight * delta.square().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class JointVel2(_JointTerm):
    """Penalize the squared joint velocity."""

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        velocity = self._velocity(physical_state)
        return -self.weight * velocity.square().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class TrackVelocity:
    """Exponentially reward matching a command component in the yaw frame."""

    velocity: BodyLinearVelocity | BodyAngularVelocity
    components: slice
    weight: float
    sigma: float

    def _initialize(self, scene):
        return replace(self, velocity=self.velocity._initialize(scene))

    def __call__(
        self, physical_state, mdp_state, *, step_features: StepFeatures | None = None
    ):
        measured = self.velocity(
            physical_state, mdp_state, step_features=step_features
        )[:, 0, self.components]
        target = mdp_state["velocity_command"][:, self.components]
        squared_error = (measured - target).square().sum(dim=-1, keepdim=True)
        return self.weight * (-squared_error / self.sigma**2).exp()
