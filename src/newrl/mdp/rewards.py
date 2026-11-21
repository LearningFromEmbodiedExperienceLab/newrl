from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import torch

from newrl.utils.string import resolve_scene_names

from .observations import BodyAngularVelocity, BodyLinearVelocity


@dataclass(frozen=True)
class ActionRate:
    """Penalize squared changes in normalized action per environment."""

    weight: float
    physical_fields: ClassVar[frozenset] = frozenset()

    def __call__(self, physical_state, mdp_state):
        delta = mdp_state["action"] - mdp_state["previous_action"]
        return -self.weight * delta.square().sum(dim=-1, keepdim=True)


def _resolve_joint_indices(entity, joint_names: str | Sequence[str]) -> torch.Tensor:
    """Full-match joint labels, or their final path component, in scene order."""
    try:
        selected = resolve_scene_names(joint_names, entity.joint_names)
    except ValueError as exc:
        raise ValueError(f"No joints match {joint_names!r}") from exc
    return torch.tensor(selected, device=entity.qd.device, dtype=torch.long)


@dataclass(frozen=True)
class _JointTerm:
    weight: float
    entity: str = "robot"
    joint_names: str | Sequence[str] = ".*"
    _indices: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def physical_fields(self):
        return frozenset({(self.entity, "joint_vel")})

    def _initialize(self, scene):
        entity = scene.entities[self.entity]
        bound = replace(self)
        object.__setattr__(
            bound, "_indices", _resolve_joint_indices(entity, self.joint_names)
        )
        return bound

    def _select(self, values: torch.Tensor) -> torch.Tensor:
        if self._indices is None:
            raise RuntimeError("Bind the joint term to a scene before evaluating it")
        return values.index_select(-1, self._indices)

    def _velocity(self, physical_state) -> torch.Tensor:
        return self._select(physical_state[self.entity, "joint_vel"])


@dataclass(frozen=True)
class JointEnergy1(_JointTerm):
    """Penalize actuator power at the last physics substep.

    Torque comes from MuJoCo's last force evaluation; joint velocity is the
    successor velocity at the end of the control step.
    """

    @property
    def physical_fields(self):
        return super().physical_fields | {(self.entity, "joint_torque")}

    def __call__(self, physical_state, mdp_state):
        torque = self._select(physical_state[self.entity, "joint_torque"])
        velocity = self._velocity(physical_state)
        return -self.weight * (torque * velocity).abs().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class JointAcc2(_JointTerm):
    """Penalize squared average joint acceleration over one control interval.

    The first step of an episode contributes nothing. The velocity used here is
    carried in ``mdp_state`` so a replayed checkpoint reproduces the same penalty.
    """

    _control_dt: float | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _initialize(self, scene):
        if scene.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        bound = super()._initialize(scene)
        object.__setattr__(bound, "_control_dt", float(scene.control_dt))
        return bound

    def __call__(self, physical_state, mdp_state):
        if self._control_dt is None:
            raise RuntimeError("Bind JointAcc2 to a scene before evaluating it")
        velocity = physical_state[self.entity, "joint_vel"]
        previous = mdp_state.get("previous_joint_vel", velocity)
        selected = self._select(velocity)
        first_step = mdp_state["episode_step"] == 1
        delta = torch.where(
            first_step, torch.zeros_like(selected), selected - self._select(previous)
        )
        # Keep every joint so later selections read the same history.
        mdp_state["previous_joint_vel"] = velocity.detach().clone()
        acceleration = delta / self._control_dt
        return -self.weight * acceleration.square().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class JointVel2(_JointTerm):
    """Penalize the squared joint velocity."""

    def __call__(self, physical_state, mdp_state):
        velocity = self._velocity(physical_state)
        return -self.weight * velocity.square().sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class TrackVelocity:
    """Exponentially reward matching a command component in the yaw frame."""

    velocity: BodyLinearVelocity | BodyAngularVelocity
    components: slice
    weight: float
    sigma: float

    @property
    def physical_fields(self):
        return self.velocity.physical_fields

    def _initialize(self, scene):
        return replace(self, velocity=self.velocity._initialize(scene))

    def __call__(self, physical_state, mdp_state):
        measured = self.velocity(physical_state, mdp_state)[:, 0, self.components]
        target = mdp_state["velocity_command"][:, self.components]
        squared_error = (measured - target).square().sum(dim=-1, keepdim=True)
        return self.weight * (-squared_error / self.sigma**2).exp()


@dataclass(frozen=True)
class FootSlip:
    """Penalize horizontal foot body velocity while the foot carries load."""

    sensor: str
    weight: float
    _velocity: BodyLinearVelocity | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def physical_fields(self):
        fields = frozenset({(self.sensor, "loaded")})
        return (
            fields
            if self._velocity is None
            else fields | self._velocity.physical_fields
        )

    def _initialize(self, scene):
        config = scene.contact_sensors[self.sensor].config
        velocity = BodyLinearVelocity(config.entity, config.body_names)._initialize(
            scene
        )
        bound = replace(self)
        object.__setattr__(bound, "_velocity", velocity)
        return bound

    def __call__(self, physical_state, mdp_state):
        velocity = self._velocity(physical_state, mdp_state)
        loaded = physical_state[self.sensor, "loaded"]
        speed_sq = velocity[..., :2].square().sum(-1)
        return -self.weight * (speed_sq * loaded).sum(-1, keepdim=True)
