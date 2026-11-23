from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import inf, isfinite
from typing import ClassVar

import torch

from newrl.utils.string import resolve_scene_names

from .observations import BodyAngularVelocity, BodyLinearVelocity


def _weighted_range(weight: float, maximum: float = inf) -> tuple[float, float]:
    """Bounds for a weighted quantity between zero and maximum."""
    if weight == 0:
        return 0.0, 0.0
    endpoint = weight * maximum
    return min(0.0, endpoint), max(0.0, endpoint)


@dataclass(frozen=True)
class ActionRate:
    """Penalize squared changes in normalized action per environment."""

    weight: float
    physical_fields: ClassVar[frozenset] = frozenset()

    @property
    def range(self) -> tuple[float, float]:
        return _weighted_range(-self.weight)

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
    def range(self) -> tuple[float, float]:
        return _weighted_range(-self.weight)

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
    def range(self) -> tuple[float, float]:
        return _weighted_range(self.weight, 1.0)

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
    def range(self) -> tuple[float, float]:
        return _weighted_range(-self.weight)

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


@dataclass(frozen=True)
class FeetAirTime:
    """Reward the number of feet with current air time strictly inside a range.

    This follows mjlab's velocity-task feet_air_time without command gating.
    """

    sensor: str
    weight: float
    threshold_min: float = 0.05
    threshold_max: float = 0.5
    _num_feet: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not 0.0 <= self.threshold_min < self.threshold_max:
            raise ValueError("Air-time thresholds must satisfy 0 <= min < max")

    @property
    def range(self) -> tuple[float, float]:
        """Use a conservative bound until the scene supplies the foot count."""
        maximum = inf if self._num_feet is None else self._num_feet
        return _weighted_range(self.weight, maximum)

    @property
    def physical_fields(self):
        return frozenset({(self.sensor, "current_air_time")})

    def _initialize(self, scene):
        scene.resolve_physical_fields(self.physical_fields)
        sensor = scene.contact_sensors[self.sensor]
        bound = replace(self)
        object.__setattr__(bound, "_num_feet", sensor.body_ids.shape[-1])
        return bound

    def __call__(self, physical_state, mdp_state):
        air = physical_state[self.sensor, "current_air_time"]
        in_range = (air > self.threshold_min) & (air < self.threshold_max)
        return self.weight * in_range.sum(dim=-1, keepdim=True).float()


@dataclass(frozen=True)
class BodyAngularVelocity2:
    """Penalize squared angular velocity after scaling each axis in the chosen frame."""

    entity: str
    body_names: str | Sequence[str]
    weight: float
    axis: tuple[float, float, float] = (1.0, 1.0, 1.0)
    frame: str = "world"
    _velocity: BodyAngularVelocity = field(
        init=False, repr=False, compare=False
    )
    _axis: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if len(self.axis) != 3 or not all(isfinite(scale) for scale in self.axis):
            raise ValueError("axis must contain three finite component scales")
        velocity = BodyAngularVelocity(self.entity, self.body_names, frame=self.frame)
        object.__setattr__(self, "_velocity", velocity)

    @property
    def range(self) -> tuple[float, float]:
        return _weighted_range(-self.weight if any(self.axis) else 0.0)

    @property
    def physical_fields(self):
        return self._velocity.physical_fields

    def _initialize(self, scene):
        bound = replace(self)
        velocity = self._velocity._initialize(scene)
        device = scene.entities[self.entity].bodies.device
        object.__setattr__(bound, "_velocity", velocity)
        object.__setattr__(bound, "_axis", torch.tensor(self.axis, device=device))
        return bound

    def __call__(self, physical_state, mdp_state):
        velocity = self._velocity(physical_state, mdp_state)
        squared_speed = (velocity * self._axis).square().sum(dim=(-2, -1))
        return -self.weight * squared_speed.unsqueeze(-1)


@dataclass(frozen=True)
class BodyLinearVelocity2:
    """Penalize squared linear velocity after scaling each axis in the chosen frame."""

    entity: str
    body_names: str | Sequence[str]
    weight: float
    axis: tuple[float, float, float] = (1.0, 1.0, 1.0)
    frame: str = "world"
    _velocity: BodyLinearVelocity = field(init=False, repr=False, compare=False)
    _axis: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if len(self.axis) != 3 or not all(isfinite(scale) for scale in self.axis):
            raise ValueError("axis must contain three finite component scales")
        velocity = BodyLinearVelocity(self.entity, self.body_names, frame=self.frame)
        object.__setattr__(self, "_velocity", velocity)

    @property
    def range(self) -> tuple[float, float]:
        return _weighted_range(-self.weight if any(self.axis) else 0.0)

    @property
    def physical_fields(self):
        return self._velocity.physical_fields

    def _initialize(self, scene):
        bound = replace(self)
        velocity = self._velocity._initialize(scene)
        device = scene.entities[self.entity].bodies.device
        object.__setattr__(bound, "_velocity", velocity)
        object.__setattr__(bound, "_axis", torch.tensor(self.axis, device=device))
        return bound

    def __call__(self, physical_state, mdp_state):
        velocity = self._velocity(physical_state, mdp_state)
        squared_speed = (velocity * self._axis).square().sum(dim=(-2, -1))
        return -self.weight * squared_speed.unsqueeze(-1)
