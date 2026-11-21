from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

import torch

from newrl.utils.math import quat_rotate_inverse, yaw_rotate
from newrl.utils.string import resolve_scene_names


@dataclass(frozen=True)
class EntityField:
    entity: str
    field: str
    _shape: torch.Size | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def physical_fields(self):
        return frozenset({(self.entity, self.field)})

    def _initialize(self, scene):
        entity = scene.entities[self.entity]
        shapes = {
            "joint_pos": entity.q.shape[1:],
            "joint_vel": entity.qd.shape[1:],
            "joint_torque": entity.qd.shape[1:],
            "body_pose_w": torch.Size((len(entity.body_names), 7)),
            "body_twist_w": torch.Size((len(entity.body_names), 6)),
        }
        bound = replace(self)
        object.__setattr__(bound, "_shape", shapes[self.field])
        return bound

    @property
    def shape(self) -> torch.Size:
        """Shape of one world's observation, excluding the batch dimension."""
        if self._shape is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return self._shape

    def __call__(self, physical_state, mdp_state):
        return physical_state[self.entity, self.field]


@dataclass(frozen=True)
class PreviousAction:
    physical_fields: ClassVar[frozenset] = frozenset()
    _size: int | None = field(default=None, init=False, repr=False, compare=False)

    def _initialize(self, scene):
        bound = replace(self)
        object.__setattr__(
            bound, "_size", sum(entity.q.shape[1] for entity in scene.entities.values())
        )
        return bound

    @property
    def shape(self) -> torch.Size:
        if self._size is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return torch.Size((self._size,))

    def __call__(self, physical_state, mdp_state):
        return mdp_state["previous_action"]


previous_action = PreviousAction()


@dataclass(frozen=True)
class _BodyObservation:
    """Resolve body selectors on construction of the environment.

    Patterns full-match a Newton body label or its final path component.
    The selection follows scene body order, with each body appearing once.
    """

    entity: str
    body_names: str | Sequence[str]
    _indices: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _body_indices: tuple[int, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _components: ClassVar[tuple[int, ...]] = (3,)

    @property
    def shape(self) -> torch.Size:
        if self._indices is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return torch.Size((self._indices.numel(), *self._components))

    def _initialize(self, scene):
        entity = scene.entities[self.entity]
        names = (
            (self.body_names,) if isinstance(self.body_names, str) else self.body_names
        )
        if not names:
            raise ValueError(f"Empty body selection for {self.entity!r}")
        try:
            selected = resolve_scene_names(names, entity.body_names)
        except ValueError as exc:
            raise ValueError(f"No bodies match {names!r} on {self.entity!r}") from exc

        bound = replace(self)
        body_indices = tuple(selected)
        indices = torch.tensor(
            body_indices, device=entity.bodies.device, dtype=torch.long
        )
        object.__setattr__(bound, "_body_indices", body_indices)
        object.__setattr__(bound, "_indices", indices)
        return bound

    def _read(self, physical_state, field_name: str) -> torch.Tensor:
        if self._indices is None:
            raise RuntimeError(
                "Initialize body observations against a scene before evaluation"
            )
        return physical_state[self.entity, field_name].index_select(1, self._indices)


@dataclass(frozen=True)
class _BodyVelocity(_BodyObservation):
    frame: Literal["world", "body", "yaw"] = "world"

    def __post_init__(self):
        if self.frame not in ("world", "body", "yaw"):
            raise ValueError(f"Unknown velocity frame {self.frame!r}")

    @property
    def physical_fields(self):
        fields = {(self.entity, "body_twist_w")}
        if self.frame != "world":
            fields.add((self.entity, "body_pose_w"))
        return frozenset(fields)

    def _velocity(self, physical_state, components: slice) -> torch.Tensor:
        vector = self._read(physical_state, "body_twist_w")[..., components]
        if self.frame == "world":
            return vector
        quat = self._read(physical_state, "body_pose_w")[..., 3:7]
        if self.frame == "body":
            return quat_rotate_inverse(quat, vector)
        x, y, z, w = quat.unbind(dim=-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        return yaw_rotate(-yaw, vector)


class BodyLinearVelocity(_BodyVelocity):
    def __call__(self, physical_state, mdp_state):
        return self._velocity(physical_state, slice(0, 3))


class BodyAngularVelocity(_BodyVelocity):
    def __call__(self, physical_state, mdp_state):
        return self._velocity(physical_state, slice(3, 6))


class ProjectedGravity(_BodyObservation):
    @property
    def physical_fields(self):
        return frozenset({(self.entity, "projected_gravity")})

    def __call__(self, physical_state, mdp_state):
        return self._read(physical_state, "projected_gravity")


class BodyHeight(_BodyObservation):
    _components = ()

    @property
    def physical_fields(self):
        return frozenset({(self.entity, "body_pose_w")})

    def __call__(self, physical_state, mdp_state):
        return self._read(physical_state, "body_pose_w")[..., 2]


@dataclass(frozen=True)
class VelocityCommandObservation:
    physical_fields: ClassVar[frozenset] = frozenset()

    @property
    def shape(self) -> torch.Size:
        return torch.Size((3,))

    def __call__(self, physical_state, mdp_state):
        return mdp_state["velocity_command"]


velocity_command = VelocityCommandObservation()
