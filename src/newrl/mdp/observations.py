from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

import torch

from newrl.utils.math import quat_rotate_inverse, yaw_rotate
from newrl.utils.string import resolve_scene_names
from newrl.utils.symmetry import SymmetryTransform


_POLAR_SIGNS = (1, -1, 1)
_AXIAL_SIGNS = (-1, 1, -1)


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

    def symmetry_transform(self, scene) -> SymmetryTransform:
        if self.field not in ("joint_pos", "joint_vel", "joint_torque"):
            raise ValueError(f"No reflection defined for entity field {self.field!r}")
        return scene.entities[self.entity].joint_symmetry_transform()


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

    def symmetry_transform(self, scene) -> SymmetryTransform:
        return scene.action_symmetry_transform()


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
    _mirror_signs: ClassVar[tuple[int, ...] | None] = None

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
        return physical_state[self.entity, field_name].index_select(
            physical_state.ndim, self._indices
        )

    def symmetry_transform(self, scene) -> SymmetryTransform:
        if self._mirror_signs is None:
            raise ValueError(f"No reflection defined for {type(self).__name__}")
        return scene.entities[self.entity].body_symmetry_transform(
            self._body_indices, self._mirror_signs
        )


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
    _mirror_signs = _POLAR_SIGNS

    def __call__(self, physical_state, mdp_state):
        return self._velocity(physical_state, slice(0, 3))


class BodyAngularVelocity(_BodyVelocity):
    _mirror_signs = _AXIAL_SIGNS

    def __call__(self, physical_state, mdp_state):
        return self._velocity(physical_state, slice(3, 6))


class ProjectedGravity(_BodyObservation):
    _mirror_signs = _POLAR_SIGNS

    @property
    def physical_fields(self):
        return frozenset({(self.entity, "projected_gravity")})

    def __call__(self, physical_state, mdp_state):
        return self._read(physical_state, "projected_gravity")


class BodyHeight(_BodyObservation):
    _components = ()
    _mirror_signs = (1,)

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

    def symmetry_transform(self, scene) -> SymmetryTransform:
        return SymmetryTransform((0, 1, 2), (1, -1, -1))


velocity_command = VelocityCommandObservation()


@dataclass(frozen=True)
class ContactObservation:
    """Read a configured contact sensor field as floating-point features."""

    sensor: str
    _shape: torch.Size | None = field(
        default=None, init=False, repr=False, compare=False
    )
    field: Literal[
        "loaded",
        "force_w",
        "first_contact",
        "current_air_time",
        "last_air_time",
        "last_contact_pos_w",
        "last_contact_pos_valid",
    ] = "loaded"

    @property
    def physical_fields(self):
        return frozenset({(self.sensor, self.field)})

    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            raise RuntimeError(
                "Bind the contact observation to a scene before reading its shape"
            )
        return self._shape

    def _initialize(self, scene):
        scene.resolve_physical_fields(self.physical_fields)
        count = scene.sensors[self.sensor].body_ids.shape[1]
        bound = replace(self)
        object.__setattr__(
            bound,
            "_shape",
            torch.Size(
                (count, 3)
                if self.field in ("force_w", "last_contact_pos_w")
                else (count,)
            ),
        )
        return bound

    def __call__(self, physical_state, mdp_state):
        return physical_state[self.sensor, self.field].float()

    def symmetry_transform(self, scene) -> SymmetryTransform:
        sensor = scene.sensors[self.sensor]
        entity = scene.entities[sensor.config.entity]
        indices = resolve_scene_names(sensor.config.body_names, entity.body_names)
        signs = _POLAR_SIGNS if self.field in ("force_w", "last_contact_pos_w") else (1,)
        return entity.body_symmetry_transform(indices, signs)


@dataclass(frozen=True)
class _GridObservation:
    sensor: str
    include_valid: bool = True
    _grid_shape: torch.Size | None = field(
        default=None, init=False, repr=False, compare=False
    )
    value_field: ClassVar[str]

    @property
    def physical_fields(self):
        fields = {(self.sensor, self.value_field)}
        if self.include_valid:
            fields.add((self.sensor, "valid"))
        return frozenset(fields)

    @property
    def shape(self) -> torch.Size:
        if self._grid_shape is None:
            raise RuntimeError("Bind the grid observation to a scene before reading its shape")
        return torch.Size((2 if self.include_valid else 1, *self._grid_shape))

    def _initialize(self, scene):
        scene.resolve_physical_fields(self.physical_fields)
        bound = replace(self)
        object.__setattr__(
            bound, "_grid_shape", scene.sensors[self.sensor].shape
        )
        return bound

    def __call__(self, physical_state, mdp_state):
        values = physical_state[self.sensor, self.value_field]
        if self.include_valid:
            valid = physical_state[self.sensor, "valid"].to(values.dtype)
            return torch.stack((values, valid), dim=-3)
        return values.unsqueeze(-3)

    def symmetry_transform(self, scene) -> SymmetryTransform:
        sensor = scene.sensors[self.sensor]
        config = sensor.config
        entity = scene.entities[config.entity]
        indices = resolve_scene_names(config.body, entity.body_names)
        entity.body_symmetry_transform(indices, (1,))
        permutation = torch.arange(self.shape.numel()).reshape(self.shape).flip(-1)
        return SymmetryTransform(permutation.flatten().tolist(), (1,) * permutation.numel())


@dataclass(frozen=True)
class DepthCameraObservation(_GridObservation):
    """Camera depth and optional validity channel in CHW order."""

    value_field: ClassVar[str] = "depth"

    def symmetry_transform(self, scene) -> SymmetryTransform:
        config = scene.sensors[self.sensor].config
        roll, _, yaw = config.rpy
        if any(abs(value) > 1e-7 for value in (config.pos[1], roll, yaw)):
            raise ValueError("Depth mirroring requires a centered camera with zero roll and yaw")
        return super().symmetry_transform(scene)


@dataclass(frozen=True)
class HeightMapObservation(_GridObservation):
    """Terrain height and optional validity channel in CHW order."""

    value_field: ClassVar[str] = "height"
