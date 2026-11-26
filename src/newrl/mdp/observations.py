from tensordict import TensorDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, Self, TYPE_CHECKING

import torch

from newrl.utils.config import runtime_field
from newrl.utils.math import quat_rotate_inverse, yaw_rotate
from newrl.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from newrl.scene import Scene


_POLAR_SIGNS = (1, -1, 1)
_AXIAL_SIGNS = (-1, 1, -1)


@dataclass(frozen=True)
class EntityField:
    entity: str
    field: str
    joint_names: str | Sequence[str] | None = field(default=None, kw_only=True)
    body_names: str | Sequence[str] | None = field(default=None, kw_only=True)
    preserve_order: bool = field(default=False, kw_only=True)
    _indices: torch.Tensor | None = runtime_field()
    _shape: torch.Size | None = runtime_field()

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, self.field)})

    def _initialize(self, scene: "Scene") -> Self:
        entity = scene.entities[self.entity]
        if self.field in entity.model_parameters:
            if self.joint_names is not None or self.body_names is not None:
                raise ValueError("Select randomization indices in the randomization config")
            return replace(self, _shape=entity.model_parameters[self.field].shape[1:])
        shapes = {
            "joint_pos": entity.joint_q_indices.shape[1:],
            "joint_vel": entity.joint_qd_indices.shape[1:],
            "joint_torque": entity.joint_qd_indices.shape[1:],
            "body_pose_w": torch.Size((len(entity.body_names), 7)),
            "body_twist_w": torch.Size((len(entity.body_names), 6)),
        }
        indices = None
        shape = shapes[self.field]
        if self.joint_names is not None and not self.field.startswith('joint_'):
            raise ValueError('joint_names requires a joint field')
        if self.body_names is not None and not self.field.startswith('body_'):
            raise ValueError('body_names requires a body field')
        names = self.joint_names if self.field.startswith('joint_') else self.body_names
        if names is not None:
            joint = self.field.startswith('joint_')
            find = entity.find_joints if joint else entity.find_bodies
            selected, _ = find(names, self.preserve_order)
            indices = torch.tensor(selected, device=(entity.joint_q_indices if joint else entity.body_indices).device, dtype=torch.long)
            shape = torch.Size((len(selected), *shape[1:]))
        return replace(self, _shape=shape, _indices=indices)

    @property
    def shape(self) -> torch.Size:
        """Shape of one world's observation, excluding the batch dimension."""
        if self._shape is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return self._shape

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        values = physical_state[self.entity, self.field]
        if self._indices is None:
            return values
        return values.index_select(-1 if self.field.startswith('joint_') else -2, self._indices)

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        if self.entity not in scene.entities or self.field not in ("joint_pos", "joint_vel", "joint_torque"):
            raise ValueError(f"No reflection defined for entity field {self.field!r}")
        entity = scene.entities[self.entity]
        if self._indices is None:
            return entity.joint_symmetry_transform()
        labels = [entity.joint_names[index] for index in self._indices.tolist()]
        return SymmetryTransform.from_names(labels, entity.joint_mirror)


@dataclass(frozen=True)
class PreviousAction:
    physical_fields: ClassVar[frozenset] = frozenset()
    _size: int | None = runtime_field()

    def _initialize(self, scene: "Scene") -> Self:
        return replace(self, _size=sum(entity.joint_q_indices.shape[1] for entity in scene.robots.values()))

    @property
    def shape(self) -> torch.Size:
        if self._size is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return torch.Size((self._size,))

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return mdp_state["previous_action"]

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
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
    preserve_order: bool = field(default=False, kw_only=True)
    _indices: torch.Tensor | None = runtime_field()
    _body_indices: tuple[int, ...] = runtime_field(())
    _components: ClassVar[tuple[int, ...]] = (3,)
    _mirror_signs: ClassVar[tuple[int, ...] | None] = None

    @property
    def shape(self) -> torch.Size:
        if self._indices is None:
            raise RuntimeError(
                "Initialize the observation against a scene before reading its shape"
            )
        return torch.Size((self._indices.numel(), *self._components))

    def _initialize(self, scene: "Scene") -> Self:
        entity = scene.entities[self.entity]
        names = (
            (self.body_names,) if isinstance(self.body_names, str) else self.body_names
        )
        if not names:
            raise ValueError(f"Empty body selection for {self.entity!r}")
        try:
            selected, _ = entity.find_bodies(names, self.preserve_order)
        except ValueError as exc:
            raise ValueError(f"Invalid body selection {names!r} on {self.entity!r}: {exc}") from exc

        body_indices = tuple(selected)
        indices = torch.tensor(
            body_indices, device=entity.body_indices.device, dtype=torch.long
        )
        return replace(self, _body_indices=body_indices, _indices=indices)

    def _read(self, physical_state: TensorDict, field_name: str) -> torch.Tensor:
        if self._indices is None:
            raise RuntimeError(
                "Initialize body observations against a scene before evaluation"
            )
        return physical_state[self.entity, field_name].index_select(
            physical_state.ndim, self._indices
        )

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        if self._mirror_signs is None:
            raise ValueError(f"No reflection defined for {type(self).__name__}")
        return scene.entities[self.entity].body_symmetry_transform(
            self._body_indices, self._mirror_signs
        )


@dataclass(frozen=True)
class _BodyVelocity(_BodyObservation):
    frame: Literal["world", "body", "yaw"] = "world"

    def __post_init__(self) -> None:
        if self.frame not in ("world", "body", "yaw"):
            raise ValueError(f"Unknown velocity frame {self.frame!r}")

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        fields = {(self.entity, "body_twist_w")}
        if self.frame != "world":
            fields.add((self.entity, "body_pose_w"))
        return frozenset(fields)

    def _velocity(self, physical_state: TensorDict, components: slice) -> torch.Tensor:
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

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return self._velocity(physical_state, slice(0, 3))


class BodyAngularVelocity(_BodyVelocity):
    _mirror_signs = _AXIAL_SIGNS

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return self._velocity(physical_state, slice(3, 6))


class ProjectedGravity(_BodyObservation):
    _mirror_signs = _POLAR_SIGNS

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, "projected_gravity")})

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return self._read(physical_state, "projected_gravity")


class BodyHeight(_BodyObservation):
    _components = ()
    _mirror_signs = (1,)

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, "body_pose_w")})

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return self._read(physical_state, "body_pose_w")[..., 2]


@dataclass(frozen=True)
class BodyPosition(_BodyObservation):
    """Selected body origins expressed relative to a root in its full frame."""

    root_body: str = "pelvis"
    _root: int | None = runtime_field()
    _mirror_signs = _POLAR_SIGNS

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, "body_pose_w")})

    def _initialize(self, scene: "Scene") -> Self:
        bound = super()._initialize(scene)
        (root,), _ = scene.entities[self.entity].find_bodies(self.root_body)
        return replace(bound, _root=root)

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        pose = self._read(physical_state, "body_pose_w")
        root = physical_state[self.entity, 'body_pose_w'][..., self._root, :]
        return quat_rotate_inverse(
            root[..., None, 3:].expand_as(pose[..., 3:]),
            pose[..., :3] - root[..., None, :3],
        )


@dataclass(frozen=True)
class VelocityCommandObservation:
    physical_fields: ClassVar[frozenset] = frozenset()

    @property
    def shape(self) -> torch.Size:
        return torch.Size((3,))

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return mdp_state["velocity_command"]

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        return SymmetryTransform((0, 1, 2), (1, -1, -1))


velocity_command = VelocityCommandObservation()


@dataclass(frozen=True)
class ContactObservation:
    """Read a configured contact sensor field as floating-point features."""

    sensor: str
    _shape: torch.Size | None = runtime_field()
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
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.sensor, self.field)})

    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            raise RuntimeError(
                "Bind the contact observation to a scene before reading its shape"
            )
        return self._shape

    def _initialize(self, scene: "Scene") -> Self:
        scene.resolve_physical_fields(self.physical_fields)
        count = scene.sensors[self.sensor].body_ids.shape[1]
        shape = (count, 3) if self.field in ("force_w", "last_contact_pos_w") else (count,)
        return replace(self, _shape=torch.Size(shape))

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return physical_state[self.sensor, self.field].float()

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        sensor = scene.sensors[self.sensor]
        entity = scene.entities[sensor.config.entity]
        indices, _ = entity.find_bodies(sensor.config.body_names)
        signs = _POLAR_SIGNS if self.field in ("force_w", "last_contact_pos_w") else (1,)
        return entity.body_symmetry_transform(indices, signs)


@dataclass(frozen=True)
class _GridObservation:
    sensor: str
    include_valid: bool = True
    _grid_shape: torch.Size | None = runtime_field()
    value_field: ClassVar[str]

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        fields = {(self.sensor, self.value_field)}
        if self.include_valid:
            fields.add((self.sensor, "valid"))
        return frozenset(fields)

    @property
    def shape(self) -> torch.Size:
        if self._grid_shape is None:
            raise RuntimeError("Bind the grid observation to a scene before reading its shape")
        return torch.Size((2 if self.include_valid else 1, *self._grid_shape))

    def _initialize(self, scene: "Scene") -> Self:
        scene.resolve_physical_fields(self.physical_fields)
        return replace(self, _grid_shape=scene.sensors[self.sensor].shape)

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        values = physical_state[self.sensor, self.value_field]
        if self.include_valid:
            valid = physical_state[self.sensor, "valid"].to(values.dtype)
            return torch.stack((values, valid), dim=-3)
        return values.unsqueeze(-3)

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        sensor = scene.sensors[self.sensor]
        config = sensor.config
        entity = scene.entities[config.entity]
        indices, _ = entity.find_bodies(config.body)
        entity.body_symmetry_transform(indices, (1,))
        permutation = torch.arange(self.shape.numel()).reshape(self.shape).flip(-1)
        return SymmetryTransform(permutation.flatten().tolist(), (1,) * permutation.numel())


@dataclass(frozen=True)
class DepthCameraObservation(_GridObservation):
    """Camera depth and optional validity channel in CHW order."""

    value_field: ClassVar[str] = "depth"

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        config = scene.sensors[self.sensor].config
        roll, _, yaw = config.rpy
        if any(abs(value) > 1e-7 for value in (config.pos[1], roll, yaw)):
            raise ValueError("Depth mirroring requires a centered camera with zero roll and yaw")
        return super().symmetry_transform(scene)


@dataclass(frozen=True)
class RGBCameraObservation:
    """Camera RGB in uint8 CHW order before observation grouping."""

    sensor: str
    _shape: torch.Size | None = runtime_field()

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.sensor, "rgb")})

    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            raise RuntimeError("Bind the RGB observation to a scene before reading its shape")
        return self._shape

    def _initialize(self, scene: "Scene") -> Self:
        scene.resolve_physical_fields(self.physical_fields)
        return replace(self, _shape=torch.Size((3, *scene.sensors[self.sensor].shape)))

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        return physical_state[self.sensor, "rgb"].movedim(-1, -3)

    def symmetry_transform(self, scene: "Scene") -> SymmetryTransform:
        sensor = scene.sensors[self.sensor]
        config = sensor.config
        roll, _, yaw = config.rpy
        if any(abs(value) > 1e-7 for value in (config.pos[1], roll, yaw)):
            raise ValueError("RGB mirroring requires a centered camera with zero roll and yaw")
        entity = scene.entities[config.entity]
        indices, _ = entity.find_bodies(config.body)
        entity.body_symmetry_transform(indices, (1,))
        permutation = torch.arange(self.shape.numel()).reshape(self.shape).flip(-1)
        return SymmetryTransform(permutation.flatten().tolist(), (1,) * permutation.numel())


@dataclass(frozen=True)
class HeightMapObservation(_GridObservation):
    """Terrain height and optional validity channel in CHW order."""

    value_field: ClassVar[str] = "height"


@dataclass(frozen=True)
class ReferenceMotion:
    """Current reference root and future root/joints in the robot root frame."""

    entity: str = "robot"
    root_body: str = "pelvis"
    goal_frames: int = 10
    _root: int | None = runtime_field()
    _joints: int | None = runtime_field()

    def __post_init__(self) -> None:
        if self.goal_frames < 1:
            raise ValueError("goal_frames must be positive")

    @property
    def physical_fields(self) -> frozenset[tuple[str, str]]:
        return frozenset({(self.entity, "body_pose_w")})

    @property
    def shape(self) -> torch.Size:
        if self._joints is None:
            raise RuntimeError("Bind ReferenceMotion before reading its shape")
        return torch.Size((9 + self.goal_frames * (9 + self._joints),))

    def _initialize(self, scene: "Scene") -> Self:
        robot = scene.entities[self.entity]
        (root,), _ = robot.find_bodies(self.root_body)
        return replace(self, _root=root, _joints=robot.joint_q_indices.shape[-1])

    def __call__(self, physical_state: TensorDict, mdp_state: TensorDict) -> torch.Tensor:
        from newrl.utils.math import quat_conjugate, quat_mul, quat_to_6d

        if self._root is None:
            raise RuntimeError("Bind ReferenceMotion before evaluating it")
        root = physical_state[self.entity, "body_pose_w"][..., self._root, :]
        reference = mdp_state["reference", "root_pose"]
        current = torch.cat((
            quat_rotate_inverse(root[..., 3:], reference[..., :3] - root[..., :3]),
            quat_to_6d(quat_mul(quat_conjugate(root[..., 3:]), reference[..., 3:])),
        ), -1)
        future = mdp_state["future_reference"]
        if future.batch_size[-1] != self.goal_frames:
            raise ValueError("ReferenceMotion goal_frames differs from MotionTrackingMDP")
        q = root[..., None, 3:].expand_as(future["root_pose"][..., 3:])
        goals = torch.cat((
            quat_rotate_inverse(q, future["root_pose"][..., :3] - root[..., None, :3]),
            quat_to_6d(quat_mul(quat_conjugate(q), future["root_pose"][..., 3:])),
            future["joint_pos"],
        ), -1).flatten(-2)
        return torch.cat((current, goals), -1)
