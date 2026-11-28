"""Newton robot configuration and actuator binding."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import newton
import warp as wp

from newrl.utils.string import resolve_scene_names


ValueSpec = float | dict[str, float]


@dataclass(frozen=True, kw_only=True)
class ActuatorGroup:
    """Gains for the joints selected by ``joint_names``.

    Patterns full-match a joint label or its final path component.
    """

    joint_names: tuple[str, ...]
    stiffness: float
    damping: float
    armature: float
    effort_limit: float | None = None
    velocity_limit: float | None = None
    friction: float | None = None
    viscous_damping: float | None = None
    """Passive velocity damping; None preserves the imported asset value.

    ``damping`` above is the active PD derivative gain.
    """

    def __post_init__(self) -> None:
        if self.viscous_damping is not None and (
            not math.isfinite(self.viscous_damping) or self.viscous_damping < 0
        ):
            raise ValueError("viscous_damping must be finite and non-negative")


@dataclass(frozen=True)
class RoundedWheel:
    """Disk swept by a sphere of radius half_width, spinning around body Y.

    The support distance along a unit direction n is half_width plus
    (radius - half_width) times the length of n projected into the disk.
    Replace the body's collision shape in place to preserve material, sensor
    IDs, visual shapes and imported rigid-body inertia.
    """

    bodies: tuple[str, ...]
    radius: float
    half_width: float
    center_y: float = 0.0

    def apply(self, builder: newton.ModelBuilder, *, mesh_segments: int | None) -> None:
        if not 0 < self.half_width < self.radius:
            raise ValueError("wheel dimensions require 0 < half_width < radius")
        core = self.radius - self.half_width
        mesh = None
        if mesh_segments is not None:
            mesh = newton.Mesh.create_cylinder(
                core, self.half_width, barrel_radius=self.half_width, up_axis=newton.Axis.Z,
                segments=mesh_segments, compute_normals=False, compute_uvs=False, compute_inertia=False,
            )
            mesh.maxhullvert = len(mesh.vertices)
        for body in resolve_scene_names(self.bodies, builder.body_label):
            (shape,) = [
                i for i in builder.body_shapes[body]
                if builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
            ]
            builder.shape_type[shape] = newton.GeoType.CYLINDER if mesh is None else newton.GeoType.CONVEX_MESH
            builder.shape_source[shape] = mesh
            builder.shape_scale[shape] = (
                (core, self.half_width, self.half_width) if mesh is None else (1.0, 1.0, 1.0)
            )
            builder.shape_transform[shape] = wp.transform(
                wp.vec3(0.0, self.center_y, 0.0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2),
            )
            builder.shape_collision_radius[shape] = self.radius


@dataclass(frozen=True)
class RobotConfig:
    asset: Path
    position: tuple[float, float, float]
    joint_position: tuple[float, ...] | dict[str, float]
    action_scale: ValueSpec
    stiffness: tuple[float, ...] | dict[str, float] | None = None
    damping: tuple[float, ...] | dict[str, float] | None = None
    armature: tuple[float, ...] | dict[str, float] | None = None
    actuators: tuple[ActuatorGroup, ...] = ()
    default_armature: float = 0.1
    joint_limit_stiffness: float = 1.0e3
    joint_limit_damping: float = 1.0e1
    contact_stiffness: float = 5.0e4
    contact_damping: float = 5.0e2
    contact_friction_stiffness: float = 1.0e3
    contact_friction: float = 0.75
    joint_mirror: dict[str, tuple[str, int]] = field(default_factory=dict)
    body_mirror: dict[str, tuple[str, int]] = field(default_factory=dict)
    velocity_joints: tuple[str, ...] = ()
    wheels: tuple[RoundedWheel, ...] = ()

    def __post_init__(self) -> None:
        if self.actuators and any(
            value is not None for value in (self.stiffness, self.damping, self.armature)
        ):
            raise ValueError("Pass either actuators or per-joint gain sequences, not both")
        if not self.actuators and any(
            value is None for value in (self.stiffness, self.damping, self.armature)
        ):
            raise ValueError("Actuator gains require actuators or explicit sequences")

    def build(self, *, wheel_mesh_segments: int | None = None) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=self.default_armature,
            limit_ke=self.joint_limit_stiffness,
            limit_kd=self.joint_limit_damping,
        )
        builder.default_shape_cfg.ke = self.contact_stiffness
        builder.default_shape_cfg.kd = self.contact_damping
        builder.default_shape_cfg.kf = self.contact_friction_stiffness
        builder.default_shape_cfg.mu = self.contact_friction
        if not self.asset.is_file():
            raise FileNotFoundError(f"Missing robot asset {self.asset}")
        builder.add_usd(
            str(self.asset),
            xform=wp.transform(wp.vec3(*self.position), wp.quat_identity()),
            joint_ordering="dfs",
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        for wheel in self.wheels:
            wheel.apply(builder, mesh_segments=wheel_mesh_segments)
        joints = scalar_joints(builder)
        labels = [builder.joint_label[joint] for joint in joints]
        position = _resolve_values(self.joint_position, labels, "joint_position")
        stiffness, damping, armature, effort, velocity, friction, viscous = self._gains(labels)
        velocity_indices = set()
        if self.velocity_joints:
            velocity_indices = set(resolve_scene_names(self.velocity_joints, labels))
        values = zip(
            joints, position, stiffness, damping, armature, effort, velocity, friction, viscous,
            strict=True,
        )
        for index, (joint, pos, ke, kd, arm, limit, speed, joint_friction, passive_damping) in enumerate(values):
            q = builder.joint_q_start[joint]
            dof = builder.joint_qd_start[joint]
            builder.joint_q[q] = pos
            builder.joint_target_ke[dof] = ke
            builder.joint_target_kd[dof] = kd
            builder.joint_armature[dof] = arm
            mode = newton.JointTargetMode.POSITION
            if index in velocity_indices and ke != 0:
                raise ValueError(f"Velocity joint {labels[index]!r} requires zero stiffness")
            if index in velocity_indices:
                mode = newton.JointTargetMode.VELOCITY
            builder.joint_target_mode[dof] = int(mode)
            if limit is not None:
                builder.joint_effort_limit[dof] = limit
            if speed is not None:
                builder.joint_velocity_limit[dof] = speed
            if joint_friction is not None:
                builder.joint_friction[dof] = joint_friction
            if passive_damping is not None:
                builder.joint_damping[dof] = passive_damping
        return builder

    def action_scales(self, labels: Sequence[str]) -> tuple[float, ...]:
        return tuple(_resolve_values(self.action_scale, labels, "action_scale"))

    def _gains(self, labels: Sequence[str]) -> tuple[list[float | None], ...]:
        if not self.actuators:
            return (
                _resolve_values(self.stiffness, labels, "stiffness"),
                _resolve_values(self.damping, labels, "damping"),
                _resolve_values(self.armature, labels, "armature"),
                [None] * len(labels),
                [None] * len(labels),
                [None] * len(labels),
                [None] * len(labels),
            )
        stiffness = [None] * len(labels)
        damping = [None] * len(labels)
        armature = [None] * len(labels)
        effort = [None] * len(labels)
        velocity = [None] * len(labels)
        friction = [None] * len(labels)
        viscous = [None] * len(labels)
        claimed = [None] * len(labels)
        for group in self.actuators:
            for index in resolve_scene_names(group.joint_names, labels):
                if claimed[index] is not None:
                    raise ValueError(
                        f"Joint {labels[index]!r} matches both {claimed[index]!r} and {group.joint_names!r}"
                    )
                claimed[index] = group.joint_names
                stiffness[index] = group.stiffness
                damping[index] = group.damping
                armature[index] = group.armature
                effort[index] = group.effort_limit
                velocity[index] = group.velocity_limit
                friction[index] = group.friction
                viscous[index] = group.viscous_damping
        missing = [label for label, owner in zip(labels, claimed) if owner is None]
        if missing:
            raise ValueError(f"Actuator groups do not cover {missing}")
        return stiffness, damping, armature, effort, velocity, friction, viscous


def _resolve_values(spec: ValueSpec, labels: Sequence[str], name: str) -> list[float]:
    if isinstance(spec, (int, float)):
        return [float(spec)] * len(labels)
    if isinstance(spec, tuple):
        if len(spec) != len(labels):
            raise ValueError(f"Expected {len(labels)} {name} values, got {len(spec)}")
        return [float(value) for value in spec]
    assigned = [None] * len(labels)
    claimed = [None] * len(labels)
    for pattern, value in spec.items():
        for index in resolve_scene_names(pattern, labels):
            if claimed[index] is not None:
                raise ValueError(
                    f"{name} pattern {pattern!r} overlaps {claimed[index]!r} on {labels[index]!r}"
                )
            claimed[index] = pattern
            assigned[index] = float(value)
    missing = [label for label, value in zip(labels, assigned) if value is None]
    if missing:
        raise ValueError(f"{name} patterns do not cover {missing}")
    return assigned


def scalar_joints(builder: newton.ModelBuilder) -> list[int]:
    return [
        j
        for j, kind in enumerate(builder.joint_type)
        if kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
    ]
