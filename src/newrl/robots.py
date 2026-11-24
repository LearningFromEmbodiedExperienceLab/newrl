"""Newton robot asset factories and position actuator configuration."""

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import newton
import warp as wp
import yaml

from .assets import cooked_robot, cooked_usd
from .utils.string import resolve_scene_names

ValueSpec = float | dict[str, float]


@dataclass(frozen=True)
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

    def __post_init__(self):
        if self.viscous_damping is not None and (
            not math.isfinite(self.viscous_damping) or self.viscous_damping < 0
        ):
            raise ValueError("viscous_damping must be finite and non-negative")


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

    def __post_init__(self):
        if self.actuators and any(
            value is not None for value in (self.stiffness, self.damping, self.armature)
        ):
            raise ValueError("Pass either actuators or per-joint gain sequences, not both")
        if not self.actuators and any(
            value is None for value in (self.stiffness, self.damping, self.armature)
        ):
            raise ValueError("Actuator gains require actuators or explicit sequences")

    def build(self) -> newton.ModelBuilder:
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

    def _gains(self, labels: Sequence[str]):
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


def scalar_joints(builder):
    return [
        j
        for j, kind in enumerate(builder.joint_type)
        if kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
    ]


def robot_model_dir() -> Path:
    """Cooked robot bundles for this repository.

    ``ROBOT_MODEL_DIR`` overrides the default ``<repo>/.cache/robots``.
    """
    if env := os.environ.get("ROBOT_MODEL_DIR"):
        return Path(env)
    return Path(__file__).resolve().parents[2] / ".cache" / "robots"


def newton_anymal_c() -> RobotConfig:
    asset = Path(newton.utils.download_asset("anybotics_anymal_c"))
    with (asset / "rl_policies" / "anymal.yaml").open() as file:
        cfg = yaml.safe_load(file)
    return RobotConfig(
        asset=asset / "usd" / "anymal_c.usda",
        position=(0.0, 0.0, 0.76),
        joint_position=tuple(cfg["mjw_joint_pos"]),
        stiffness=tuple(cfg["mjw_joint_stiffness"]),
        damping=tuple(cfg["mjw_joint_damping"]),
        armature=tuple(cfg["mjw_joint_armature"]),
        action_scale=float(cfg["action_scale"]),
        joint_mirror={
            "LF_HAA": ("RF_HAA", -1),
            "RF_HAA": ("LF_HAA", -1),
            "LF_HFE": ("RF_HFE", 1),
            "RF_HFE": ("LF_HFE", 1),
            "LF_KFE": ("RF_KFE", 1),
            "RF_KFE": ("LF_KFE", 1),
            "LH_HAA": ("RH_HAA", -1),
            "RH_HAA": ("LH_HAA", -1),
            "LH_HFE": ("RH_HFE", 1),
            "RH_HFE": ("LH_HFE", 1),
            "LH_KFE": ("RH_KFE", 1),
            "RH_KFE": ("LH_KFE", 1),
        },
        body_mirror={
            "base": ("base", 1),
            "LF_FOOT": ("RF_FOOT", 1),
            "RF_FOOT": ("LF_FOOT", 1),
            "LH_FOOT": ("RH_FOOT", 1),
            "RH_FOOT": ("LH_FOOT", 1),
        },
    )


def assetx_g1() -> RobotConfig:
    """mjlab's 29-DoF G1 with its bent-knee pose and implicit motor gains."""
    bundle = cooked_robot("g1", robot_model_dir())
    # Reflected motor inertias from mjlab's unitree_g1/g1_constants.py.
    armature_5020 = 0.139e-4 * ((1 + 46 / 18) * 4.5) ** 2 + 0.017e-4 * 4.5**2 + 0.169e-4
    armature_7520_14 = (
        0.489e-4 * (4.5 * (1 + 48 / 22)) ** 2
        + 0.098e-4 * (1 + 48 / 22)**2 + 0.533e-4
    )
    armature_7520_22 = 0.489e-4 * (4.5 * 5)**2 + 0.109e-4 * 5**2 + 0.738e-4
    armature_4010 = 0.068e-4 * (5 * 5)**2
    frequency = 10 * 2 * math.pi
    actuators = tuple(
        ActuatorGroup(
            names,
            stiffness=armature * frequency**2,
            damping=4 * armature * frequency,
            armature=armature,
            effort_limit=effort,
            velocity_limit=speed,
        )
        for names, armature, effort, speed in (
            (
                (".*_elbow_joint", ".*_shoulder_(pitch|roll|yaw)_joint", ".*_wrist_roll_joint"),
                armature_5020, 25.0, 37.0,
            ),
            ((".*_hip_(pitch|yaw)_joint", "waist_yaw_joint"), armature_7520_14, 88.0, 32.0),
            ((".*_hip_roll_joint", ".*_knee_joint"), armature_7520_22, 139.0, 20.0),
            ((".*_wrist_(pitch|yaw)_joint",), armature_4010, 5.0, 22.0),
            # Parallel linkages use two 5020 motors per joint.
            (
                ("waist_(pitch|roll)_joint", ".*_ankle_(pitch|roll)_joint"),
                2 * armature_5020, 50.0, 37.0,
            ),
        )
    )
    return RobotConfig(
        asset=cooked_usd(bundle),
        position=(0.0, 0.0, 0.76),
        joint_position={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            ".*_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_(hip_roll|hip_yaw|ankle_roll|shoulder_yaw|wrist_roll|wrist_pitch|wrist_yaw)_joint": 0.0,
            "waist_(yaw|roll|pitch)_joint": 0.0,
        },
        actuators=actuators,
        action_scale={
            pattern: 0.25 * group.effort_limit / group.stiffness
            for group in actuators
            for pattern in group.joint_names
        },
    )


def assetx_spot() -> RobotConfig:
    # Gains and pose match the usual Spot implicit actuators.
    bundle = cooked_robot("spot", robot_model_dir())
    return RobotConfig(
        asset=cooked_usd(bundle),
        position=(0.0, 0.0, 0.60),
        joint_position={".*_hx": 0.0, ".*_hy": 1.04, ".*_kn": -1.8},
        actuators=(
            ActuatorGroup(
                (".*_hx", ".*_hy"),
                stiffness=60.0,
                damping=1.5,
                armature=0.01,
                effort_limit=45.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
            ActuatorGroup(
                (".*_kn",),
                stiffness=60.0,
                damping=1.5,
                armature=0.01,
                effort_limit=90.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
        ),
        action_scale=0.5,
        joint_mirror={
            "fl_hx": ("fr_hx", -1),
            "fr_hx": ("fl_hx", -1),
            "fl_hy": ("fr_hy", 1),
            "fr_hy": ("fl_hy", 1),
            "fl_kn": ("fr_kn", 1),
            "fr_kn": ("fl_kn", 1),
            "hl_hx": ("hr_hx", -1),
            "hr_hx": ("hl_hx", -1),
            "hl_hy": ("hr_hy", 1),
            "hr_hy": ("hl_hy", 1),
            "hl_kn": ("hr_kn", 1),
            "hr_kn": ("hl_kn", 1),
        },
        body_mirror={
            "base_link": ("base_link", 1),
            "fl_foot": ("fr_foot", 1),
            "fr_foot": ("fl_foot", 1),
            "hl_foot": ("hr_foot", 1),
            "hr_foot": ("hl_foot", 1),
        },
    )


def assetx_as2() -> RobotConfig:
    # Gains and pose match the usual AS2 implicit actuators.
    bundle = cooked_robot("as2", robot_model_dir())
    return RobotConfig(
        asset=cooked_usd(bundle),
        position=(0.0, 0.0, 0.5),
        joint_position={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.7,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        actuators=(
            ActuatorGroup(
                (".*_hip_joint",),
                stiffness=60.0,
                damping=1.5,
                armature=0.026,
                effort_limit=60.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
            ActuatorGroup(
                (".*_thigh_joint",),
                stiffness=60.0,
                damping=1.5,
                armature=0.026,
                effort_limit=60.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
            ActuatorGroup(
                (".*_calf_joint",),
                stiffness=60.0,
                damping=1.5,
                armature=0.038,
                effort_limit=90.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
        ),
        action_scale={".*_joint": 0.4},
        joint_mirror={
            "FL_hip_joint": ("FR_hip_joint", -1),
            "FR_hip_joint": ("FL_hip_joint", -1),
            "FL_thigh_joint": ("FR_thigh_joint", 1),
            "FR_thigh_joint": ("FL_thigh_joint", 1),
            "FL_calf_joint": ("FR_calf_joint", 1),
            "FR_calf_joint": ("FL_calf_joint", 1),
            "RL_hip_joint": ("RR_hip_joint", -1),
            "RR_hip_joint": ("RL_hip_joint", -1),
            "RL_thigh_joint": ("RR_thigh_joint", 1),
            "RR_thigh_joint": ("RL_thigh_joint", 1),
            "RL_calf_joint": ("RR_calf_joint", 1),
            "RR_calf_joint": ("RL_calf_joint", 1),
        },
        body_mirror={
            "base_link": ("base_link", 1),
            "FL_foot": ("FR_foot", 1),
            "FR_foot": ("FL_foot", 1),
            "RL_foot": ("RR_foot", 1),
            "RR_foot": ("RL_foot", 1),
        },
    )


def assetx_as2w(*, wheel_geometry: Literal["mesh", "sphere"] = "mesh") -> RobotConfig:
    recipe = {"mesh": "as2w", "sphere": "as2w_sphere"}[wheel_geometry]
    bundle = cooked_robot(recipe, robot_model_dir())
    return RobotConfig(
        asset=cooked_usd(bundle),
        position=(0.0, 0.0, 0.5),
        joint_position={
            ".*L_hip": 0.1,
            ".*R_hip": -0.1,
            "F[L,R]_thigh": 0.7,
            "R[L,R]_thigh": 0.8,
            ".*_calf": -1.5,
            ".*_foot": 0.0,
        },
        actuators=(
            ActuatorGroup((".*_hip", ".*_thigh"), 60.0, 1.5, 0.026, 60.0, 20.0, 0.01),
            ActuatorGroup((".*_calf",), 60.0, 1.5, 0.038, 90.0, 20.0, 0.01),
            ActuatorGroup((".*_foot",), 0.0, 1.5, 0.01, 15.0, 30.0, 0.01),
        ),
        action_scale={".*_(hip|thigh|calf)": 0.4, ".*_foot": 10.0},
        velocity_joints=(".*_foot",),
        joint_mirror={
            "FL_hip": ("FR_hip", -1),
            "FR_hip": ("FL_hip", -1),
            "FL_thigh": ("FR_thigh", 1),
            "FR_thigh": ("FL_thigh", 1),
            "FL_calf": ("FR_calf", 1),
            "FR_calf": ("FL_calf", 1),
            "FL_foot": ("FR_foot", 1),
            "FR_foot": ("FL_foot", 1),
            "RL_hip": ("RR_hip", -1),
            "RR_hip": ("RL_hip", -1),
            "RL_thigh": ("RR_thigh", 1),
            "RR_thigh": ("RL_thigh", 1),
            "RL_calf": ("RR_calf", 1),
            "RR_calf": ("RL_calf", 1),
            "RL_foot": ("RR_foot", 1),
            "RR_foot": ("RL_foot", 1),
        },
        body_mirror={
            "base_link": ("base_link", 1),
            "FL_foot": ("FR_foot", 1),
            "FR_foot": ("FL_foot", 1),
            "RL_foot": ("RR_foot", 1),
            "RR_foot": ("RL_foot", 1),
        },
    )


def assetx_as2w_sphere() -> RobotConfig:
    return assetx_as2w(wheel_geometry="sphere")


def assetx_m20s() -> RobotConfig:
    bundle = cooked_robot("m20s", robot_model_dir())
    return RobotConfig(
        asset=cooked_usd(bundle),
        position=(0.0, 0.0, 0.55),
        joint_position={
            ".*_hipx_joint": 0.0,
            "(fl|fr)_hipy_joint": 0.7,
            "(hl|hr)_hipy_joint": -0.7,
            "(fl|fr)_knee_joint": -1.4,
            "(hl|hr)_knee_joint": 1.4,
            ".*_wheel_joint": 0.0,
        },
        actuators=(
            ActuatorGroup((".*_(hipx|hipy|knee)_joint",), 60.0, 1.5, 0.02, 60.0, 20.0, 0.01),
            ActuatorGroup((".*_wheel_joint",), 0.0, 1.5, 0.01, 28.5, 30.0, 0.01),
        ),
        action_scale={".*_(hipx|hipy|knee)_joint": 0.4, ".*_wheel_joint": 10.0},
        velocity_joints=(".*_wheel_joint",),
        joint_mirror={
            "fl_hipx_joint": ("fr_hipx_joint", -1),
            "fr_hipx_joint": ("fl_hipx_joint", -1),
            "fl_hipy_joint": ("fr_hipy_joint", 1),
            "fr_hipy_joint": ("fl_hipy_joint", 1),
            "fl_knee_joint": ("fr_knee_joint", 1),
            "fr_knee_joint": ("fl_knee_joint", 1),
            "fl_wheel_joint": ("fr_wheel_joint", 1),
            "fr_wheel_joint": ("fl_wheel_joint", 1),
            "hl_hipx_joint": ("hr_hipx_joint", -1),
            "hr_hipx_joint": ("hl_hipx_joint", -1),
            "hl_hipy_joint": ("hr_hipy_joint", 1),
            "hr_hipy_joint": ("hl_hipy_joint", 1),
            "hl_knee_joint": ("hr_knee_joint", 1),
            "hr_knee_joint": ("hl_knee_joint", 1),
            "hl_wheel_joint": ("hr_wheel_joint", 1),
            "hr_wheel_joint": ("hl_wheel_joint", 1),
        },
        body_mirror={
            "base_link": ("base_link", 1),
            "fl_wheel": ("fr_wheel", 1),
            "fr_wheel": ("fl_wheel", 1),
            "hl_wheel": ("hr_wheel", 1),
            "hr_wheel": ("hl_wheel", 1),
        },
    )
