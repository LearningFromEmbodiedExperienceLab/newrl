"""Newton robot asset factories and position actuator configuration."""

from dataclasses import dataclass
from pathlib import Path

import newton
import warp as wp
import yaml


@dataclass(frozen=True)
class RobotConfig:
    asset: Path
    position: tuple[float, float, float]
    joint_position: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    armature: tuple[float, ...]
    action_scale: float
    default_armature: float = 0.1
    joint_limit_stiffness: float = 1.0e3
    joint_limit_damping: float = 1.0e1
    contact_stiffness: float = 5.0e4
    contact_damping: float = 5.0e2
    contact_friction_stiffness: float = 1.0e3
    contact_friction: float = 0.75

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
        builder.add_usd(
            str(self.asset),
            xform=wp.transform(wp.vec3(*self.position), wp.quat_identity()),
            joint_ordering="dfs",
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        joints = scalar_joints(builder)
        params = (self.joint_position, self.stiffness, self.damping, self.armature)
        if any(len(values) != len(joints) for values in params):
            raise ValueError(
                f"Expected {len(joints)} position actuator settings for {self.asset}"
            )
        for joint, pos, ke, kd, armature in zip(joints, *params, strict=True):
            q = builder.joint_q_start[joint]
            dof = builder.joint_qd_start[joint]
            builder.joint_q[q] = pos
            builder.joint_target_ke[dof] = ke
            builder.joint_target_kd[dof] = kd
            builder.joint_armature[dof] = armature
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        return builder


def scalar_joints(builder):
    return [
        j
        for j, kind in enumerate(builder.joint_type)
        if kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
    ]


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
    )


def assetx_spot() -> RobotConfig:
    raise NotImplementedError(
        "The AssetX Spot asset and actuator settings are pending."
    )


def assetx_as2() -> RobotConfig:
    raise NotImplementedError("The AssetX AS2 asset and actuator settings are pending.")
