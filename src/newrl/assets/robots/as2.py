from pathlib import Path

from assetx import (
    Compose,
    GeomsToBody,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    ReplaceCylinderWithCapsule,
)

from ..cooking import cooked_robot, cooked_usd, robot_model_dir
from .config import ActuatorGroup, RobotConfig


UNITREE_AS2 = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "38a9b00ecbff0b70e85a7627eb9c65db9a2341f7/robots/as2_description"
)


_AS2_FEET = (
    ("FL", "FL_calf_collision3", "FL_calf_visual1"),
    ("FR", "FR_calf_collision3", "FR_calf_visual1"),
    ("RL", "RL_calf_collision3", "RL_calf_visual1"),
    ("RR", "RR_calf_collision3", "RR_calf_visual1"),
)


def build_as2(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*pos", ".*torque", "imu.*", ".*vel"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            *(
                GeomsToBody([collision, visual], f"{prefix}_foot", mass=0.05)
                for prefix, collision, visual in _AS2_FEET
            ),
            ReplaceCylinderWithCapsule(),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / "as2.xml"))


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
