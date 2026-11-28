import math
from pathlib import Path

from assetx import (
    Compose,
    EditBodies,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
)

from ..cooking import cooked_robot, cooked_usd, robot_model_dir
from .config import ActuatorGroup, RobotConfig


MJLAB_G1 = (
    "https://github.com/mujocolab/mjlab/tree/"
    "1f48001802bd89257537c2dc16979d41b9686e86/"
    "src/mjlab/asset_zoo/robots/unitree_g1/xmls"
)


def build_g1(vendor_dir: Path) -> MujocoAsset:
    return Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        # RobotConfig.position owns the root placement.
        EditBodies("pelvis", pos=[0.0, 0.0, 0.0]),
    ]).transform(MujocoAsset.from_file(vendor_dir / "g1.xml"))


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
            joint_names=names,
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
