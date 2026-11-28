import tempfile
from typing import Literal
from pathlib import Path

from assetx import (
    AddJoint,
    ApproximateWithCapsule,
    Compose,
    JointCfg,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
)
from assetx.conversion.urdf2mjcf import urdf_to_mjcf

from ..cooking import cooked_robot, cooked_usd, robot_model_dir
from .config import ActuatorGroup, RobotConfig, RoundedWheel


UNITREE_AS2W = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48/robots/as2w_description"
)


def build_as2w(vendor_dir: Path, *, sphere_wheels: bool = False) -> MujocoAsset:
    xml = Path(tempfile.mkdtemp(prefix="newrl-as2w-")) / "as2w.xml"
    urdf_to_mjcf(vendor_dir / "as2w.urdf", out_xml=xml)
    transforms = [
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        AddJoint("base_link", JointCfg(name="floating_base_joint", type="free")),
        ApproximateWithCapsule(
            [[f"{leg}_calf_collision0"] for leg in ("FL", "FR", "RL", "RR")],
            names=[f"{leg}_calf_collision0" for leg in ("FL", "FR", "RL", "RR")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.9,
        ),
    ]
    if sphere_wheels:
        from assetx import ApproximateWithSphere

        wheels = [f"{leg}_foot_collision0" for leg in ("FL", "FR", "RL", "RR")]
        transforms.append(
            ApproximateWithSphere([[wheel] for wheel in wheels], names=wheels, replace=True)
        )
    return Compose(transforms).transform(MujocoAsset.from_file(xml))


def build_as2w_sphere(vendor_dir: Path) -> MujocoAsset:
    return build_as2w(vendor_dir, sphere_wheels=True)


def assetx_as2w(*, wheel_geometry: Literal["rounded", "mesh", "sphere"] = "rounded") -> RobotConfig:
    recipe = {"rounded": "as2w", "mesh": "as2w", "sphere": "as2w_sphere"}[wheel_geometry]
    bundle = cooked_robot(recipe, robot_model_dir())
    return RobotConfig(
        asset=cooked_usd(bundle),
        wheels=(
            RoundedWheel(("[FR]L_foot",), 0.086, 0.026, 0.0546),
            RoundedWheel(("[FR]R_foot",), 0.086, 0.026, -0.0546),
        ) if wheel_geometry == "rounded" else (),
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
            ActuatorGroup(
                joint_names=(".*_hip", ".*_thigh"),
                stiffness=60.0,
                damping=1.5,
                armature=0.026,
                effort_limit=60.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
            ActuatorGroup(
                joint_names=(".*_calf",),
                stiffness=60.0,
                damping=1.5,
                armature=0.038,
                effort_limit=90.0,
                velocity_limit=20.0,
                friction=0.01,
            ),
            ActuatorGroup(
                joint_names=(".*_foot",),
                stiffness=0.0,
                damping=1.5,
                armature=0.01,
                effort_limit=15.0,
                velocity_limit=30.0,
                friction=0.01,
            ),
        ),
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
