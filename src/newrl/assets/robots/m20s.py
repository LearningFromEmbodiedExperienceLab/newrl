from dataclasses import replace
from pathlib import Path

import mujoco
from assetx import (
    Compose,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveGeoms,
    RemoveSceneSettings,
    RemoveSensors,
    Transform,
)

from ..cooking import cooked_robot, cooked_usd, robot_model_dir
from .config import ActuatorGroup, RobotConfig


DEEP_M20S = (
    "https://github.com/DeepRoboticsLab/deep_robotics_model/tree/"
    "a547a0289deb7cda23578b252b4d54debfdcec19/M20S/mjcf"
)


class _RemoveUnnamedSensors(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for sensor in list(spec.sensors):
            if not sensor.name:
                spec.delete(sensor)
        spec.compile()
        return replace(asset, spec=spec)


class _M20SCapsules(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and "_wheel" not in geom.name:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        return replace(asset, spec=spec)


def build_m20s(vendor_dir: Path) -> MujocoAsset:
    return Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        _RemoveUnnamedSensors(),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        RemoveGeoms(["floor"]),
        _M20SCapsules(),
    ]).transform(MujocoAsset.from_file(vendor_dir / "M20S.xml"))


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
