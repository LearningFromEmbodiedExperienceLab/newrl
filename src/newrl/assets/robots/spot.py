from pathlib import Path

from assetx import (
    ApproximateWithCapsule,
    Compose,
    GeomsToBody,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    RenameBodies,
)

from ..cooking import cooked_robot, cooked_usd, robot_model_dir
from .config import ActuatorGroup, RobotConfig


MENAGERIE_SPOT = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/"
    "71f066ad0be9cd271f7ed58c030243ef157af9f4/boston_dynamics_spot"
)


_SPOT_LEGS = ("fl", "fr", "hl", "hr")


def build_spot(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            RenameBodies({"body": "base_link"}),
            *(
                GeomsToBody([leg.upper()], f"{leg}_foot", mass=0.05)
                for leg in _SPOT_LEGS
            ),
            ApproximateWithCapsule(
                [[f"{leg}_uleg_collision0"] for leg in _SPOT_LEGS],
                names=[f"{leg}_uleg_collision0" for leg in _SPOT_LEGS],
                replace=True,
                coverage_quantile=0.95,
                radius_scale=0.6,
            ),
            ApproximateWithCapsule(
                [[f"{leg}_lleg_collision0"] for leg in _SPOT_LEGS],
                names=[f"{leg}_lleg_collision0" for leg in _SPOT_LEGS],
                replace=True,
                coverage_quantile=0.95,
                radius_scale=0.9,
            ),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / "spot.xml"))


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
